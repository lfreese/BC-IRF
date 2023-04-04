#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --mem=MaxMemPerNode
#SBATCH --cpus-per-task=2


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import pandas as pd
from geopy.geocoders import Nominatim
from matplotlib.colors import SymLogNorm
from matplotlib.pyplot import cm

import xesmf as xe
#from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import dask
import utils

import geopandas
import argparse


# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--start_year', type=int, required=True)
parser.add_argument('--end_year', type=int, required=True)

# Parse the argument
args = parser.parse_args()
# check it worked
print('Start year', args.start_year, 'End year', args.end_year, )

# In[2]:


#import data from:
### Gallagher, Kevin P. (2021), “China’s Global Energy Finance,” Global Development Policy Center, Boston University.
### Gallagher, Kevin P., Li, Zhongshu, Chen, Xu, Ma, Xinyue (2019), “China’s Global Power Database,” Global Development Policy Center, Boston University.


# In[3]:


CGP_df = pd.read_excel('BU_data/CGP-Database_2020-1-2.xlsx')


# In[4]:


CGP_df = CGP_df.loc[CGP_df['Technology'] == 'Coal']


# In[5]:


#CGEF_df['Country'] = CGEF_df['Country'].str.lower()
CGP_df['Country'] = CGP_df['Country'].str.lower()


# In[6]:


## https://pubs.acs.org/doi/abs/10.1021/es3003684 Table S1 
EF = np.exp(-3.64) #g/kg BC/coal

## https://www.nap.edu/read/9736/chapter/8 
HHF = 22.51 #GJ/t

#conversion factors
GJ_to_MwH = .28

Mw_to_MwH = 24 #daily

ton_to_kg = 0.001 #metric tons

MW_gpDay = Mw_to_MwH/GJ_to_MwH/HHF*ton_to_kg*EF #g/day


# In[7]:


years = 50
coal_year_range = np.arange( args.start_year, args.end_year)


# In[8]:


## Add time dimension
length_simulation = years*365

time_array = np.arange(0, length_simulation)


# In[9]:


CGP_df = CGP_df.loc[CGP_df['Region'] == 'Southeast Asia']


# In[10]:


CGP_df.loc[:,'BC (g/day)'] = CGP_df['Capacity (MW)']*MW_gpDay


# In[11]:


CGP_df.columns = CGP_df.columns.str.replace(' ', '_')


# In[12]:


CGP_df.loc[CGP_df.Year_of_Commission == 'Pending', 'Year_of_Commission'] = 9999


# In[13]:


CGP_df['CO2_weighted_capacity_1000tonsperMW'] = CGP_df['Estimated_Annual_CO2_Emission_from_Power_Generation_(1000_ton)']/CGP_df['Capacity_(MW)']


# In[14]:


CGP_op = CGP_df.loc[CGP_df['Project_Status'] == 'In Operation']


# In[44]:


print('Emis data prepped and loaded')


# # Create Scenario

# In[16]:


#######in progress retiring by year after co2#########
def early_retirement_by_CO2_year(year_early, df, CO2_val, time_array, shutdown_years):
    ''' df must have a variable 'Year_of_Commission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day'''
    min_comission_yr = df['Year_of_Commission'].min()
    shutdown_days = shutdown_years*365
    E = np.zeros(len(time_array))
        #print(year_comis)
    #print(CGP_op.loc[CGP_op.Year_of_Commission == year_comis]['BC_(g/day)'].sum())
    test_array = np.where(time_array <= year_early*365, True, False)
    #plt.plot(test_array)
    E += test_array* df.loc[df.CO2_weighted_capacity_1000tonsperMW >= CO2_val]['BC_(g/day)'].sum()
        #fig, ax = plt.subplots()
        #plt.plot(E[year])
    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):
        #print(year_comis)
        #print(CGP_op.loc[CGP_op.Year_of_Commission == year_comis]['BC_(g/day)'].sum())
        test_array = np.where((time_array <= (year_comis-min_comission_yr)*365 + shutdown_days), True, False)
        #fig, ax = plt.subplots()
        #plt.plot(test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum())
        E += test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum()
        #E[year] += (time_array>=0) * df.loc[df.CO2_weighted_capacity_1000tonsperMW < CO2_val]['BC_(g/day)'].sum()
        #plt.plot(E)

    
    return(E)


# In[18]:


percent = np.arange(1,101)


# In[19]:


E_CO2_all_opts = {}
for year in coal_year_range:
    E_CO2_all_opts[year] = {}
    for r in percent:
        E_CO2_all_opts[year][r] = early_retirement_by_CO2_year(year, CGP_op, np.percentile(CGP_op['CO2_weighted_capacity_1000tonsperMW'],r), time_array, 40)
print('Emis profiiles created')


# # Convolve with G

# In[20]:


ds_area = xr.open_dataset('/net/fs11/d0/emfreese/GCrundirs/IRF_runs/RRTMG_pulse/SEA/Jan/mod_output/GEOSChem.SpeciesConc.20160101_0000z.nc4', engine = 'netcdf4')


# In[21]:


#import the green's function and set our time step
G = xr.open_dataarray('Outputs/G_SEA_Jan_BC_total.nc4')
dt = 1 #day


# In[22]:


G_lev0 = G.isel(lev = 0).sel().compute()
print('G prepped')




country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
countries = ['China','India','Indonesia','Malaysia','Vietnam','Australia', 'Cambodia','Myanmar', 'Laos','Philippines','Nepal','Bangladesh','Thailand','Bhutan']
print('countries uploaded')


for c in countries:
    loc_df = pd.DataFrame(columns = percent, index = coal_year_range)
    mask = country_mask.mask(ds_area, lon_name = 'lon', lat_name = 'lat')
    contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(c))
    for yr in coal_year_range:
        for pc in percent:
            print(yr,pc)
            loc_df[pc].loc[yr] = utils.convolve_single_lev(G_lev0 , E_CO2_all_opts[yr][pc], dt).sum(dim = 's').where(contiguous_mask).weighted(ds_area['area']).mean().values
            print(c + 'done')
            loc_df.to_csv(f'Outputs/{c}_co2_year_shutdown_{args.start_year}_{args.end_year}.csv')
            print(c + 'saved')

