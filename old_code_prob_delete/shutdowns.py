#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12



import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import xesmf as xe
import cartopy.crs as ccrs
import cartopy.feature as cfeat

import utils


# In[2]:


#import data from:
### Gallagher, Kevin P. (2021), “China’s Global Energy Finance,” Global Development Policy Center, Boston University.
### Gallagher, Kevin P., Li, Zhongshu, Chen, Xu, Ma, Xinyue (2019), “China’s Global Power Database,” Global Development Policy Center, Boston University.


CGP_df = pd.read_excel('BU_data/CGP-Database_2020-1-2.xlsx')
print('data loaded')

CGP_df = CGP_df.loc[CGP_df['Technology'] == 'Coal']
CGP_df['Country'] = CGP_df['Country'].str.lower()
print('data modified')


## https://pubs.acs.org/doi/abs/10.1021/es3003684 Table S1 
EF = np.exp(-3.64) #g/kg BC/coal

## https://www.nap.edu/read/9736/chapter/8 
HHF = 22.51 #GJ/t

#conversion factors
GJ_to_MwH = .28

Mw_to_MwH = 24 #daily

ton_to_kg = 0.001 #metric tons

MW_gpDay = Mw_to_MwH/GJ_to_MwH/HHF*ton_to_kg*EF #g/day

CGP_df = CGP_df.loc[CGP_df['Region'] == 'Southeast Asia']
CGP_df.loc[:,'BC (g/day)'] = CGP_df['Capacity (MW)']*MW_gpDay
print('BC applied to data')



CGP_df.columns = CGP_df.columns.str.replace(' ', '_')
CGP_df.loc[CGP_df.Year_of_Commission == 'Pending', 'Year_of_Commission'] = 9999


#### Create Scenarios

## Add time dimension
length_simulation = 30*365

time_array = np.arange(0, length_simulation)


# ## Scenario 1: Commission Year

E_comis = {}
for yr in np.arange(0,30):
    utils.shutdown_early_commission_year(yr, time_array, 2016, E_comis, CGP_df)


# ## Scenario 2: Lender

E_lender = {}
for yr in np.arange(0,30):
    utils.shutdown_by_lender(yr, time_array, 'CDB', E_lender, CGP_df)
for yr in np.arange(0,30):
    utils.shutdown_by_lender(yr, time_array, 'Ex-Im Bank', E_lender, CGP_df)
for yr in np.arange(0,30):
    utils.shutdown_by_lender(yr, time_array, 'CDB-Ex-Im Cofinancing', E_lender, CGP_df)


# ## Scenario 3: Country

E_country = {}
for yr in np.arange(0,30):
    for country in ['vietnam','cambodia','malaysia','indonesia','singapore']:
        utils.shutdown_by_country(yr, time_array, country, E_country, CGP_df)


# ## TO DO Scenario 4: Operating Status

# ## Plots of scenarios

# In[16]:


fig, ax = plt.subplots(figsize = [8,5])
for scen in E_country.keys():
    if '_1yrs' in scen:
        if 'vietnam' in scen:
            plt.plot(E_country[scen], c = 'red', linestyle = '--')
            ax.text(400, E_country[scen].min() + .02, 'Vietnam')
        if 'singapore' in scen:
            plt.plot(E_country[scen], c = 'cyan', linestyle = '--')
            ax.text(440, E_country[scen].min() + .02, 'Singapore')
        if 'malaysia' in scen:
            plt.plot(E_country[scen], c = 'orange', linestyle = '--')
            ax.text(400, E_country[scen].min() + .02, 'Malaysia')
        if 'indonesia' in scen:
            plt.plot(E_country[scen], c = 'blue', linestyle = '--')
            ax.text(400, E_country[scen].min() + .02, 'Indonesia')
        if 'cambodia' in scen:
            plt.plot(E_country[scen], c = 'green', linestyle = '--')
            ax.text(380, E_country[scen].min() + .02, 'Cambodia')
plt.axhline(y=E_country[scen].max(), xmin = 0, xmax = .72, color='k', linestyle='-')
ax.text(150, E_country[scen].max() - .08, 'Total')
plt.xlim(0,500)
plt.ylabel('Emissions (g/day)', fontsize = 16)
plt.xlabel('Day', fontsize = 16)


# In[17]:


from matplotlib.pyplot import cm
color = iter(cm.Blues(np.linspace(0.2, 1, 31)))

fig, ax = plt.subplots(figsize = [8,5])
for scen in E_country.keys():
    
    if 'indonesia' in scen:
        c = next(color)
        plt.plot(E_country[scen].cumsum()/1e4, color = c)
plt.axhline(y=E_country['indonesia_0yrs'].sum()/1e4, xmin = 0, xmax = 1, color='k', linestyle=':')
ax.text(150, E_country['indonesia_0yrs'].sum()/1e4+.03, 'Min Cumulative')

plt.axhline(y=E_country['indonesia_29yrs'].sum()/1e4, xmin = 0, xmax = 1, color='k', linestyle=':')
ax.text(150, E_country['indonesia_29yrs'].sum()/1e4 +.03, 'Max cumulative')

plt.xlabel('Years',fontsize = 16)
plt.ylabel('Emissions (kg)',fontsize = 16)
ax.set_xticks(np.linspace(0, 10950, 31))
# Set the tick labels
ax.set_xticklabels(np.arange(0,31));
#plt.xlim(0,500)


# # Convolve with G

# In[18]:


#import the green's function and set our time step
G = xr.open_dataarray('Outputs/G_SEA_BC_total.nc4')
dt = 1 #day
print('G data loaded')

G_lev0 = G.isel(lev = 0, tp = 0).compute()


#### single lev convolution ####
def convolve_single_lev(C, G, E, E_len, G_len):
    '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
    for i, tp in enumerate(np.arange(0,E_len)):
        C[i:i+G_len] = C[i:i+G_len]+ G*E[i]
        C = np.nan_to_num(C, nan=0.0)
    return C

def convolve_applied_single_lev(G, E):
    '''applies the single lev convolution'''
    E_len = len(E)
    G_len = len(G.s)
    C = np.ndarray(((E_len+G_len), len(G.lat), len(G.lon)))
    test = xr.apply_ufunc(
        convolve_single_lev,
        C,
        G,
        input_core_dims = [[],['s','lat','lon']],
        exclude_dims = set('s'),
        output_core_dims = [['s','lat','lon']],
        #vectorize=True,
        kwargs={"E": E, "E_len":E_len, "G_len":G_len},
        )
    test = test.assign_coords(s = np.arange(0,E_len + G_len))
    return(test)

C_country = {}
for nm in ['indonesia_10yrs', 'indonesia_15yrs']:
    C_country[nm] = utils.convolve_applied_single_lev(G.isel(lev = 0, tp = 0), E_country[nm])
    C_country[nm].to_netcdf(f'./Outputs/{nm}.nc4')
    print(f'convolved and saved {nm}')




