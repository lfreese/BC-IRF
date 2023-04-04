#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=MaxMemPerNode

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



############ SET LENGTH OF SIMULATION AND RANGE OF COAL LIFETIMES ########### 
years = 50

coal_year_range = np.arange( args.start_year, args.end_year)

length_simulation = years*365

time_array = np.arange(0, length_simulation)

############# IMPORT EMISSIONS DATA ###########

CGP_df = pd.read_csv('mod_coal_inputs/BC_limited_country_SEA_GAINS_Springer.csv')

CGP_df.loc[:,'BC (g/day)'] 

CGP_df.columns = CGP_df.columns.str.replace(' ', '_')

CGP_df = CGP_df.rename(columns = {'YEAR':'Year_of_Commission', 'EMISFACTOR.PLATTS':'CO2_weighted_capacity_1000tonsperMW'})

print('Emis data prepped and loaded')



#######in progress retiring by year after co2#########

def early_retirement_by_CO2_year(year_early, df, CO2_val, time_array, shutdown_years):
    ''' df must have a variable 'Year_of_Commission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day'''
    min_comission_yr = df['Year_of_Commission'].min()
    shutdown_days = shutdown_years*365
    E = np.zeros(len(time_array))
        #print(year_comis)
    
    test_array = np.where(time_array <= year_early*365, True, False)
    #plt.plot(test_array)
    E += test_array* df.loc[df.CO2_weighted_capacity_1000tonsperMW >= CO2_val]['BC_(g/day)'].sum()
        #fig, ax = plt.subplots()
        #plt.plot(E[year])
    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):
        #print(year_comis)
        #print(df.loc[df.Year_of_Commission == year_comis]['BC_(g/day)'].sum())
        #print(np.nanpercentile(CGP_df['CO2_weighted_capacity_1000tonsperMW'],r))
        test_array = np.where((time_array <= (year_comis-min_comission_yr)*365 + shutdown_days), True, False)
        #fig, ax = plt.subplots()
        #plt.plot(test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum())
        E += test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum()
        #E[year] += (time_array>=0) * df.loc[df.CO2_weighted_capacity_1000tonsperMW < CO2_val]['BC_(g/day)'].sum()
        #plt.plot(E)

    
    return(E)


########### CREATE EMIS PROFILE #########

percent = np.arange(1,101)


E_CO2_all_opts = {}
for year in coal_year_range:
    E_CO2_all_opts[year] = {}
    for r in percent:
        E_CO2_all_opts[year][r] = early_retirement_by_CO2_year(year, CGP_df, np.nanpercentile(CGP_df['CO2_weighted_capacity_1000tonsperMW'],r), time_array, 40).astype('float32')
print('Emis profiiles created')


############# AREA FOR WEIGHTED MEAN #############

ds_area = xr.open_dataset('/net/fs11/d0/emfreese/GCrundirs/IRF_runs/RRTMG_pulse/SEA/Jan/mod_output/GEOSChem.SpeciesConc.20160101_0000z.nc4', engine = 'netcdf4')



############# IMPORT GREEN'S FUNCTION ###########
G = xr.open_dataarray('Outputs/G_SEA_all_times_BC_total.nc4')
dt = 1 #day


############ SELECT MEAN AND LEVEL, LIMIT THE SPATIAL EXTENT OF LAT AND LON ###########
G_lev0 = G.isel(lev = 0).mean(dim = 'run').compute()
G_lev0 = G_lev0.rename({'time':'s'})
print('G prepped')
G_lev0 = G_lev0.astype('float32')
G_lev0 = G_lev0.sel(lat = slice(-45,50)).sel(lon= slice(45,150))


########### DEFINE SINGLE LEVEL CONVOLUTION #########

def convolve_single_lev(G, E, dt):
    '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
    E_len = len(E)
    G_len = len(G.s)
    C = dask.array.empty(((E_len+G_len), len(G.lat), len(G.lon))) ########### Switch this to numba or fortran to compile it bc then it is two vectors, numba flag to do compilation in vector form, do this; or numba and apply together ###########
    for i, tp in enumerate(np.arange(0,E_len)): ####THIS WILL BE REWRITTEN IN NUMBA (129-131)#######
        C[i:i+G_len] = C[i:i+G_len]+ G*E[i]*dt #C.loc slice or where
        #print((G*E[i]*dt).values)
        #print(C.compute())
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return C

########### CONVOLUTION #########
C_out = {}
C_sum = {}
for yr in coal_year_range:
    C_out[yr] = {}
    C_sum[yr] = {}
    for pc in percent:    
        C_out[yr][pc] = convolve_single_lev(G_lev0 , E_CO2_all_opts[yr][pc], dt = 1)
        C_sum[yr][pc] = C_out[yr][pc].sum(dim = 's')


### mask for each country 

country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
countries = ['China', 'India','Indonesia','Malaysia','Vietnam','Australia', 'Cambodia','Myanmar', 'Laos','Philippines','Nepal','Bangladesh','Thailand','Bhutan']
print('countries uploaded')


### take mean of each country and simulation 

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


