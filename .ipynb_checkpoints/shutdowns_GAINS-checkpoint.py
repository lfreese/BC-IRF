#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=48:00:00

#SBATCH --cpus-per-task=1
#SBATCH --partition=edr


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import pandas as pd
from geopy.geocoders import Nominatim
from matplotlib.colors import SymLogNorm
from matplotlib.pyplot import cm
import psutil 

import argparse

import xesmf as xe
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import dask
import utils
import gc
import geopandas
import dask.array as da

from numba import jit
import numpy as np
from numba import guvectorize, float64, int64, void

import scipy.signal as signal

####### There are three options for the type of run: weighted_co2, annual_co2, and age_retire.
####### weighted_co2 = shutdowns occur based on the percentile of capacity weighted co2 emissions (dirtier plants = bigger emissions)
####### annual_co2 = shutdowns occur based on the percentile of annual co2 emissions (so bigger plants likely = bigger co2 emissions)
####### age_retire = shutdowns occure based on the age of the plant

###### Must choose one of these four countries for emissions: 'CAMBODIA', 'INDONESIA', 'MALAYSIA', 'VIETNAM'

################## Parse arguments and set constants ##############

parser = argparse.ArgumentParser()
parser.add_argument('--start_year', type=int, required=True)
parser.add_argument('--end_year', type=int, required=True)
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--country_emit', type=str, required=True)
args = parser.parse_args()
print('Start year', args.start_year, 'End year', args.end_year, 'Run type', args.type, 'Country of emissions', args.country_emit)


years = 50
coal_year_range = np.arange(args.start_year, args.end_year)[::5]
percent = np.arange(0,101)[::5]

weighted_co2 = False
age_retire = False
annual_co2 = False
if args.type == 'weighted_co2':
    weighted_co2 = True
elif args.type == 'annual_co2':
    annual_co2 = True
elif args.type == 'age_retire':
    age_retire = True
country_emit = args.country_emit

## Add time dimension
length_simulation = years*365
time_array = np.arange(0, length_simulation)

## import the china global powerplant database
### Gallagher, Kevin P. (2021), “China’s Global Energy Finance,” Global Development Policy Center, Boston University.
### Gallagher, Kevin P., Li, Zhongshu, Chen, Xu, Ma, Xinyue (2019), “China’s Global Power Database,” Global Development Policy Center, Boston University.

CGP_df = pd.read_csv('mod_coal_inputs/BC_limited_country_SEA_GAINS_Springer.csv')

CGP_df.columns = CGP_df.columns.str.replace(' ', '_')

CGP_df = CGP_df.rename(columns = {'YEAR':'Year_of_Commission', 'EMISFACTOR.PLATTS':'CO2_weighted_capacity_1000tonsperMW'})

## reduce to one country for emissions

CGP_df = CGP_df.loc[CGP_df['COUNTRY'] == country_emit]

print('Emis data prepped and loaded')


######## Country mask and dataframe ######

country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
country_df = geopandas.read_file('ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
countries = ['China']#,'Indonesia','Malaysia','Vietnam','Cambodia']#'Australia', 'Myanmar', 'Laos','Philippines','Nepal','Bangladesh','Thailand','Bhutan']
country_df = country_df.rename(columns = {'SOVEREIGNT':'country'})

ds_area = xr.open_dataset('/net/fs11/d0/emfreese/GCrundirs/IRF_runs/stretch_2x_pulse/SEA/Jan/mod_output/GEOSChem.SpeciesConc.20160101_0000z.nc4', engine = 'netcdf4')
utils.fix_area_ij_latlon(ds_area);


######## Set health data #########

RR = 1.02 #global mean
del_x = 10 #ug/m3
beta = np.log(RR)/del_x


#2019 mortalities to match 2019 population data from the GBD 
I_val = {}
I_val['China'] = 10462043.68
I_val['Indonesia'] = 35874.09
I_val['Malaysia'] = 169483.46
I_val['Vietnam'] = 606145.89
I_val['Australia'] = 169053.20
I_val['Cambodia'] = 96284.85
I_val['Myanmar'] = 368031.84
I_val['Laos'] = 35874.09
I_val['Philippines'] = 557809.29
I_val['Nepal'] = 170032.44
I_val['Bangladesh'] = 740684.73
I_val['Thailand'] = 486556.52
I_val['Bhutan'] = 3713.40

I_val_df = pd.DataFrame(I_val.values(), index = I_val.keys()).rename(columns = {0:'Ival'}) 
I_val_df.index.rename('country', inplace = True)
pop_df = country_df.loc[country_df['country'].isin(countries)].loc[country_df['POP_YEAR'] == 2019][['country','POP_EST']].set_index('country').groupby('country').max() #select 2019 population data
I0_pop_df = pd.merge(pop_df, I_val_df, left_index=True, right_index=True) #combine Initial mortality and total population by country
I0_pop_df['I_obs'] = I0_pop_df['Ival']/I0_pop_df['POP_EST'] #calculate initial mortality rate, I_obs

regrid_area_ds = xr.open_dataset('Outputs/regridded_population_data.nc')

####### Functions #########

def early_retirement_by_CO2_weighted(year_early, df, CO2_val, time_array, shutdown_years):
    ''' Shutdown a plant early based on its capacity weighted CO2 emissions. Shutdowns occur by the percentile of the plant (eg: top 10%, top 90%). The df must have a variable 'Year_of_Commission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day
        year_early is the number of years the plant runs
        min_comission_yr is the earliest year a plant was built and is where our timeline starts
        time_array is the length of time for our simulation
        shutdown_years is the typical lifetime of a coal plant'''
    min_comission_yr = df['Year_of_Commission'].min()
    shutdown_days = shutdown_years*365
    E = np.zeros(len(time_array))
    #print(min_comission_yr)
    #print(shutdown_days)
    test_array = np.where(time_array <= year_early*365, True, False)
    #print('test array len', len(test_array))
    #plt.plot(test_array)
    E += test_array* df.loc[df.CO2_weighted_capacity_1000tonsperMW >= CO2_val]['BC_(g/day)'].sum()
    #fig, ax = plt.subplots()
    #plt.plot(E)
    #plt.title('E')
    #print(E)
    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):
        #print(year_comis)
        #print(df.loc[df.Year_of_Commission == year_comis]['BC_(g/day)'].sum())
        #print(np.nanpercentile(CGP_df['CO2_weighted_capacity_1000tonsperMW'],r))
        test_array = np.where((time_array <= (year_comis-min_comission_yr)*365 + shutdown_days), True, False)
        #plt.plot(test_array)
        #fig, ax = plt.subplots()
        #plt.plot(test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum())
        E += test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum()
        #E[year] += (time_array>=0) * df.loc[df.CO2_weighted_capacity_1000tonsperMW < CO2_val]['BC_(g/day)'].sum()
        #plt.plot(E)

    
    return(E)


def early_retirement_by_CO2_annual(year_early, df, CO2_val, time_array, shutdown_years):
    ''' Shutdown a plant early based on its annual CO2 emissions. Shutdowns occur by the percentile of the plant (eg: tope 10%, top 90%). The df must have a variable 'Year_of_Commission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day
        year_early is the number of years the plant runs
        min_comission_yr is the earliest year a plant was built and is where our timeline starts
        time_array is the length of time for our simulation
        shutdown_years is the typical lifetime of a coal plant'''
    min_comission_yr = df['Year_of_Commission'].min()
    shutdown_days = shutdown_years*365
    E = np.zeros(len(time_array))
    #print(min_comission_yr)
    #print(shutdown_days)
    test_array = np.where(time_array <= year_early*365, True, False)
    #print('test array len', len(test_array))
    #plt.plot(test_array)
    E += test_array* df.loc[df.ANNUALCO2 >= CO2_val]['BC_(g/day)'].sum()
    #fig, ax = plt.subplots()
    #plt.plot(E)
    #plt.title('E')
    #print(E)
    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):
        #print(year_comis)
        #print(df.loc[df.Year_of_Commission == year_comis]['BC_(g/day)'].sum())
        #print(np.nanpercentile(CGP_df['CO2_weighted_capacity_1000tonsperMW'],r))
        test_array = np.where((time_array <= (year_comis-min_comission_yr)*365 + shutdown_days), True, False)
        #plt.plot(test_array)
        #fig, ax = plt.subplots()
        #plt.plot(test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum())
        E += test_array* df.loc[(df.ANNUALCO2 < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum()
        #E[year] += (time_array>=0) * df.loc[df.CO2_weighted_capacity_1000tonsperMW < CO2_val]['BC_(g/day)'].sum()
        #plt.plot(E)
    return(E)


def early_retirement_by_year(df, time_array, shutdown_years):
    ''' Shutdown a plant early if comissioned before a certain year, all other plants stay on until they reach 40 year time limit. The df must have a variable 'Year_of_Comission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day'''
    #shutdown_years = 10
    min_comission_yr = df['Year_of_Commission'].min()
    shutdown_days = shutdown_years*365

    E = np.zeros(len(time_array))
    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):
        #print(year_comis)
        #print(CGP_op.loc[CGP_op.Year_of_Commission == year_comis]['BC_(g/day)'].sum())
        test_array = np.where((time_array < (year_comis-min_comission_yr)*365 + shutdown_days) & (time_array >= (year_comis-min_comission_yr)*365), True, False)
        #plt.plot(test_array)
        E += test_array* df.loc[df.Year_of_Commission == year_comis]['BC_(g/day)'].sum()
        #plt.plot(E)
    return(E)


########## Create emissions profile ##########


if weighted_co2 == True:

    E_CO2_all_opts = {}
    for year in coal_year_range:
        E_CO2_all_opts[year] = {}
        for r in percent:
            E_CO2_all_opts[year][r] = early_retirement_by_CO2_weighted(year, CGP_df, np.nanpercentile(CGP_df['CO2_weighted_capacity_1000tonsperMW'].dropna(),r), time_array, 40)
    print('Emissions profiles based on weighted capacity of CO2 emissions percentiles createdd')


elif annual_co2 == True:

    E_CO2_all_opts = {}
    for year in coal_year_range:
        E_CO2_all_opts[year] = {}
        for r in percent:
            E_CO2_all_opts[year][r] = early_retirement_by_CO2_annual(year, CGP_df, np.nanpercentile(CGP_df['ANNUALCO2'].dropna(),r), time_array, 40)
    print('Emissions profiles based on Annual CO2 emissions percentiles created')

    

elif age_retire == True:

    E_CO2_all_opts = {}
    for year in coal_year_range:
        E_CO2_all_opts[year] = early_retirement_by_year(CGP_df, time_array, year)
    print('Emissions profiles based on age')

    
    
############### Convolution, selection of location, health impact assessment ##########
    
## Convolve with G
country_emit_dict = {'INDONESIA':['Indo_Jan', 'Indo_Apr', 'Indo_July','Indo_Oct'], 'CAMBODIA':['Cambod_Jan', 'Cambod_Apr', 'Cambod_July','Cambod_Oct'] , 
               'MALAYSIA': ['Malay_Jan','Malay_Apr','Malay_July','Malay_Oct'], 'VIETNAM': ['Viet_Jan','Viet_Apr','Viet_July','Viet_Oct']}

#import the green's function and set our time step
G = xr.open_dataarray('Outputs/G_all_loc_all_times_BC_total.nc4', chunks = 'auto')
dt = 1 #day
G_lev0 = G.where(G.run.isin(country_emit_dict[country_emit]), drop = True).mean(dim = 'run').isel(lev = 0).compute()
G_lev0 = G_lev0.rename({'time':'s'})
print('G prepped')



def np_to_xr(C, G, E):
    E_len = len(E)
    G_len = len(G.s)
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0, C.shape[0])), #np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return(C)

## define our base pollution level
E_base = early_retirement_by_year(CGP_df, time_array, 40)
ds_base = signal.convolve(G_lev0.to_numpy(), E_base[..., None, None], mode = 'full')
ds_base = np_to_xr(ds_base, G_lev0, E_base)


# def multiprocess_func(G_lev0, E_CO2_all_opts, country_mask, countries, yr, pc):
#     data = pd.DataFrame(columns = ['Mortalities','BC_Conc'], index = countries)
#      #concentration
#     C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][pc][..., None, None], mode = 'full')
#     C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
#     C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr][pc])

#     #heath impacts
#     mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
#     for country_impacted in countries:
#         contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(country_impacted))
#         country_impacted_ds = C_out.where(contiguous_mask)
#         country_impacted_ds = country_impacted_ds.to_dataset(name = 'BC_conc')
#         country_area = ds_area['area'].where(contiguous_mask)
#              #   print(country_impacted_ds)
#         country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
#         country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
#          #       print(country_impacted_ds['AF'].max().values, country_impacted_ds['AF'].mean().values, )
#         mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
#         conc_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test

#     data.loc[country_impacted] = [mort_out, conc_out]
            
#     data.to_csv(f'Outputs/weighted_co2/C_out_{country_emit}_{runtype}_{pc}pct_{yr}yr.nc')

# import multiprocessing 

# import itertools

# paramlist = list(itertools.product(coal_year_range, percent))
    
# if __name__ == '__main__':
#     if weighted_co2 == True:
#         def multiprocess_func(params):
#             yr = params[0]
#             pc = params[1]

#             data = pd.DataFrame(columns = ['Mortalities','BC_Conc'], index = countries)
#              #concentration
#             C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][pc][..., None, None], mode = 'full')
#             C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
#             C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr][pc])

#             #heath impacts
#             mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
#             for country_impacted in countries:
#                 contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(country_impacted))
#                 country_impacted_ds = C_out.where(contiguous_mask)
#                 country_impacted_ds = country_impacted_ds.to_dataset(name = 'BC_conc')
#                 country_area = ds_area['area'].where(contiguous_mask)
#                      #   print(country_impacted_ds)
#                 country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
#                 country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
#                  #       print(country_impacted_ds['AF'].max().values, country_impacted_ds['AF'].mean().values, )
#                 mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
#                 conc_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test

#             data.loc[country_impacted] = [mort_out, conc_out]

#             data.to_csv(f'Outputs/weighted_co2/C_out_{country_emit}_{runtype}_{pc}pct_{yr}yr.nc')



#         pool = multiprocessing.Pool()

#         res = pool.map(multiprocess_func, paramlist)
#         pool.close()
#         pool.join()
#         results_df = pd.concat(res)
#         print(results_df)


if weighted_co2 == True:
    runtype = 'weighted'
    for yr in coal_year_range:
        processes = []
        for pc in percent:   #active
#             p = multiprocessing.Process(target = multiprocess_func, args = (G_lev0, E_CO2_all_opts, country_mask, countries, yr, pc))
#             processes.append(p)
#             p.start()
#             for process in processes:
#                 process.join()
            data = pd.DataFrame(columns = ['Mortalities','BC_Conc'], index = countries)
            #concentration
            C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][pc][..., None, None], mode = 'full')
            C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
            C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr][pc])

            #heath impacts
            mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
            for country_impacted in countries:
                contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(country_impacted))
                country_impacted_ds = C_out.where(contiguous_mask)
                country_impacted_ds = country_impacted_ds.to_dataset(name = 'BC_conc')
                country_area = ds_area['area'].where(contiguous_mask)
                print(country_impacted_ds)
                country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
                country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
                #print(country_impacted_ds['AF'].max().values, country_impacted_ds['AF'].mean().values, )
                mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
                conc_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test

                data.loc[country_impacted] = [mort_out, conc_out]
                print(data)
            data.to_csv(f'Outputs/weighted_co2/C_out_{country_emit}_{runtype}_{pc}pct_{yr}yr.nc')
            print(f'saved out {country_emit}, {runtype}, {pc} percent, {yr} year')
            print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
            lst = [data]
            del data
            del lst
            gc.collect()
            print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)


elif annual_co2 == True:
    runtype = 'annual'
    
    for yr in coal_year_range:
        for pc in percent:  
            data = pd.DataFrame(columns = ['Mortalities','BC_Conc'], index = countries)
            #concentration
            C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][pc][..., None, None], mode = 'full')
            C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
            C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr][pc])
            #heath impacts
            mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
            for country_impacted in countries:
                contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(country_impacted))
                country_impacted_ds = C_out.where(contiguous_mask)
                country_impacted_ds = country_impacted_ds.to_dataset(name = 'BC_conc')
                country_area = ds_area['area'].where(contiguous_mask)
                
                country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
                country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
                
                mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
                conc_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
                
                
                data.loc[country_impacted] = [mort_out, conc_out]
            
            data.to_csv(f'Outputs/annual_co2/C_out_{country_emit}_{runtype}_{pc}pct_{yr}yr.nc')
            print(f'saved out {country_emit}, {runtype}, {pc} percent, {yr} year')
            lst = [data]
            del data
            del lst


elif age_retire == True:
    runtype = 'age'
    for yr in coal_year_range:
        data = pd.DataFrame(columns = ['Mortalities','BC_Conc'], index = countries)
        #concentration
        C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][..., None, None], mode = 'full')
        C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
        C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr])
        #heath impacts
        mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
        for country_impacted in countries:
            contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(country_impacted))
            country_impacted_ds = C_out.where(contiguous_mask)
            country_impacted_ds = country_impacted_ds.to_dataset(name ='BC_conc')
            country_area = ds_area['area'].where(contiguous_mask)

            country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
            country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
            print(country_impacted_ds['delta_I'].weighted(ds_area['area']).mean(dim = ['lat','lon']))
            print(country_impacted_ds['BC_conc'].weighted(ds_area['area']).mean(dim = ['lat','lon']))
            mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
            conc_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test


            data.loc[country_impacted] = [mort_out, conc_out]
            print(data)
        data.to_csv(f'Outputs/retire_age/C_out_{country_emit}_{runtype}_{yr}yr.nc')
        print(f'saved out {country_emit}, {runtype}, {yr} year')
        lst = [data]
        del data
        del lst
