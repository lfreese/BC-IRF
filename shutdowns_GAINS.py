#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=4-00:00:00

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
import sparse

####### There are three options for the type of run: weighted_co2, annual_co2, and age_retire.
####### weighted_co2 = shutdowns occur based on the percentile of capacity weighted co2 emissions (dirtier plants = bigger emissions)
####### annual_co2 = shutdowns occur based on the percentile of annual co2 emissions (so bigger plants likely = bigger co2 emissions)
####### age_retire = shutdowns occure based on the age of the plant

###### Must choose one of these four countries for emissions:  'INDONESIA', 'MALAYSIA', 'VIETNAM'

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

## Days per season
season_days = {'DJF': 90, 'MAM':92, 'JJA':92, 'SON':91}


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

country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_50
country_df = geopandas.read_file('ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
countries = ['China','Australia', 'India','Myanmar', 'Cambodia', 'Laos','Philippines','Nepal','Bangladesh','Thailand','Bhutan','Brunei','Singapore', 'Papua New Guinea', 'Solomon Islands', 'East Timor', 'Taiwan']
country_df = country_df.rename(columns = {'SOVEREIGNT':'country'})

ds_area = xr.open_dataset('/net/fs11/d0/emfreese/GCrundirs/IRF_runs/stretch_2x_pulse/SEA/Jan/mod_output/GEOSChem.SpeciesConc.20160101_0000z.nc4', engine = 'netcdf4')
utils.fix_area_ij_latlon(ds_area);


######## Set health data #########

RR = 1.02 #global mean
del_x = 10 #ug/m3
beta = np.log(RR)/del_x


#2019 mortalities to match 2019 population data from the GBD #20+, all gender mortalities https://vizhub.healthdata.org/gbd-results/
I_val = {}
I_val['China'] = 10462043.68
#I_val['Indonesia'] = 35874.09
#I_val['Malaysia'] = 169483.46
#I_val['Vietnam'] = 606145.89
I_val['Australia'] = 169053.20
I_val['Cambodia'] = 96284.85
I_val['Myanmar'] = 368031.84
I_val['Laos'] = 35874.09
I_val['Philippines'] = 557809.29
I_val['Nepal'] = 170032.44
I_val['Bangladesh'] = 740684.73
I_val['Thailand'] = 486556.52
I_val['Bhutan'] = 3713.40
I_val['East Timor'] = 6240.29
I_val['Solomon Islands'] = 5484.73
I_val['Brunei'] = 1790.80
I_val['Papua New Guinea'] = 49174.72
I_val['India'] = 8288847.22
I_val['Singapore'] = 23028.19
I_val['Taiwan'] = 183845.08


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

    test_array = np.where(time_array <= year_early*365, True, False)

    E += test_array* df.loc[df.CO2_weighted_capacity_1000tonsperMW >= CO2_val]['BC_(g/day)'].sum()

    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):
        test_array = np.where((time_array <= (year_comis-min_comission_yr)*365 + shutdown_days), True, False)

        E += test_array* df.loc[(df.CO2_weighted_capacity_1000tonsperMW < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum()


    
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

    test_array = np.where(time_array <= year_early*365, True, False)

    E += test_array* df.loc[df.ANNUALCO2 >= CO2_val]['BC_(g/day)'].sum()

    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):

        test_array = np.where((time_array <= (year_comis-min_comission_yr)*365 + shutdown_days), True, False)

        E += test_array* df.loc[(df.ANNUALCO2 < CO2_val) & (df.Year_of_Commission == year_comis)]['BC_(g/day)'].sum()

    return(E)


def early_retirement_by_year(df, time_array, shutdown_years):
    ''' Shutdown a plant early if comissioned before a certain year, all other plants stay on until they reach 40 year time limit. The df must have a variable 'Year_of_Comission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day'''
    min_comission_yr = df['Year_of_Commission'].min()
    shutdown_days = shutdown_years*365

    E = np.zeros(len(time_array))
    for year_comis in np.arange(min_comission_yr, df['Year_of_Commission'].max()):

        test_array = np.where((time_array < (year_comis-min_comission_yr)*365 + shutdown_days) & (time_array >= (year_comis-min_comission_yr)*365), True, False)
        E += test_array* df.loc[df.Year_of_Commission == year_comis]['BC_(g/day)'].sum()
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
G = xr.open_dataarray('Outputs/G_combined_new.nc', chunks = 'auto')
dt = 1 #day

G_lev0 = G.where(G.run.isin(country_emit_dict[country_emit]), drop = True).isel(lev = 0).compute()
G_lev0 = G_lev0.rename({'time':'s'})

###########CHECK THIS CHANGE ###########
#G_lev0 = {}
# for season in season_days:
#     G_lev0[season] = G.where(G.run.isin(country_emit_dict[country_emit]), drop = True).mean(dim = 'run').isel(lev = 0).compute()
#     G_lev0[season] = G_lev0[season].rename({'time':'s'})

print('G prepped')
############################################

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

E_base = early_retirement_by_year(CGP_df, time_array, 40)
C_conv = {}
C_init = {}
yr_range = coal_year_range
n = 0
for yr_num in yr_range:
    n_init = n

    for idx, season in enumerate(season_days.keys()):
        C_conv[season] = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx]).dropna(dim = 's'), 
                     E_base[n:n+season_days[season]+1][..., None, None], mode = 'full')
        #switch the tail (that goes into the following months) to follow the GF of the next month 
        if idx == 0 or idx == 1 or idx == 2:
            idx_next = idx + 1
        elif idx == 3:
            idx_next = 0

        tail_switch = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx_next]).dropna(dim = 's'), 
                             E_base[n:n+season_days[season]][..., None, None], mode = 'full')

        C_conv[season][season_days[season]:tail_switch[season_days[season]:C_conv[season].shape[0]].shape[0]+season_days[season]] = tail_switch[season_days[season]:C_conv[season].shape[0]]

        n = n + season_days[season]

    C = {}

    C['DJF'] = sparse.COO.from_numpy((np.pad(C_conv['DJF'],((((n_init),
                                       (len(E_base) - len(C_conv['DJF']) - (n_init)))),
                                     (0,0),(0,0)))))
    C['MAM'] = sparse.COO.from_numpy((np.pad(C_conv['MAM'],((((season_days['DJF'] + (n_init)),
                                       (len(E_base) - season_days['DJF'] - len(C_conv['MAM']) - (n_init)))),
                                     (0,0),(0,0)))))
    C['JJA'] = sparse.COO.from_numpy((np.pad(C_conv['JJA'],(((season_days['DJF'] + season_days['MAM'] + (n_init),
                                       (len(E_base) - season_days['DJF'] - season_days['MAM'] - len(C_conv['JJA']) - (n_init)))),
                                     (0,0),(0,0)))))
    C['SON'] = sparse.COO.from_numpy((np.pad(C_conv['SON'], (((season_days['DJF'] + season_days['MAM'] + season_days['JJA'] + (n_init),
                                        (len(E_base) - season_days['DJF'] - season_days['MAM'] - season_days['JJA'] - len(C_conv['SON']) - (n_init)))),
                                      (0,0),(0,0)))))
    C_init[yr_num] = C['DJF']+C['MAM']+C['JJA']+C['SON']

C_sum = sum(C_init[yr_num] for yr_num in yr_range)
C_dense = sparse.COO.todense(C_sum)
ds_base = np_to_xr(C_dense, G_lev0, E_base)

################old code ######################
## define our base pollution level
# E_base = early_retirement_by_year(CGP_df, time_array, 40)
# ds_base = signal.convolve(G_lev0.to_numpy(), E_base[..., None, None], mode = 'full')
# ds_base = np_to_xr(ds_base, G_lev0, E_base)


if weighted_co2 == True:
    runtype = 'weighted'
    for yr in coal_year_range:
        processes = []
        for pc in percent:   #active
            data = pd.DataFrame(columns = ['Mortalities','BC_mean_Conc', 'BC_sum_conc','BC_pop_weight_mean_conc'], index = countries)
            #concentration
            #####start new#############
            C_conv = {}
            C_init = {}
            yr_range = coal_year_range
            n = 0
            for yr_num in yr_range:
                n_init = n
                
                for idx, season in enumerate(season_days.keys()):
                    C_conv[season] = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx]).dropna(dim = 's'), 
                                 E_CO2_all_opts[yr][pc][n:n+season_days[season]][..., None, None], mode = 'full')
                    #switch the tail (that goes into the following months) to follow the GF of the next month 
                    if idx == 0 or idx == 1 or idx == 2:
                        idx_next = idx + 1
                    elif idx == 3:
                        idx_next = 0

                    tail_switch = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx_next]).dropna(dim = 's'), 
                                         E_CO2_all_opts[yr][pc][n:n+season_days[season]][..., None, None], mode = 'full')

                    C_conv[season][season_days[season]:tail_switch[season_days[season]:C_conv[season].shape[0]].shape[0]+season_days[season]] = tail_switch[season_days[season]:C_conv[season].shape[0]]

                    n = n + season_days[season]
                
                C = {}
                
                C['DJF'] = sparse.COO.from_numpy((np.pad(C_conv['DJF'],((((n_init),
                                                   (len(E_CO2_all_opts[yr][pc]) - len(C_conv['DJF']) - (n_init)))),
                                                 (0,0),(0,0)))))
                C['MAM'] = sparse.COO.from_numpy((np.pad(C_conv['MAM'],((((season_days['DJF'] + (n_init)),
                                                   (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - len(C_conv['MAM']) - (n_init)))),
                                                 (0,0),(0,0)))))
                C['JJA'] = sparse.COO.from_numpy((np.pad(C_conv['JJA'],(((season_days['DJF'] + season_days['MAM'] + (n_init),
                                                   (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - season_days['MAM'] - len(C_conv['JJA']) - (n_init)))),
                                                 (0,0),(0,0)))))
                C['SON'] = sparse.COO.from_numpy((np.pad(C_conv['SON'], (((season_days['DJF'] + season_days['MAM'] + season_days['JJA'] + (n_init),
                                                    (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - season_days['MAM'] - season_days['JJA'] - len(C_conv['SON']) - (n_init)))),
                                                  (0,0),(0,0)))))
                C_init[yr_num] = C['DJF']+C['MAM']+C['JJA']+C['SON']

            C_sum = sum(C_init[yr_num] for yr_num in yr_range)
            C_dense = sparse.COO.todense(C_sum)
            C_out = np_to_xr(C_dense, G_lev0, E_CO2_all_opts[yr][pc])

            #####end new###############
            ######old###########
#             C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][pc][..., None, None], mode = 'full')
#             C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
#             C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr][pc])
            ######old###########
            #heath impacts
            mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
            for country_impacted in countries:
                if country_impacted == 'East Timor':
                    c_imp = 'Timor-Leste'
                elif country_impacted == 'Solomon Islands':
                    c_imp = 'Solomon Is.'
                else:
                    c_imp = country_impacted
                contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(c_imp))
                country_impacted_ds = C_out.where(contiguous_mask)
                country_impacted_ds = country_impacted_ds.to_dataset(name = 'BC_conc')
                country_area = ds_area['area'].where(contiguous_mask)
                print(country_impacted_ds)
                country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
                country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
                #print(country_impacted_ds['AF'].max().values, country_impacted_ds['AF'].mean().values, )
                mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
                conc_mean_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
                conc_sum_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values #take a mean to test
                
            
                pop_weight_conc = utils.grouped_weighted_avg(country_impacted_ds['BC_conc'], regrid_area_ds['regrid_pop_count']).values
                #pop_weight_mean_conc_out = ((pop_weight_conc*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
                #pop_weight_sum_conc_out = ((pop_weight_conc*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values #take a mean to test

                data.loc[country_impacted] = [mort_out, conc_mean_out, conc_sum_out, pop_weight_conc]
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
            data = pd.DataFrame(columns = ['Mortalities','BC_mean_Conc', 'BC_sum_conc','BC_pop_weight_mean_conc'], index = countries)
            #concentration
            #####start new#############
            C_conv = {}
            C_init = {}
            yr_range = coal_year_range
            n = 0
            for yr_num in yr_range:
                n_init = n
                print(n)
                for idx, season in enumerate(season_days.keys()):
                    C_conv[season] = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx]).dropna(dim = 's'), 
                                 E_CO2_all_opts[yr][pc][n:n+season_days[season]][..., None, None], mode = 'full')
                    #switch the tail (that goes into the following months) to follow the GF of the next month 
                    if idx == 0 or idx == 1 or idx == 2:
                        idx_next = idx + 1
                    elif idx == 3:
                        idx_next = 0

                    tail_switch = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx_next]).dropna(dim = 's'), 
                                         E_CO2_all_opts[yr][pc][n:n+season_days[season]][..., None, None], mode = 'full')

                    C_conv[season][season_days[season]:tail_switch[season_days[season]:C_conv[season].shape[0]].shape[0]+season_days[season]] = tail_switch[season_days[season]:C_conv[season].shape[0]]

                    n = n + season_days[season]
                    print(n)
                C = {}
                print(f'initial n {n_init}')
                C['DJF'] = sparse.COO.from_numpy((np.pad(C_conv['DJF'],((((n_init),
                                                   (len(E_CO2_all_opts[yr][pc]) - len(C_conv['DJF']) - (n_init)))),
                                                 (0,0),(0,0)))))
                C['MAM'] = sparse.COO.from_numpy((np.pad(C_conv['MAM'],((((season_days['DJF'] + (n_init)),
                                                   (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - len(C_conv['MAM']) - (n_init)))),
                                                 (0,0),(0,0)))))
                C['JJA'] = sparse.COO.from_numpy((np.pad(C_conv['JJA'],(((season_days['DJF'] + season_days['MAM'] + (n_init),
                                                   (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - season_days['MAM'] - len(C_conv['JJA']) - (n_init)))),
                                                 (0,0),(0,0)))))
                C['SON'] = sparse.COO.from_numpy((np.pad(C_conv['SON'], (((season_days['DJF'] + season_days['MAM'] + season_days['JJA'] + (n_init),
                                                    (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - season_days['MAM'] - season_days['JJA'] - len(C_conv['SON']) - (n_init)))),
                                                  (0,0),(0,0)))))
                C_init[yr_num] = C['DJF']+C['MAM']+C['JJA']+C['SON']

            C_sum = sum(C_init[yr_num] for yr_num in yr_range)
            C_dense = sparse.COO.todense(C_sum)
            C_out = np_to_xr(C_dense, G_lev0, E_CO2_all_opts[yr][pc])

            #####end new###############
            ##### old #########
#             #concentration
#             C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][pc][..., None, None], mode = 'full')
#             C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
#             C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr][pc])
            ##### old #########
    
            #heath impacts
            mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
            for country_impacted in countries:
                if country_impacted == 'East Timor':
                        c_imp = 'Timor-Leste'
                elif country_impacted == 'Solomon Islands':
                    c_imp = 'Solomon Is.'
                else:
                    c_imp = country_impacted
                contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(c_imp))
                country_impacted_ds = C_out.where(contiguous_mask)
                country_impacted_ds = country_impacted_ds.to_dataset(name = 'BC_conc')
                country_area = ds_area['area'].where(contiguous_mask)
                
                country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
                country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']

                mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
                conc_mean_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
                conc_sum_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values #take a mean to test
                
            
                pop_weight_conc = utils.grouped_weighted_avg(country_impacted_ds['BC_conc'], regrid_area_ds['regrid_pop_count']).values
                #pop_weight_mean_conc_out = ((pop_weight_conc*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
                #pop_weight_sum_conc_out = ((pop_weight_conc*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values #take a mean to test

                data.loc[country_impacted] = [mort_out, conc_mean_out, conc_sum_out, pop_weight_conc]
                            
            data.to_csv(f'Outputs/annual_co2/C_out_{country_emit}_{runtype}_{pc}pct_{yr}yr.nc')
            print(f'saved out {country_emit}, {runtype}, {pc} percent, {yr} year')
            lst = [data]
            del data
            del lst


elif age_retire == True:
    runtype = 'age'
    for yr in coal_year_range:
        data = pd.DataFrame(columns = ['Mortalities','BC_mean_Conc', 'BC_sum_conc','BC_pop_weight_mean_conc'], index = countries)
        #concentration
        #####start new#############
        C_conv = {}
        C_init = {}
        yr_range = np.arange(1,5)#len(E_CO2_all_opts[yr][pc]/365))
        n = 0
        for yr_num in yr_range:
            n_init = n
            print(n)
            for idx, season in enumerate(season_days.keys()):
                C_conv[season] = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx]).dropna(dim = 's'), 
                             E_CO2_all_opts[yr][pc][n:n+season_days[season]][..., None, None], mode = 'full')
                #switch the tail (that goes into the following months) to follow the GF of the next month 
                if idx == 0 or idx == 1 or idx == 2:
                    idx_next = idx + 1
                elif idx == 3:
                    idx_next = 0

                tail_switch = signal.convolve(G_lev0.sel(run = country_emit_dict[country_emit][idx_next]).dropna(dim = 's'), 
                                     E_CO2_all_opts[yr][pc][n:n+season_days[season]][..., None, None], mode = 'full')

                C_conv[season][season_days[season]:tail_switch[season_days[season]:C_conv[season].shape[0]].shape[0]+season_days[season]] = tail_switch[season_days[season]:C_conv[season].shape[0]]

                n = n + season_days[season]
                print(n)
            C = {}
            print(f'initial n {n_init}')
            C['DJF'] = sparse.COO.from_numpy((np.pad(C_conv['DJF'],((((n_init),
                                               (len(E_CO2_all_opts[yr][pc]) - len(C_conv['DJF']) - (n_init)))),
                                             (0,0),(0,0)))))
            C['MAM'] = sparse.COO.from_numpy((np.pad(C_conv['MAM'],((((season_days['DJF'] + (n_init)),
                                               (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - len(C_conv['MAM']) - (n_init)))),
                                             (0,0),(0,0)))))
            C['JJA'] = sparse.COO.from_numpy((np.pad(C_conv['JJA'],(((season_days['DJF'] + season_days['MAM'] + (n_init),
                                               (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - season_days['MAM'] - len(C_conv['JJA']) - (n_init)))),
                                             (0,0),(0,0)))))
            C['SON'] = sparse.COO.from_numpy((np.pad(C_conv['SON'], (((season_days['DJF'] + season_days['MAM'] + season_days['JJA'] + (n_init),
                                                (len(E_CO2_all_opts[yr][pc]) - season_days['DJF'] - season_days['MAM'] - season_days['JJA'] - len(C_conv['SON']) - (n_init)))),
                                              (0,0),(0,0)))))
            C_init[yr_num] = C['DJF']+C['MAM']+C['JJA']+C['SON']

            C_sum = sum(C_init[yr_num] for yr_num in yr_range)
            C_dense = sparse.COO.todense(C_sum)
            C_out = np_to_xr(C_dense, G_lev0, E_CO2_all_opts[yr][pc])

            #####end new###############
            
        
        ####old######
#         C_out =  signal.convolve(G_lev0.to_numpy(), E_CO2_all_opts[yr][..., None, None], mode = 'full')
#         C_out = da.from_array(C_out, chunks = [C_out.shape[0]//10,C_out.shape[1]//10,C_out.shape[2]//10])
#         C_out = np_to_xr(C_out, G_lev0, E_CO2_all_opts[yr])
        ####old######
    
        #heath impacts
        mask = country_mask.mask(C_out, lon_name = 'lon', lat_name = 'lat')
        for country_impacted in countries:
            if country_impacted == 'East Timor':
                c_imp = 'Timor-Leste'
            elif country_impacted == 'Solomon Islands':
                c_imp = 'Solomon Is.'
            else:
                c_imp = country_impacted
            contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(c_imp))
            country_impacted_ds = C_out.where(contiguous_mask)
            country_impacted_ds = country_impacted_ds.to_dataset(name ='BC_conc')
            country_area = ds_area['area'].where(contiguous_mask)

            country_impacted_ds['AF'] = (np.exp(beta*(country_impacted_ds['BC_conc']- ds_base)) - 1)#/np.exp(beta*(country_impacted_ds['BC_conc']- ds_base))
            country_impacted_ds['delta_I'] = country_impacted_ds['AF']*regrid_area_ds['regrid_pop_count']*I0_pop_df.loc[country_impacted]['I_obs']
            #print(country_impacted_ds['delta_I'].weighted(ds_area['area']).mean(dim = ['lat','lon']))
            #print(country_impacted_ds['BC_conc'].weighted(ds_area['area']).mean(dim = ['lat','lon']))

            mort_out = ((country_impacted_ds['delta_I']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values
            conc_mean_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
            conc_sum_out = ((country_impacted_ds['BC_conc']*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values #take a mean to test


            
            pop_weight_conc = utils.grouped_weighted_avg(country_impacted_ds['BC_conc'], regrid_area_ds['regrid_pop_count']).values
            #pop_weight_mean_conc_out = ((pop_weight_conc*country_area).sum(dim = ['lat','lon'])/country_area.sum()).mean(dim = ['s']).values #take a mean to test
            #pop_weight_sum_conc_out = ((pop_weight_conc*country_area).sum(dim = ['lat','lon'])/country_area.sum()).sum(dim = ['s']).values #take a mean to test

            data.loc[country_impacted] = [mort_out, conc_mean_out, conc_sum_out, pop_weight_conc]
            print(data)
        data.to_csv(f'Outputs/retire_age/C_out_{country_emit}_{runtype}_{yr}yr.nc')
        print(f'saved out {country_emit}, {runtype}, {yr} year')
        lst = [data]
        del data
        del lst
