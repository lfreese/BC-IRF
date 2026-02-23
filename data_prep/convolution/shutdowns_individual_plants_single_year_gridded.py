#!/home/emfreese/miniconda3/envs/bc-irf/bin/python
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=edr
import os
# Set PROJ_LIB path before importing geopandas
os.environ['PROJ_LIB'] = '/home/emfreese/miniconda3/envs/bc-irf/share/proj'

import regionmask
import pandas as pd
import psutil 
import argparse
import xesmf as xe
import gc
import geopandas
import dask.array as da
import scipy.signal as signal
import sparse
import xarray as xr
import numpy as np
import argparse
import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})
import sys
sys.path.insert(0, '/net/fs11/d0/emfreese/BC-IRF/')
import utils


################## Parse arguments and set constants ##############

parser = argparse.ArgumentParser()
parser.add_argument('--country', type=str, required=True)
args = parser.parse_args()
print('Country of emissions', args.country)

years = utils.years
country_emit = args.country

## Add time dimension in terms of days
length_simulation = years*365
time_array = np.arange(0, length_simulation)

## Days per season
season_days = {'DJF': 90, 'MAM':92, 'JJA':92, 'SON':91}

## import the china global powerplant database from Springer et al.
CGP_df = pd.read_csv(f'{utils.data_output_path}plants/BC_SE_Asia_all_financing_SEA_GAINS_Springer.csv')
CGP_df.columns = CGP_df.columns.str.replace(' ', '_')
CGP_df = CGP_df.rename(columns = {'YEAR':'Year_of_Commission', 'EMISFACTOR.PLATTS':'CO2_weighted_capacity_1000tonsperMW'})
min_year = CGP_df['Year_of_Commission'].min()

## reduce to one country for emissions
CGP_df = CGP_df.loc[CGP_df['COUNTRY'] == country_emit]

##remove any plants that have no data on commission year
CGP_df = CGP_df.loc[CGP_df['Year_of_Commission'].dropna().index]
print('Emis data prepped and loaded')

######## Country mask and dataframe ######
country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_50
country_df = geopandas.read_file(f'{utils.raw_data_in_path}/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
countries = ['China','Australia', 'India','Myanmar', 'Cambodia', 'Laos','Philippines','Nepal','Bangladesh','Thailand','Bhutan','Brunei','Singapore', 'Papua New Guinea', 'Solomon Islands', 'East Timor', 'Taiwan', 'Malaysia', 'Vietnam','Indonesia']
country_df = country_df.rename(columns = {'SOVEREIGNT':'country'})

ds_area = xr.open_dataset('/net/fs11/d0/emfreese/GCrundirs/IRF_runs/stretch_2x_pulse/SEA/Jan/mod_output/GEOSChem.SpeciesConc.20160101_0000z.nc4', engine = 'netcdf4')
utils.fix_area_ij_latlon(ds_area);

########## Create emissions profile for each plant over our shutdown times ##########
E_CO2_all_opts = {}
year = 1
typical_shutdown_years = 40
for unique_id in CGP_df.loc[CGP_df['BC_(g/day)'] >0]['unique_ID'].values:
    E_CO2_all_opts[unique_id] = utils.individual_plant_shutdown(year, CGP_df, time_array, typical_shutdown_years, unique_id, min_year)
print('Emissions profiles based on weighted capacity of CO2 emissions percentiles created')

############### Convolution, selection of location, health impact assessment ##########
    
## Dictionary for location and time names in the Green's Functions (season and location dependent)
country_emit_dict = {'INDONESIA':['Indo_Jan', 'Indo_Apr', 'Indo_July','Indo_Oct'], 
                     'CAMBODIA':['Cambod_Jan', 'Cambod_Apr', 'Cambod_July','Cambod_Oct'], 
                     'MALAYSIA': ['Malay_Jan','Malay_Apr','Malay_July','Malay_Oct'], 
                     'VIETNAM': ['Viet_Jan','Viet_Apr','Viet_July','Viet_Oct']}

#import the green's function and set our time step
G = xr.open_dataarray(f'{utils.GF_name_path}/G_combined.nc', chunks = 'auto')

#column sum Green's function, only select our country of emissions
G_column_sum = G.where(G.run.isin(country_emit_dict[country_emit]), drop = True).sum(dim = 'lev').compute()
G_column_sum = G_column_sum.where((G_column_sum > 0), drop = True).rename({'time':'s'})

#select only the surface level for concentration at the surface, only select our country of emissions
G_lev0 = G.where(G.run.isin(country_emit_dict[country_emit]), drop = True).isel(lev = 0).compute()
G_lev0 = G_lev0.where((G_lev0 > 0), drop = True).rename({'time':'s'})

print('G prepped')

## convolution
for unique_id in CGP_df.loc[CGP_df['BC_(g/day)'] >0]['unique_ID']: 
    print(unique_id)
    
    C_conv_sfc = {}
    C_conv_column = {}
    print(E_CO2_all_opts[unique_id])
    
    if E_CO2_all_opts[unique_id].sum()>0:
        n = np.unique([int(i) for i, x in enumerate(E_CO2_all_opts[unique_id]>0) if x])[0]
            
        for C_dict, G_ds, nm in zip([C_conv_sfc, C_conv_column], [G_lev0, G_column_sum], ['surface', 'column']): 

            for idx, season in enumerate(season_days.keys()):
                C_dict[season] = signal.convolve(G_ds.sel(run = country_emit_dict[country_emit][idx]).fillna(0), 
                             E_CO2_all_opts[unique_id][n:n+season_days[season]][..., None, None], mode = 'full')
                
                #switch the tail (that goes into the following months) to follow the GF of the next month 
                if idx == 0 or idx == 1 or idx == 2:
                    idx_next = idx + 1
                elif idx == 3:
                    idx_next = 0

                tail_switch = signal.convolve(G_ds.sel(run = country_emit_dict[country_emit][idx_next]).fillna(0), 
                                     E_CO2_all_opts[unique_id][n:n+season_days[season]][..., None, None], mode = 'full')

                C_dict[season][season_days[season]:tail_switch[season_days[season]:C_dict[season].shape[0]].shape[0]+season_days[season]] = tail_switch[season_days[season]:C_dict[season].shape[0]]

                n = n + season_days[season]

            C = {}

            C['DJF'] = sparse.COO.from_numpy(np.pad(C_dict['DJF'],((((0),
                                               (365 - len(C_dict['DJF'])))),
                                             (0,0),(0,0))))
            C['MAM'] = sparse.COO.from_numpy(np.pad(C_dict['MAM'],((((season_days['DJF']),
                                               (365 - season_days['DJF'] - len(C_dict['MAM'])))),
                                             (0,0),(0,0))))
            C['JJA'] = sparse.COO.from_numpy(np.pad(C_dict['JJA'],((((season_days['DJF'] + season_days['MAM'] ),
                                               (365 - season_days['DJF'] - season_days['MAM'] - len(C_dict['JJA'])))),
                                             (0,0),(0,0))))
            C['SON'] = sparse.COO.from_numpy(np.pad(C_dict['SON'][:season_days['SON']], ((((season_days['DJF'] + season_days['MAM'] + season_days['JJA']),
                                                (365 - season_days['DJF'] - season_days['MAM'] - season_days['JJA'] - len(C_dict['SON'][:season_days['SON']])))),
                                              (0,0),(0,0))))
            C_sum = C['DJF']+C['MAM']+C['JJA']+C['SON']

            C_dense = sparse.COO.todense(C_sum)
            
            C_out = utils.np_to_xr_time_specific(C_dense, G_ds, E_CO2_all_opts[unique_id], time_init = np.unique([i for i, x in enumerate(E_CO2_all_opts[unique_id]>0) if x])[0])
       
            # Store gridded data as datasets
            if nm == 'surface':
                C_out_surface = C_out.to_dataset(name='BC_surface_conc')
            elif nm == 'column':
                C_out_column = C_out.to_dataset(name='BC_column_conc')
        
        # Merge surface and column data
        gridded_output = xr.merge([C_out_surface, C_out_column])
        
        # Add metadata
        gridded_output.attrs['country_emit'] = country_emit
        gridded_output.attrs['unique_id'] = unique_id
        gridded_output.attrs['description'] = 'Gridded BC concentration from single plant emissions'
        
        # Save as NetCDF
        output_path = f'../../data_output/convolution/gridded_convolution_{country_emit}_{unique_id}_uniqueid.nc'
        gridded_output.to_netcdf(output_path)
        
        print(f'saved out {country_emit}, {unique_id} unique id, gridded data')
        print(f'Memory available: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:.2f}%')
        
        # Clean up
        del C_sum
        del C_dense
        del C_out_surface
        del C_out_column
        del gridded_output
        gc.collect()
        print(f'Memory available after cleanup: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total:.2f}%')