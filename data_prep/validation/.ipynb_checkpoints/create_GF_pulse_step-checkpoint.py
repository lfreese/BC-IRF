#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=edr

import xarray as xr
import numpy as np
import regionmask
import pandas as pd
from datetime import datetime, timedelta
import xesmf as xe

from scipy import signal


import sys

sys.path.insert(0, '/net/fs11/d0/emfreese/BC-IRF')
import utils


import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})


# # Options

r = ['all_countries']
time = '20160'
diagnostic = 'SpeciesConc'


############### Import Data ###############

dict_conc = {}
dict_emis = {}

#### Concentration

#15x step for GF
dict_conc['all_countries_step'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_2x/all_countries/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')
#pulse run
dict_conc['all_countries_pulse'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_2x_pulse/all_countries/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')
#1.5e-10 addition run
dict_conc['all_countries_add'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_step/all_countries_add/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')
#complex test run
dict_conc['Indo_cos'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_step/Indo_cos/cos_jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')

#### Emissions

#15x step for GF
dict_emis['all_countries_step'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_2x/all_countries/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
#pulse run
dict_emis['all_countries_pulse'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_2x_pulse/all_countries/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
#1.5e-10 addition run
dict_emis['all_countries_add'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_step/all_countries_add/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
#complex test run
dict_emis['Indo_cos'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_step/Indo_cos/cos_jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')

#### Base run
dict_conc['base'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_base/template/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords', engine = 'netcdf4')
dict_emis['base'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_base/template/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords', engine = 'netcdf4')


#combine data
ds_conc = xr.concat([dict_conc[r] for r in dict_conc.keys()], pd.Index([r for r in dict_conc.keys()], name='run'))
ds_emis = xr.concat([dict_emis[r] for r in dict_emis.keys()], pd.Index([r for r in dict_emis.keys()], name='run'))

#correct times/datetime type
ds_conc['time'] = ds_conc['time'].astype('datetime64')
utils.switch_conc_time(ds_conc)
#fix the area
ds_conc = utils.fix_area_ij_latlon(ds_conc)
#sum all BC conc
ds_conc['BC_total'] = ds_conc['SpeciesConc_BCPI'] + ds_conc['SpeciesConc_BCPO']
ds_emis = utils.fix_area_ij_latlon(ds_emis)
ds_emis = ds_emis.isel(lev = -1) #select surface since we only have surface emissions
utils.combine_BC(ds_emis)


#### Convert from ppb to ug/m3 

#Add height to data
height_ds = utils.height_ds
ds_conc = xr.merge([ds_conc, height_ds], join = 'inner')


### convert to correct units
mw_BC = {'BC_total':12.011}

T_p_ds = xr.open_mfdataset('/net/geoschem/data/gcgrid/data/ExtData/GEOS_0.5x0.625/MERRA2/2016/*/MERRA2.2016*.I3.05x0625.nc4')
T_p_ds = T_p_ds.groupby('time.date').mean(dim = 'time').rename({'date':'time'})

#create regridder (reusing weights)
regridder = xe.Regridder(T_p_ds, dict_conc[list(dict_conc.keys())[0]], 'bilinear', reuse_weights = True, weights = f'{utils.data_output_path}tp_bilinear_weights.nc')
regridder  # print basic regridder information.

#regrid according to our ds_out grid
T_p_ds = regridder(T_p_ds)
T_p_ds = T_p_ds.isel(time = slice(0, len(ds_conc['time'])))
T_p_ds['time'] = ds_conc['time']

pressure_ds = utils.pressure_ds

ds_conc['BC_total'] = ds_conc['BC_total']*1e9 #convert mol/mol to ppb
utils.ppb_to_ug(ds_conc, ['BC_total'], mw_BC, pressure_ds, T_p_ds['T'])


print(ds_conc)
print('finish ds prep')

############### Calculate the concentration differences (dc/dt) ###########
poll_name = 'BC_total'
#### calculate the dc/dt
dict_dc_dt = {}
dict_dc_dt_gmean = {}
dict_dc_dt_gmean_lev0 = {}
f0 = {}
delta_types = ['all_countries_step', 'all_countries_add']
#calculate step and addition difference
for r in delta_types: #,
    #change in concentration over time
    dict_dc_dt_gmean[r] = utils.calc_δc_δt_mean(ds_conc, poll_name, r, 'base')
    dict_dc_dt_gmean[r] = dict_dc_dt_gmean[r].assign_coords(time = np.arange(.5,len(dict_dc_dt_gmean[r]['time'])+.5))

    dict_dc_dt[r] = utils.calc_δc_δt(ds_conc, poll_name, r, 'base')
    dict_dc_dt[r] = dict_dc_dt[r].assign_coords(time = np.arange(.5,len(dict_dc_dt[r]['time'])+.5))

    dict_dc_dt_gmean_lev0[r] = utils.calc_δc_δt_lev0_mean(ds_conc, poll_name, r, 'base')
    dict_dc_dt_gmean_lev0[r] = dict_dc_dt_gmean_lev0[r].assign_coords(time = np.arange(.5,len(dict_dc_dt_gmean_lev0[r]['time'])+.5))

for r in delta_types:
    #f0 calculation    
    f0[r] = (ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = r) - 
        ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = 'base')).isel(time = 0).values 
    
#calculate pulse difference
f0['all_countries_pulse'] = (ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = 'all_countries_pulse') - 
        ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = 'base')).isel(time = 0).values


#### create the dataset of our dc/dt
dc_dt = xr.concat([dict_dc_dt[r] for r in dict_dc_dt.keys()], pd.Index([r for r in dict_dc_dt.keys()], name='run'))
#global mean
dc_dt_gmean = xr.concat([dict_dc_dt_gmean[r] for r in dict_dc_dt_gmean.keys()], pd.Index([r for r in dict_dc_dt_gmean.keys()], name='run'))
#surface global mean
dc_dt_gmean_lev0 = xr.concat([dict_dc_dt_gmean_lev0[r] for r in dict_dc_dt_gmean_lev0.keys()], pd.Index([r for r in dict_dc_dt_gmean_lev0.keys()], name='run'))


dc_dt = dc_dt.rename({'time':'s'})
dc_dt_gmean = dc_dt_gmean.rename({'time':'s'})
dc_dt_gmean_lev0 = dc_dt_gmean_lev0.rename({'time':'s'})

print('finish calculating the differences in concentration')



############# Calculate the Green's function as dc/dt/f0 #########
G_dict = {}
G_dict_gmean = {}
G_dict_gmean_lev0 = {}
#calculate step and addition Green's function
for r in ['all_countries_step', 'all_countries_add']:
    G_dict[r] = dc_dt.sel(run = r)/f0[r]
    G_dict_gmean[r] = dc_dt_gmean.sel(run = r)/f0[r]
    G_dict_gmean_lev0[r] = dc_dt_gmean_lev0.sel(run = r)/f0[r]
print('test output', G_dict_gmean['all_countries_add'].max().values)

#calculate pulse Green's function
G_dict['all_countries_pulse'] = (ds_conc.sel(run ='all_countries_pulse')-ds_conc.sel(run ='base'))['BC_total']/f0['all_countries_pulse']
G_dict_gmean['all_countries_pulse'] = (ds_conc.sel(run ='all_countries_pulse')-ds_conc.sel(run ='base'))['BC_total'].weighted(ds_conc['area'].sel(run = 'base').fillna(0)*
                                                                                                          ds_conc['Altitude'].fillna(0)).mean(['lat','lon','lev'])/f0['all_countries_pulse']
G_dict_gmean_lev0['all_countries_pulse'] = (ds_conc.sel(run ='all_countries_pulse')-ds_conc.sel(run ='base'))['BC_total'].weighted(ds_conc['area'].sel(run = 'base').fillna(0)).mean(['lat','lon'])/f0['all_countries_pulse']

#rename the time variable
G_dict_gmean['all_countries_pulse'] = G_dict_gmean['all_countries_pulse'].rename({'time':'s'})
G_dict_gmean['all_countries_pulse']['s'] = G_dict_gmean['all_countries_step']['s'].values

G_dict_gmean_lev0['all_countries_pulse'] = G_dict_gmean_lev0['all_countries_pulse'].rename({'time':'s'})
G_dict_gmean_lev0['all_countries_pulse']['s'] = G_dict_gmean_lev0['all_countries_step']['s'].values

G_dict['all_countries_pulse'] = G_dict['all_countries_pulse'].rename({'time':'s'})
G_dict['all_countries_pulse']['s'] = G_dict['all_countries_step']['s'].values

#convert to datasets, calculate mean GF
G = xr.concat([G_dict[r] for r in G_dict.keys()], pd.Index([r for r in G_dict.keys()], name='run'))
G_mean = xr.concat([G_dict_gmean[r] for r in G_dict_gmean.keys()], pd.Index([r for r in G_dict_gmean.keys()], name='run'))
G_mean_lev0 = xr.concat([G_dict_gmean_lev0[r] for r in G_dict_gmean_lev0.keys()], pd.Index([r for r in G_dict_gmean_lev0.keys()], name='run'))

print('created, saving')

#################### Save out #####################
ds_conc_mean = ds_conc.weighted(ds_conc['area'].sel(run = 'base').fillna(0)*ds_conc['Altitude'].fillna(0)).mean(['lat','lon','lev'])['BC_total']
ds_conc_lev0_mean = ds_conc.weighted(ds_conc['area'].sel(run = 'base').fillna(0)).mean(['lat','lon']).isel(lev = 0)['BC_total']


ds_conc_lev0_mean.to_netcdf(f'{utils.data_output_path}validation/concentration_lev0_mean_step_v_pulse.nc')
ds_conc.to_netcdf(f'{utils.data_output_path}validation/concentration_step_v_pulse.nc')
ds_conc_mean.to_netcdf(f'{utils.data_output_path}validation/concentration_mean_step_v_pulse.nc')

G_mean_lev0.to_netcdf(f'{utils.data_output_path}validation/Greens_function_lev0_mean_step_v_pulse.nc')
G.to_netcdf(f'{utils.data_output_path}validation/Greens_function_step_v_pulse.nc')
G_mean.to_netcdf(f'{utils.data_output_path}validation/Greens_function_mean_step_v_pulse.nc')

ds_emis.to_netcdf(f'{utils.data_output_path}validation/emissions_step_v_pulse.nc')
