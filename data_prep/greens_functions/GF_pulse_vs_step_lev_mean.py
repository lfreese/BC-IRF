#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=edr

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import regionmask
import pandas as pd
from datetime import datetime, timedelta
import utils
from matplotlib.colors import SymLogNorm
import xesmf as xe
from matplotlib import pyplot as plt, animation
from IPython.display import HTML, display

from scipy import signal


# In[2]:



import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})


# # Options

# In[3]:


r = ['all_countries']
month = 'Jan' #options are Jan, Apr, Jul, Oct
time = '20160'
length_simulation = 60 #days
diagnostic = 'SpeciesConc'


# ## Import data

dict_conc = {}
dict_emis = {}

#15x step for GF
dict_conc['all_countries_step'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_2x/all_countries/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')
#pulse run
dict_conc['all_countries_pulse'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_2x_pulse/all_countries/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')
#1.5e-10 addition run
dict_conc['all_countries_add'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_step/all_countries_add/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')
#complex test run
dict_conc['all_countries_cos'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_step/all_countries_cos/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords')

#15x step for GF
dict_emis['all_countries_step'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_2x/all_countries/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
#pulse run
dict_emis['all_countries_pulse'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_2x_pulse/all_countries/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
#1.5e-10 addition run
dict_emis['all_countries_add'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_step/all_countries_add/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
#complex test run
dict_emis['all_countries_cos'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_step/all_countries_cos/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')

#base run
dict_conc['base'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_base/template/Jan/mod_output/GEOSChem.{diagnostic}.{time}*', combine = 'by_coords', engine = 'netcdf4')
dict_emis['base'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_base/template/Jan/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords', engine = 'netcdf4')


#combine data
ds_conc = xr.concat([dict_conc[r] for r in dict_conc.keys()], pd.Index([r for r in dict_conc.keys()], name='run'))
ds_emis = xr.concat([dict_emis[r] for r in dict_emis.keys()], pd.Index([r for r in dict_emis.keys()], name='run'))

#modify data
ds_conc['time'] = ds_conc['time'].astype('datetime64')
utils.switch_conc_time(ds_conc)
#fix the area
ds_conc = utils.fix_area_ij_latlon(ds_conc)
#sum all BC conc
ds_conc['BC_total'] = ds_conc['SpeciesConc_BCPI'] + ds_conc['SpeciesConc_BCPO']
ds_emis = utils.fix_area_ij_latlon(ds_emis)
ds_emis = ds_emis.isel(lev = -1) #select surface since we only have surface emissions
utils.combine_BC(ds_emis)


### Add height to data
height_ds = utils.height_ds
ds_conc = xr.merge([ds_conc, height_ds], join = 'inner')


### convert to correct units
def ppb_to_ug(ds, species_to_convert, mw_species_list, P, T):
    '''Convert species to ug/m3 from ppb'''
    R = 8.314 #J/K/mol
    mol_per_m3= (P / (T * R)) #Pa/K/(J/K/mol) = mol/m3
    
    for spec in species_to_convert:
        attrs = ds[spec].attrs
        ds[spec] = ds[spec]*mw_species_list[spec]*mol_per_m3*1e-3 #ppb*g/mol*mol/m3*ug/ng
        ds[spec].attrs['units'] = 'μg m-3'

mw_BC = {'BC_total':12.011}

T_p_ds = xr.open_mfdataset('/net/geoschem/data/gcgrid/data/ExtData/GEOS_0.5x0.625/MERRA2/2016/*/MERRA2.2016*.I3.05x0625.nc4')
T_p_ds = T_p_ds.groupby('time.date').mean(dim = 'time').rename({'date':'time'})

#create regridder (reusing weights)
regridder = xe.Regridder(T_p_ds, dict_conc[list(dict_conc.keys())[0]], 'bilinear', reuse_weights = True, weights = 'tp_bilinear_weights.nc')
regridder  # print basic regridder information.

#regrid according to our ds_out grid
T_p_ds = regridder(T_p_ds)
T_p_ds = T_p_ds.isel(time = slice(0, len(ds_conc['time'])))
T_p_ds['time'] = ds_conc['time']

pressure_ds = utils.pressure_ds

print(T_p_ds)
ds_conc['BC_total'] = ds_conc['BC_total']*1e9 #convert mol/mol to ppb
ppb_to_ug(ds_conc, ['BC_total'], mw_BC, pressure_ds, T_p_ds['T'])

print(ds_conc)


# ## Define our Conc Difference and Initial Forcing

sec_day = 86400
runs = ['all_countries_pulse','all_countries_step', 'all_countries_add']
poll_name = 'BC_total'
dt = 1 #day


#shift our time so that it is halfway through the day to represent the daily mean

print(ds_conc)
print('finish ds prep')


#calculate the dc/dt
dict_dc_dt = {}
dict_dc_dt_gmean = {}
dict_dc_dt_gmean_lev0 = {}
f0 = {}
for r in ['all_countries_step', 'all_countries_add']: #,
    #change in concentration over time
    dict_dc_dt_gmean[r] = utils.calc_δc_δt_mean(ds_conc, poll_name, r, 'base')
    dict_dc_dt_gmean[r] = dict_dc_dt_gmean[r].assign_coords(time = np.arange(.5,len(dict_dc_dt_gmean[r]['time'])+.5))

    dict_dc_dt[r] = utils.calc_δc_δt(ds_conc, poll_name, r, 'base')
    dict_dc_dt[r] = dict_dc_dt[r].assign_coords(time = np.arange(.5,len(dict_dc_dt[r]['time'])+.5))

    dict_dc_dt_gmean_lev0[r] = utils.calc_δc_δt_lev0_mean(ds_conc, poll_name, r, 'base')
    dict_dc_dt_gmean_lev0[r] = dict_dc_dt_gmean_lev0[r].assign_coords(time = np.arange(.5,len(dict_dc_dt_gmean_lev0[r]['time'])+.5))

for r in ['all_countries_step', 'all_countries_add']:
    #f0 calculation    
    f0[r] = (ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = r) - 
        ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = 'base')).isel(time = 0).values 
    
f0['all_countries_pulse'] = (ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = 'all_countries_pulse') - 
        ds_emis['EmisBC_Total'].weighted(ds_emis['area'].fillna(0)).sum(dim = ['lat','lon']).sel(run = 'base')).isel(time = 0).values


# In[19]:


#create the dataset of our dc/dt
dc_dt = xr.concat([dict_dc_dt[r] for r in dict_dc_dt.keys()], pd.Index([r for r in dict_dc_dt.keys()], name='run'))
dc_dt_gmean = xr.concat([dict_dc_dt_gmean[r] for r in dict_dc_dt_gmean.keys()], pd.Index([r for r in dict_dc_dt_gmean.keys()], name='run'))
dc_dt_gmean_lev0 = xr.concat([dict_dc_dt_gmean_lev0[r] for r in dict_dc_dt_gmean_lev0.keys()], pd.Index([r for r in dict_dc_dt_gmean_lev0.keys()], name='run'))


dc_dt = dc_dt.rename({'time':'s'})
dc_dt_gmean = dc_dt_gmean.rename({'time':'s'})
dc_dt_gmean_lev0 = dc_dt_gmean_lev0.rename({'time':'s'})

# In[20]:
print('create G')

#calculate the Green's function as dc/dt/f0
G_dict = {}
G_dict_gmean = {}
G_dict_gmean_lev0 = {}
for r in ['all_countries_step', 'all_countries_add']:
    G_dict[r] = dc_dt.sel(run = r)/f0[r]#[:,np.newaxis, np.newaxis, np.newaxis]
    G_dict_gmean[r] = dc_dt_gmean.sel(run = r)/f0[r]
    G_dict_gmean_lev0[r] = dc_dt_gmean_lev0.sel(run = r)/f0[r]
print('test output', G_dict_gmean['all_countries_add'].max().values)
# In[21]:


G_dict['all_countries_pulse'] = (ds_conc.sel(run ='all_countries_pulse')-ds_conc.sel(run ='base'))['BC_total']/f0['all_countries_pulse']
G_dict_gmean['all_countries_pulse'] = (ds_conc.sel(run ='all_countries_pulse')-ds_conc.sel(run ='base'))['BC_total'].weighted(ds_conc['area'].sel(run = 'base').fillna(0)*
                                                                                                          ds_conc['Altitude'].fillna(0)).mean(['lat','lon','lev'])/f0['all_countries_pulse']
G_dict_gmean_lev0['all_countries_pulse'] = (ds_conc.sel(run ='all_countries_pulse')-ds_conc.sel(run ='base'))['BC_total'].weighted(ds_conc['area'].sel(run = 'base').fillna(0)).mean(['lat','lon'])/f0['all_countries_pulse']

# In[22]:
print(G_dict['all_countries_pulse'])
#change the time to be delta time
# for run in G_dict.keys(): 
#     G_dict[run]['time'] = G_dict[run]['time']-G_dict[run]['time'][0]
#     G_dict_gmean[run]['time'] = G_dict_gmean[run]['time']-G_dict_gmean[run]['time'][0]
#     G_dict_gmean_lev0[run]['time'] = G_dict_gmean_lev0[run]['time']-G_dict_gmean_lev0[run]['time'][0]
#     print(G_dict[run]['time'])

G_dict_gmean['all_countries_pulse'] = G_dict_gmean['all_countries_pulse'].rename({'time':'s'})
G_dict_gmean['all_countries_pulse']['s'] = G_dict_gmean['all_countries_step']['s'].values
print('test output', G_dict_gmean['all_countries_pulse'].max().values)

G_dict_gmean_lev0['all_countries_pulse'] = G_dict_gmean_lev0['all_countries_pulse'].rename({'time':'s'})
G_dict_gmean_lev0['all_countries_pulse']['s'] = G_dict_gmean_lev0['all_countries_step']['s'].values

# In[23]:


G_dict['all_countries_pulse'] = G_dict['all_countries_pulse'].rename({'time':'s'})
G_dict['all_countries_pulse']['s'] = G_dict['all_countries_step']['s'].values


# In[24]:


#convert to datasets, calculate mean GF
G = xr.concat([G_dict[r] for r in G_dict.keys()], pd.Index([r for r in G_dict.keys()], name='run'))
G_mean = xr.concat([G_dict_gmean[r] for r in G_dict_gmean.keys()], pd.Index([r for r in G_dict_gmean.keys()], name='run'))
G_mean_lev0 = xr.concat([G_dict_gmean_lev0[r] for r in G_dict_gmean_lev0.keys()], pd.Index([r for r in G_dict_gmean_lev0.keys()], name='run'))

print('created, saving')


ds_conc_mean = ds_conc.weighted(ds_conc['area'].sel(run = 'base').fillna(0)*ds_conc['Altitude'].fillna(0)).mean(['lat','lon','lev'])['BC_total']
ds_conc_lev0_mean = ds_conc.weighted(ds_conc['area'].sel(run = 'base').fillna(0)).mean(['lat','lon']).isel(lev = 0)['BC_total']

# ds_conc['time'] = ds_conc['time'] - ds_conc['time'][0]
# ds_conc_mean['time'] = ds_conc_mean['time'] - ds_conc_mean['time'][0]
# ds_conc_lev0_mean['time'] = ds_conc_lev0_mean['time'] - ds_conc_lev0_mean['time'][0]

ds_conc_lev0_mean.to_netcdf('Outputs/ds_conc_lev0_mean_step_v_pulse.nc')
ds_conc.to_netcdf('Outputs/ds_conc_step_v_pulse.nc')
ds_conc_mean.to_netcdf('Outputs/ds_conc_mean_step_v_pulse.nc')

G_mean_lev0.to_netcdf('Outputs/g_lev0_mean_step_v_pulse.nc')
G.to_netcdf('Outputs/g_step_v_pulse.nc')
G_mean.to_netcdf('Outputs/g_mean_step_v_pulse.nc')

ds_emis.to_netcdf('Outputs/ds_emis_step_v_pulse.nc')
