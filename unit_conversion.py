#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=fdr

import xarray as xr
import matplotlib.pyplot as plt


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


import itertools

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs



regions = ['SEA', 'Indo','Malay','all_countries','Viet','Cambod']
months = ['Jan','Apr','July', 'Oct'] #options are Jan, Apr, July, Oct
time = '2016'
#compare_2x = True #True allows for comparison to the 2x simulation to see the performance of the GF; False just creates a new GF
#global_mean = True #False turns on a spatially explicit version with lat and lon; True turns on a global weighted mean
length_simulation = 60 #days
diagnostic = 'SpeciesConc'


# In[3]:


for m in months:
    print(m)


# ## Import monthly data for 2x

dict_conc = {}
dict_emis = {}


pulse_size = '2x'
for r in regions:
    for m in months:
        print(m)
        print(r)
        #2x pulse for GF
        dict_conc[r + '_' + m] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/{r}/{m}/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords')
        #2x pulse for GF
        dict_emis[r + '_' + m] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/{r}/{m}/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
        if (dict_conc[r + '_' + m]['time'].diff('time').astype('float64') > 86400000000000).any():
            print('CHECK TIME, FAILED')

for m in months:
    dict_conc[f'base_{m}'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_base/template/{m}/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords', engine = 'netcdf4')
    dict_emis[f'base_{m}'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_base/template/{m}/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords', engine = 'netcdf4')
    if (dict_conc[f'base_{m}']['time'].diff('time').astype('float64') > 86400000000000).any():
            print('CHECK TIME, FAILED')


# ## Import 16x for comparison

pulse_size = '16x'
for r in regions:
    for m in ['Jan']:
        #2x pulse for GF
        dict_conc[r + '_' + m + '_' + pulse_size] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/{r}/{m}/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords')
        #2x pulse for GF
        dict_emis[r + '_' + m+ '_' + pulse_size] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/{r}/{m}/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
        
        if (dict_conc[r + '_' + m + '_' + pulse_size]['time'].diff('time').astype('float64') > 86400000000000).any():
            print('CHECK TIME, FAILED-- Check the regridding as not all times were regridded')




for m in months:
    dict_conc[f'all_countries_summed_{m}'] = dict_conc[f'Indo_{m}'] + dict_conc[f'Malay_{m}'] + dict_conc[f'Viet_{m}'] + dict_conc[f'Cambod_{m}']


### Import Data across January Comparison


days = ['6', '11', '16', '21', '26']
pulse_size = '2x'
for d in days:
    print(d)
    #2x pulse for GF
    dict_conc['Indo_Jan_' + d] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/Jan_uncertainty_Indo/Jan{d}_pulse/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords')
    #2x pulse for GF
    dict_emis['Indo_Jan_' + d] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/Jan_uncertainty_Indo/Jan{d}_pulse/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
    
    if (dict_conc['Indo_Jan_' + d]['time'].diff('time').astype('float64') > 86400000000000).any():
        print('CHECK TIME, FAILED-- Check the regridding as not all times were regridded')
        
for d in days:
    dict_conc[f'base_Jan_{d}'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_base/template/Jan{d}_pulse/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords', engine = 'netcdf4')
    dict_emis[f'base_Jan_{d}'] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_base/template/Jan{d}_pulse/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords', engine = 'netcdf4')


### Import Data across Indonesia location comparison


locations = ['Center','NW','SE']
pulse_size = '2x'
for loc in locations:
    print(loc)
    #2x pulse for GF
    dict_conc['Indo_Jan_' + loc] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/loc_uncertainty_Indo/{loc}_pulse/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords')
    #2x pulse for GF
    dict_emis['Indo_Jan_' + loc] = xr.open_mfdataset(f'../GCrundirs/IRF_runs/stretch_{pulse_size}_pulse/loc_uncertainty_Indo/{loc}_pulse/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
    if (dict_conc['Indo_Jan_' + loc]['time'].diff('time').astype('float64') > 86400000000000).any():
        print('CHECK TIME, FAILED-- Check the regridding as not all times were regridded')

print('Data imported')

### Make dataset and modify time

#shift our time so that it is halfway through the day to represent the daily mean
for i in dict_emis.keys():
    utils.switch_conc_time(dict_emis[i])
    #shift our time so that it is halfway through the day to represent the daily mean
for i in dict_conc.keys():
    utils.switch_conc_time(dict_conc[i])
    #change the time to be delta time
    dict_conc[i]['time'] = dict_conc[i]['time']-dict_conc[i]['time'][0]
    #fix the area
    dict_conc[i]= utils.fix_area_ij_latlon(dict_conc[i])
    #sum all BC conc
    dict_conc[i]['BC_total'] = dict_conc[i]['SpeciesConc_BCPI'] + dict_conc[i]['SpeciesConc_BCPO']
    
for i in dict_emis.keys():
    #change the time to be delta time
    dict_emis[i]['time'] = dict_emis[i]['time']-dict_emis[i]['time'][0]
    #combine the BCPI and BCPO emissions
    utils.combine_BC(dict_emis[i])
    #fix the area
    dict_emis[i] = utils.fix_area_ij_latlon(dict_emis[i])
    
print('time shifted')

### Add height to data

height = pd.read_excel('gc_72_estimate.xlsx', index_col = 0)
height = height.reindex(index=height.index[::-1])
height_ds = height.diff().dropna().to_xarray().rename({'L':'lev'})
height_ds = height_ds.rename({'Altitude (km)':'dz'}) 
height_ds['dz']*=1e3 #convert to meters
height_ds['dz'].attrs = {'units':'m'}



### Define our Conc Difference and Initial Forcing
poll_name = 'BC_total'
dt = 1 #day


### initial forcing
f0 = {}
for r in regions:
    for m in months:
        f0[r + '_' + m] = (dict_emis[r + '_' + m]['EmisBC_Total'].weighted(dict_emis[r + '_' + m]['area'].fillna(0)).sum(dim = ['lat','lon']) - 
                dict_emis[f'base_{m}']['EmisBC_Total'].weighted(dict_emis[f'base_{m}']['area'].fillna(0)).sum(dim = ['lat','lon'])).isel(lev = -1).isel(time = 0).values ## multiply by area and time (1 day)

for m in months:
    f0[f'all_countries_summed_{m}'] = f0[f'Indo_{m}'] + f0[f'Malay_{m}'] + f0[f'Viet_{m}'] + f0[f'Cambod_{m}']

m = 'Jan'
for r in regions:
    f0[r + '_' + m + '_16x'] = (dict_emis[r + '_' + m + '_16x']['EmisBC_Total'].weighted(dict_emis[r + '_' + m + '_' + '16x']['area'].fillna(0)).sum(dim = ['lat','lon']) - 
            dict_emis[f'base_{m}']['EmisBC_Total'].weighted(dict_emis[f'base_{m}']['area'].fillna(0)).sum(dim = ['lat','lon'])).isel(lev = -1).isel(time = 0).values


for d in days:
    f0[f'Indo_Jan_{d}'] = (dict_emis[f'Indo_Jan_{d}']['EmisBC_Total'].weighted(dict_emis[f'Indo_Jan_{d}']['area'].fillna(0)).sum(dim = ['lat','lon']) - 
            dict_emis[f'base_Jan_{d}']['EmisBC_Total'].weighted(dict_emis[f'base_Jan_{d}']['area'].fillna(0)).sum(dim = ['lat','lon'])).isel(lev = -1).isel(time = 0).values


for d in locations:
    f0['Indo_Jan_' + d] = (dict_emis['Indo_Jan_' + d]['EmisBC_Total'].weighted(dict_emis['Indo_Jan_' + d]['area']).sum(dim = ['lat','lon']) - 
            dict_emis['base_Jan']['EmisBC_Total'].weighted(dict_emis['base_Jan']['area']).sum(dim = ['lat','lon'])).isel(lev = -1).isel(time = 0).values

print('F0, initial forcing, complete')



t_data = xr.open_mfdataset('')
### convert to correct units
def ppb_to_ug(ds, species_to_convert, mw_species_list, stp_p = 101325, stp_t = 298.):
    '''Convert species to ug/m3 from ppb'''
    R = 8.314 #J/K/mol
    mol_per_m3= (stp_p / (stp_t * R)) #Pa/K/(J/K/mol) = mol/m3
    
    for spec in species_to_convert:
        attrs = ds[spec].attrs
        x = ds[spec]*mw_species_list[spec]*mol_per_m3*1e-3 #ppb*g/mol*mol/m3*ug/ng
        ds[spec] = x
        ds[spec].attrs['units'] = 'μg m-3'
    return(x)

mw_BC = {'BC_total':12.011}

for i in dict_conc.keys():
    dict_conc[i]['BC_total'] = dict_conc[i]['BC_total']*1e9 #convert mol/mol to ppb
    ppb_to_ug(dict_conc[i], [poll_name], mw_BC)

print('Converted units')









### calculate the Green's function as dc/f0

G_dict = {}
G_dict_gmean = {}
regions = ['SEA', 'Indo','Malay','all_countries','Viet','Cambod']
    
for r in regions:
    for m in months:
        G_dict[r + '_' + m] = (dict_conc[r + '_' + m]-dict_conc[f'base_{m}'])['BC_total']/f0[r + '_' + m]
        G_dict_gmean[r + '_' + m] = (dict_conc[r + '_' + m]-dict_conc[f'base_{m}'])['BC_total'].weighted(
            dict_conc[f'base_{m}']['area'].fillna(0)*height_ds['dz'].fillna(0)).mean(['lat','lon','lev'])/f0[r + '_' + m]



for m in months:
    G_dict[f'all_countries_summed_{m}'] = (((dict_conc['Indo_' + m]-dict_conc[f'base_{m}'])['BC_total']+ 
                                          (dict_conc['Malay_' + m]-dict_conc[f'base_{m}'])['BC_total']+
                                          (dict_conc['Viet_' + m]-dict_conc[f'base_{m}'])['BC_total']+
                                          (dict_conc['Cambod_' + m]-dict_conc[f'base_{m}'])['BC_total'])/
                                           (f0['Indo_Jan'] + f0['Viet_Jan'] + f0['Malay_Jan'] + f0['Cambod_Jan']))
    G_dict_gmean[f'all_countries_summed_{m}'] = (((dict_conc['Indo_' + m]-dict_conc[f'base_{m}'])['BC_total']+ 
                                          (dict_conc['Malay_' + m]-dict_conc[f'base_{m}'])['BC_total']+
                                          (dict_conc['Viet_' + m]-dict_conc[f'base_{m}'])['BC_total']+
                                          (dict_conc['Cambod_' + m]-dict_conc[f'base_{m}'])['BC_total']).weighted(
        dict_conc[f'base_{m}']['area'].fillna(0)*height_ds['dz'].fillna(0)).mean(['lat','lon','lev'])/
 (f0['Indo_Jan'] + f0['Viet_Jan'] + f0['Malay_Jan'] + f0['Cambod_Jan']))
    
    


m = 'Jan'
for r in regions:
    G_dict[r + '_' + m + '_16x'] = (dict_conc[r + '_' + m + '_16x']-dict_conc[f'base_{m}'])['BC_total']/f0[r + '_' + m + '_16x']
    G_dict_gmean[r + '_' + m + '_16x'] = (dict_conc[r + '_' + m]-dict_conc[f'base_{m}'])['BC_total'].weighted(
        dict_conc[f'base_{m}']['area'].fillna(0)*height_ds['dz'].fillna(0)).mean(['lat','lon','lev'])/f0[r + '_' + m + '_16x']


for d in days:
    G_dict['Indo_Jan_' + d] = (dict_conc['Indo_Jan_' + d]-dict_conc[f'base_Jan_{d}'])['BC_total']/f0['Indo_Jan_' + d]
    G_dict_gmean['Indo_Jan_' + d] = (dict_conc['Indo_Jan_' + d]-dict_conc[f'base_Jan_{d}'])['BC_total'].weighted(
        dict_conc[f'base_Jan_{d}']['area'].fillna(0)*height_ds['dz'].fillna(0)).mean(['lat','lon','lev'])/f0['Indo_Jan_' + d]


for d in locations:
    G_dict['Indo_Jan_' + d] = (dict_conc['Indo_Jan_' + d]-dict_conc[f'base_Jan'])['BC_total']/f0['Indo_Jan_' + d]
    G_dict_gmean['Indo_Jan_' + d] = (dict_conc['Indo_Jan_' + d]-dict_conc[f'base_Jan'])['BC_total'].weighted(
        dict_conc[f'base_Jan']['area'].fillna(0)*height_ds['dz'].fillna(0)).mean(['lat','lon','lev'])/f0['Indo_Jan_' + d]
print('Calculated GF')




### Add tail to zero based on the mean at that level

import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})

def exponential_decay(a, b, N):
    return a * (1-b) ** np.arange(N)

full_ds = {}
for r in G_dict.keys():
    exp_decay = exponential_decay(1, 0.5, 13)
    exp_decay = np.append(exp_decay, 0)
    dates = pd.date_range(start='1/1/2018', end='1/14/2018')
    times = dates - dates[0] + timedelta(days = len(G_dict[r]['time']))
    exp_da = xr.DataArray(data = exp_decay,
                 dims = ['time'],
                 coords = dict(time=times))
    exp_app = G_dict[r].isel(time = -1)*exp_da
    full_ds[r] = xr.concat([G_dict[r], exp_app], dim = 'time')
print('Added tail')


### convert to datasets, calculate mean GF
G = xr.concat([full_ds[r] for r in full_ds.keys()], pd.Index([r for r in full_ds.keys()], name='run'))
G_mean = xr.concat([G_dict_gmean[r] for r in G_dict_gmean.keys()], pd.Index([r for r in G_dict_gmean.keys()], name='run'))


### Save out the Green's function
G.to_netcdf(f'Outputs/new_G_all_loc_all_times_{poll_name}.nc4', mode = 'w')
G_mean.to_netcdf(f'Outputs/new_G_mean_all_loc_all_times_{poll_name}.nc4',  mode = 'w')


