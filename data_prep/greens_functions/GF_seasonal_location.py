#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --partition=edr
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=emfreese@mit.edu


import xarray as xr
import numpy as np
import numpy as np
import argparse
import dask
dask.config.set(**{'array.slicing.split_large_chunks': True})
import sys
sys.path.insert(0, '/net/fs11/d0/emfreese/BC-IRF/')
import utils


# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--region',type=str, required=True) # one of ['SEA', 'Indo','Malay']#,'all_countries','Viet','Cambod']

args = parser.parse_args()

reg = args.region

regions = [reg]
months = ['Jan','Apr','July', 'Oct']
time = '2016'
length_simulation = 60 #days
diagnostic = 'SpeciesConc'


dict_conc = {}
dict_emis = {}

##Import pulse 2x data
pulse_size = '2x'
for r in regions:
    for m in months:
        print(m)
        print(r)
        #2x pulse for GF
        dict_conc[r + '_' + m] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_{pulse_size}_pulse/{r}/{m}/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords')
        #2x pulse for GF
        dict_emis[r + '_' + m] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_{pulse_size}_pulse/{r}/{m}/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords')
        if (dict_conc[r + '_' + m]['time'].diff('time').astype('float64') > 86400000000000).any():
            print('CHECK TIME, FAILED')

##Import base data
for m in months:
    dict_conc[f'base_{m}'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_base/template/{m}/mod_output/GEOSChem.SpeciesConc.{time}*', combine = 'by_coords', engine = 'netcdf4')
    dict_emis[f'base_{m}'] = xr.open_mfdataset(f'{utils.geos_chem_data_path}stretch_base/template/{m}/mod_output/GEOSChem.Emissions.{time}*', combine = 'by_coords', engine = 'netcdf4')
    if (dict_conc[f'base_{m}']['time'].diff('time').astype('float64') > 86400000000000).any():
            print('CHECK TIME, FAILED')


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

height = pd.read_excel(f'{utils.raw_data_in_path}gc_72_estimate.xlsx', index_col = 0)
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


print('F0, initial forcing, complete')



### convert to correct units
mw_BC = {'BC_total':12.011}

T_p_ds = xr.open_mfdataset('/net/geoschem/data/gcgrid/data/ExtData/GEOS_0.5x0.625/MERRA2/2016/*/MERRA2.2016*.I3.05x0625.nc4')
T_p_ds = T_p_ds.groupby('time.date').mean(dim = 'time').rename({'date':'time'})
#create regridder (reusing weights)
regridder = xe.Regridder(T_p_ds, dict_conc[list(dict_conc.keys())[0]], 'bilinear', reuse_weights = True, weights = 'tp_bilinear_weights.nc')
regridder  # print basic regridder information.

#regrid according to our ds_out grid
T_p_ds = regridder(T_p_ds)
T_p_ds['time']  = T_p_ds['time'] - T_p_ds['time'][0]

pressure_ds = utils.pressure_ds


for i in dict_conc.keys():
    dict_conc[i]['BC_total'] = dict_conc[i]['BC_total']*1e9 #convert mol/mol to ppb
    utils.ppb_to_ug(dict_conc[i], [poll_name], mw_BC, pressure_ds, T_p_ds['T'])

del(T_p_ds)
del(pressure_ds)
import gc
gc.collect()

### calculate the Green's function as dc/f0

G_dict = {}
    
for r in regions:
    for m in months:
        G_dict[r + '_' + m] = (dict_conc[r + '_' + m]-dict_conc[f'base_{m}'])['BC_total']/f0[r + '_' + m]

print('Calculated GF')


### Add tail to zero based on the mean at that level
full_ds = {}
for r in G_dict.keys():
    exp_decay = utils.exponential_decay(1, 0.5, 13)
    exp_decay = np.append(exp_decay, 0)
    dates = pd.date_range(start='1/1/2018', end='1/14/2018')
    times = dates - dates[0] + timedelta(days = len(G_dict[r]['time']))
    exp_da = xr.DataArray(data = exp_decay,
                 dims = ['time'],
                 coords = dict(time=times))
    full_ds[r] = xr.concat([G_dict[r], G_dict[r].isel(time = -1)*exp_da], dim = 'time')


### Save out the Green's function
xr.concat([full_ds[r] for r in full_ds.keys()], pd.Index([r for r in full_ds.keys()], name='run')).to_netcdf(f'{utils.data_output_path}/greens_functions/Greens_function_{poll_name}_{reg}_allseasons.nc4', mode = 'w')


