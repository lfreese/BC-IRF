import xarray as xr
import sys
sys.path.insert(0, '/net/fs11/d0/emfreese/BC-IRF/')
import utils


poll_name = 'BC_total'
ds = xr.open_mfdataset(f'{utils.data_output_path}/greens_functions/Greens_function_{poll_name}*', combine = 'nested', concat_dim = 'run')
ds['BC_total'].attrs = {'units':'ug/m3 per g/m2/day'}
ds.to_netcdf(f'{utils.data_output_path}/greens_functions/GF_combined.nc')