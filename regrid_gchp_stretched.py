#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=01:10:00
#SBATCH --mem=MaxMemPerNode
#SBATCH --cpus-per-task=2
#SBATCH --partition=edr

import os

import gcpy.constants as gcon
import sparselt.esmf
import sparselt.xr
import xarray as xr
from dask.diagnostics import ProgressBar
import gcpy
import numpy as np
import argparse

skip_vars = gcon.skip_these_vars


############# Inputs #################
#typical inputs for things are: cube_res = 90, lat = 180, lon = 288, 

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--cube_res',type=int, required=True)
parser.add_argument('--lat',type=int, required=True)
parser.add_argument('--lon',type=int, required=True)
parser.add_argument('--month',type=int, required=True)
parser.add_argument('--model_run', type=str, required=True)
parser.add_argument('--location', type=str, required=True)
parser.add_argument('--month_step', type=str, required=True)

# Parse the argument
args = parser.parse_args()


#set variables
cube_res = args.cube_res
lat = args.lat
lon = args.lon
month = args.month
model_run = args.model_run
location = args.location
month_step = args.month_step

print(location)
#create paths to data
regrid_path = "/net/fs11/d0/emfreese/GCrundirs/IRF_runs/regrid_files/"
weights_file = f"esmf_regrid_weights_c{cube_res}_s2_11_112_to_latlon{lat}x{lon}.nc"#f"esmf_regrid_weights_c{cube_res}_to_latlon{lat}x{lon}.nc"
reg_latlon = f"regular_lat_lon_{lat}x{lon}.nc"


#select date ranges
if month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
    dates = np.arange(1,6)

elif month == 2:
    dates = np.arange(1, 28)
    
else:
    dates = np.arange(1,31)
print(dates)

filenames = [
    f"GEOSChem.SpeciesConc.2016{str(month).zfill(2)}{str(date).zfill(2)}_0000z.nc4" for date in dates
] + [
    f"GEOSChem.AerosolMass.2016{str(month).zfill(2)}{str(date).zfill(2)}_0000z.nc4" for date in dates
]+ [
    f"GEOSChem.Emissions.2016{str(month).zfill(2)}{str(date).zfill(2)}_0000z.nc4" for date in dates
]


input_path = f'/net/fs11/d0/emfreese/GCrundirs/IRF_runs/{model_run}/{location}/{month_step}/OutputDir/' #"/path/to/your/cubesphere/input/files/"
destination_path = f'/net/fs11/d0/emfreese/GCrundirs/IRF_runs/{model_run}/{location}/{month_step}/mod_output/' #"/path/to/your/latlon/output/files/"
from pathlib import Path
Path(destination_path).mkdir(parents=True, exist_ok=True) 
#os.makedirs(destination_path, exist_ok=True) 

############ Define regridding function ############


def main():
    for filename in filenames:
                    
        print("Opening cube-sphere data")
        ds = xr.open_dataset(
            input_path + filename, engine="netcdf4", drop_variables=skip_vars
        )
        ##lyssa added
        ds = ds.drop(labels = ['corner_lons','corner_lats'])
        ##finished lyssa added
        transform = sparselt.esmf.load_weights(
            regrid_path + weights_file,
            input_dims=[("nf", "Ydim", "Xdim"), (6, cube_res, cube_res)],
            output_dims=[("lat", "lon"), (lat, lon)],
        )

        output_template = xr.open_dataset(regrid_path + reg_latlon)

        print("Transforming data")
        with ProgressBar():
            ds_new = sparselt.xr.apply(transform, ds, output_template)
        
        print("Saving to NetCDF")
        output_file = os.path.join(destination_path, filename)
        with ProgressBar():
             ds_new.load().to_netcdf(output_file)


############ Run the function ############


if __name__ == "__main__":
    main()


# In[ ]:




