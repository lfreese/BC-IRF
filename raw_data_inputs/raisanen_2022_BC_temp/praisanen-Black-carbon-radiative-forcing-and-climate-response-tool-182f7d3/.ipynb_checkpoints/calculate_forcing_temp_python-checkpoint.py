import xarray as xr
import numpy as np
import pandas as pd
import pdb

nlon = 12
nlat = 16
nmonth = 12
nrecl = 1
UNDEF = -9999.0
pi = np.pi
rearth = 6.37122e6 
area_earth = 4*pi*rearth**2


# Read NetCDF data
def read_netcdf_data(filename):
    return xr.open_dataset(filename)

# Read input data
def read_input_data(filename):
    df = pd.read_csv(filename, index_col = [0])

    emissions = df.loc['Emissions_rate']
    
    if len(emissions) == 1:
        lmonthly_emissions = False
        emissions_annual = emissions[0]
        emissions_monthly = None
    elif len(emissions) == 12:
        lmonthly_emissions = True
        emissions_monthly = xr.DataArray(emissions, dims=['month'], coords={'month': range(1, 13)})
        emissions_annual = None
    else:
        raise ValueError("Invalid number of emission values")
        
    return df.loc['Longitude'][0], df.loc['Longitude'][1], df.loc['Latitude'][0], df.loc['Latitude'][1], lmonthly_emissions, emissions_monthly, emissions_annual

# Calculate overlap area
def calc_overlap_area(lon_w, lon_e, lat_s, lat_n, ds):
    sin_s = np.sin(np.radians(lat_s))
    sin_n = np.sin(np.radians(lat_n))
    sin_south = np.sin(np.radians(ds.latitude_south))
    sin_north = np.sin(np.radians(ds.latitude_north))
    
    def common_lon(xlon1, xlon2, ylon1, ylon2):
        xlon1a, xlon2a, xlon1b, xlon2b = split_lon(xlon1, xlon2)
        breakpoint()
        ylon1a, ylon2a, ylon1b, ylon2b = split_lon(ylon1, ylon2)
        
        delta1 = max(min(xlon2a, ylon2a) - max(xlon1a, ylon1a), 0)
        delta2 = max(min(xlon2a, ylon2b) - max(xlon1a, ylon1b), 0)
        delta3 = max(min(xlon2b, ylon2a) - max(xlon1b, ylon1a), 0)
        delta4 = max(min(xlon2b, ylon2b) - max(xlon1b, ylon1b), 0)
        
        return delta1 + delta2 + delta3 + delta4
    
    delta_lon = xr.apply_ufunc(common_lon, lon_w, lon_e, ds.longitude_west, ds.longitude_east)
    delta_sinlat = xr.ufuncs.maximum(xr.ufuncs.minimum(sin_n, sin_north) - xr.ufuncs.maximum(sin_s, sin_south), 0)
    overlap_area = (delta_lon/360) * (delta_sinlat/2) * AREA_EARTH
    return overlap_area

# Calculate global annual mean radiative forcing
def global_annual_mean_rf(rf, emissions, overlap_area, sec_per_season, sec_per_year):
    summa = (overlap_area * emissions * rf * sec_per_season).sum()
    return 1e12 * summa / (AREA_EARTH * sec_per_year)

# Calculate global annual mean temperature response  
def global_annual_mean_dt(dt, emissions, overlap_area, sec_per_season, sec_per_year):
    summa = (overlap_area * emissions * dt * sec_per_season).sum()
    return summa / sec_per_year

# Main calculation
def calculate_forcing_and_response():
    # Read data
    nc_file = "BC_forcing_and_climate_response_normalized_by_emissions.nc"
    ds = read_netcdf_data(nc_file)
    breakpoint()
    lon_w, lon_e, lat_s, lat_n, lmonthly_emissions, emissions_monthly, emissions_annual = read_input_data("input_data.csv")
    
    # Calculate seasonal emissions
    ndays_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    seasons = ['ANN', 'DJF', 'MAM', 'JJA', 'SON']
    sec_per_month = xr.DataArray(np.array(ndays_per_month) * 86400, dims=['month'], coords={'month': range(1, 13)})
    months_in_season = xr.DataArray([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,0,0,1],
        [0,0,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,0]
    ], dims=['season', 'month'], coords={'season': seasons, 'month': range(1, 13)})
    sec_per_season = (months_in_season * sec_per_month).sum('month')
    sec_per_year = sec_per_season.sel(season = 'ANN')
    
    if lmonthly_emissions:
        emissions_seasonal = (months_in_season * sec_per_month * emissions_monthly).sum(dim = 'month') / sec_per_season
        
    else:
        emissions_seasonal = xr.DataArray(np.zeros(len(seasons)), dims=['season'], coords={'season': seasons})
        emissions_seasonal.loc[emissions_seasonal['season'] == 'ANN'] = emissions_annual
    
    # Calculate overlap area
    overlap_area = calc_overlap_area(lon_w, lon_e, lat_s, lat_n, ds)
    
    # Calculate forcings and responses
    drf_toa_out = global_annual_mean_rf(ds.dirsf_toa, emissions_seasonal, overlap_area, sec_per_season, sec_per_year)
    snowrf_toa_out = global_annual_mean_rf(ds.snowsf_toa, emissions_seasonal, overlap_area, sec_per_season, sec_per_year)
    dt_drf_out = global_annual_mean_dt(ds.dt_norm_drf, emissions_seasonal, overlap_area, sec_per_season, sec_per_year)
    dt_snowrf_out = global_annual_mean_dt(ds.dt_norm_snowrf, emissions_seasonal, overlap_area, sec_per_season, sec_per_year)
    
    # Print results
    print("\nEstimated global annual-mean radiative forcings [W m-2]")
    print(f"DRF_TOA    {drf_toa_out.values:.5e}")
    print(f"SNOWRF_TOA {snowrf_toa_out.values:.5e}")
    print(f"SUM_RF_TOA {(drf_toa_out + snowrf_toa_out).values:.5e}")
    
    print("\nEstimated global annual-mean temperature responses [K]")
    print(f"DT_DRF     {dt_drf_out.values:.5e}")
    print(f"DT_SNOWRF  {dt_snowrf_out.values:.5e}")  
    print(f"DT_SUM     {(dt_drf_out + dt_snowrf_out).values:.5e}")

# Helper function
def split_lon(rlon1, rlon2):
    if rlon1 < 0:
        rlon1a, rlon2a = 0, max(rlon2, 0)
        rlon1b, rlon2b = rlon1 + 360, min(rlon2, 0) + 360
    else:
        rlon1a, rlon2a = rlon1, rlon2
        rlon1b, rlon2b = 0, 0
    return rlon1a, rlon2a, rlon1b, rlon2b

if __name__ == "__main__":
    calculate_forcing_and_response()