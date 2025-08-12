import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import dask
import pandas as pd

################ Dicts #############
names_dict = {'SEA':'Southeast Asia', 'Indo':'Indonesia', 'Malay':'Malaysia', 'Viet':'Vietnam', 'Cambod':'Cambodia', 
              'all_countries':'All Countries, Stepped', 
              'SEA_2x':'Southeast Asia Doubled', 'Indo_2x':'Indonesia Doubled', 
              'Malay_2x':'Malaysia Doubled', 'Viet_2x':'Vietnam Doubled', 'Cambod_2x':'Cambodia Doubled', 
              'all_countries_2x':'All Countries, Stepped Doubled'}


################## Data paths #############
GF_name_path = '/net/fs11/d0/emfreese/BC-IRF/data_output/greens_functions/'
data_output_path = '/net/fs11/d0/emfreese/BC-IRF/data_output/'
raw_data_in_path = '/net/fs11/d0/emfreese/BC-IRF/raw_data_inputs/'
data_prep_path = '/net/fs11/d0/emfreese/BC-IRF/data_prep/'
geos_chem_data_path = '/net/fs11/d0/emfreese/GCrundirs/IRF_runs/'
figures_data_path = '/net/fs11/d0/emfreese/BC-IRF/figures/'



################## Constants #################
years = 60

################ General functions ################
def combine_BC(ds):
    """Combines BCPI and BCPO and converts it to emissions in g/m2/day"""
    sec_day = 86400
    g_per_kg = 1000 #g/kg
    ds['EmisBC_Total'] = (ds['EmisBCPI_Total'] + ds['EmisBCPO_Total'])
    ds['EmisBC_Total'] *=sec_day
    ds['EmisBC_Total']*=g_per_kg
    ds['EmisBC_Total'].attrs = {'full_name':'total black carbon','units':'g/m2/day'}
    
def global_w_sum_vol(ds, variable):
    """Takes the globally weighted sum of a dataset and its variable, as long as the dataset has a Altitude and area"""
    return (ds[variable].weighted(ds['area']*ds['Altitude']).sum(dim = ['lat','lon','lev']))

def global_w_sum_area(ds, variable):
    """Takes the globally weighted sum of a dataset and its variable, as long as the dataset has a Altitude and area"""
    return (ds[variable].weighted(ds['area']).sum(dim = ['lat','lon']))

    
def global_w_mean(ds, variable):
    """Takes the globally weighted mean of a dataset and its variable, as long as the dataset has a Altitude and area"""
    return (ds[variable].weighted(ds['area']*ds['Altitude']).mean(dim = ['lat','lon','lev']))

def global_sfc_w_mean(ds, variable):
    return (ds[variable].isel(lev = 0).weighted(ds['area']).mean(dim = ['lat','lon']))

def fix_area_ij_latlon(ds):
    ds_area = ds['area']
    ds_area = ds_area.rename({'i':'lat','j':'lon'})
    ds_area['i']= ds['lat']
    ds_area['j'] = ds['lon']
    ds['area'] = ds_area.drop(['j','i'])
    return(ds)

#### function to find area of a grid cell from lat/lon ####
def find_area(ds, R = 6378.1):
    """ ds is the dataset, i is the number of longitudes to assess, j is the number of latitudes, and R is the radius of the earth in km. 
    Returns Area of Grid cell in km"""
    
    ds['lat_b'] = ds['lat_b'].sortby('lat_b')
    ds['lon_b'] = ds['lon_b'].sortby('lon_b', )
    ds['lat'] = ds['lat'].sortby('lat')
    ds['lon'] = ds['lon'].sortby('lon')
    
    circumference = (2*np.pi)*R
    deg_to_m = (circumference/360) 
    dy = (ds['lat_b'].roll({'lat_b':-1}, roll_coords = False) - ds['lat_b'])[:-1]*deg_to_m

    dx1 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b']))
    
    dx2 = (ds['lon_b'].roll({'lon_b':-1}, roll_coords = False) - 
           ds['lon_b'])[:-1]*deg_to_m*np.cos(np.deg2rad(ds['lat_b'].roll({'lat_b':-1}, roll_coords = False)[:-1]))
    
    A = .5*(dx1+dx2)*dy
    
    #### assign new lat and lon coords based on the center of the grid box instead of edges ####
    A = A.assign_coords(lon_b = ds.lon.values,
                    lat_b = ds.lat.values)
    A = A.rename({'lon_b':'lon','lat_b':'lat'})

    A = A.transpose()
    
    return(A)


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




def np_to_xr_mean(C, G, E):
    E_len = len(E)
    G_len = len(G.s)
    C = xr.DataArray(
    data = C,
    dims = ['s'],
    coords = dict(
        s = (['s'], np.arange(0, C.shape[0])), #np.arange(0,(E_len+G_len))),
            )
        )
    return(C)


############## Green's function calculation ################

def G_(GC_out, t, t_p, f_0, Δt):
    """Calculate the green's function"""
    'where G(t-t_p) such that t-t_p = s), if s<0, the function goes to 0'
    s = t-t_p
    if t>t_p:
        G = GC_out.interp({'time':s})/(Δt*f_0) #divide by our original f0 and our dt in order to get into units of concentration
    elif t_p>=t:
        G = xr.zeros_like(GC_out.interp({'time':s})) #np.nan
        
    return(G)

def G_f_kernel(raw_G, raw_f, t, t_p, Δt, dt, f_0, ds_output = True):
    """Convolve our green's function and our forcing"""
    'where C(t) = int[G(t-t_p)f(t_p)]dt and t_p is the midpoint of the integral (eg: if integrating from 2020-2030, it is dt/2 away from 2020)'
    if ds_output == True:
        return(G_(raw_G, t, t_p, f_0, Δt)*f_(raw_f, t_p)*dt)
    else:
        return(G_(raw_G, t, t_p, f_0, Δt).values*f_(raw_f, t_p).values*dt)

    
def switch_conc_time(ds):
    ds['time'] = ds['time'] + np.timedelta64(12,'h')
    
    
#### Derivative for the pulse vs. base ####
#doing a sustained pulse, not single day pulse
def calc_δc_δt_mean(ds, conc_species, run_delta_name, run_base_name):
    '''take the backwards time difference for the concentration, adding in a 0th timestep with delta_c = 0 '''
    ds['conc_dif'] = (global_w_mean(ds.sel(run = run_delta_name).fillna(0), conc_species)- 
                            global_w_mean(ds.sel(run = run_base_name).fillna(0), conc_species))
    time = pd.date_range((ds['time'][0]- np.timedelta64(24,'h')).values, freq='H', periods=1)
    ds_0 = xr.Dataset({'conc_dif': ('time', [0]), 'time': time})
    ds = xr.concat([ds_0,ds], dim = 'time')
    
    δc_δt = ds['conc_dif'].diff('time').fillna(0)/(ds['time'].diff('time')/(24*60*60*1e9)).astype('float64')
    return(δc_δt)

def calc_δc_δt(ds, conc_species, run_delta_name, run_base_name):
    ds['conc_dif'] = (ds.sel(run = run_delta_name)[conc_species]- 
                            ds.sel(run = run_base_name)[conc_species])
    time = pd.date_range((ds['time'][0]- np.timedelta64(24,'h')).values, freq='H', periods=1)
    ds_0 = xr.Dataset(data_vars = dict(conc_dif = (['time','lat','lon'], np.zeros([1,len(ds['lat']),len(ds['lon'])]))), coords = dict(
            time = time, 
            lat = ds['lat'], lon = ds['lon']))

    ds = xr.concat([ds_0,ds], dim = 'time')
    δc_δt = ds['conc_dif'].diff('time').fillna(0)/(ds['time'].diff('time')/(24*60*60*1e9)).astype('float64')

    return(δc_δt)

def calc_δc_δt_lev0_mean(ds, conc_species, run_delta_name, run_base_name):
    '''take the backwards time difference for the concentration, adding in a 0th timestep with delta_c = 0 '''
    ds['conc_dif'] = (global_sfc_w_mean(ds.sel(run = run_delta_name).fillna(0), conc_species)- 
                            global_sfc_w_mean(ds.sel(run = run_base_name).fillna(0), conc_species))
    time = pd.date_range((ds['time'][0]- np.timedelta64(24,'h')).values, freq='H', periods=1)
    ds_0 = xr.Dataset({'conc_dif': ('time', [0]), 'time': time})
    ds = xr.concat([ds_0,ds], dim = 'time')
    
    δc_δt = ds['conc_dif'].diff('time').fillna(0)/(ds['time'].diff('time')/(24*60*60*1e9)).astype('float64')
    return(δc_δt)

#### forcing to divide out ####

def f_(raw_f, t_p): 
    """Select our forcing at time tp"""
    return raw_f.interp({'time':t_p})


######## weighted average ########

def grouped_weighted_avg(values, weights):
    return (values * weights).sum() / weights.sum()
# ######## height #####
## from http://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids

height = pd.read_excel(f'{utils.raw_data_in_path}gc_72_estimate.xlsx', index_col = 0)
height = height.reindex(index=height.index[::-1])
height_ds = height.diff().dropna().to_xarray().rename({'L':'lev'})
height_ds = height_ds.rename({'Altitude (km)':'dz'}) 
height_ds['dz']*=1e3 #convert to meters
height_ds['dz'].attrs = {'units':'m'}


# pressure = pd.read_excel('/net/fs11/d0/emfreese/BC-IRF/GC_lev72.xlsx')
# pressure_ds = (pressure.loc[~pressure['L'].isna()].set_index(np.arange(72,0,-1))['Pressure']*100).to_xarray().rename({'index':'lev'}).sortby('lev')

pressure = pd.read_excel(f'{utils.raw_data_in_path}gc_72_estimate.xlsx', index_col = 0)
pressure = pressure.reindex(index=pressure.index[::-1])*-1
pressure_ds = pressure.diff().dropna().to_xarray().rename({'L':'lev'})*100 # convert to Pa
pressure_ds = pressure_ds.rename({'Pressure (hPa)':'dP'}) 
pressure_ds['dP'].attrs = {'units':'Pa'}

###### Conversions and extensions #######

def ppb_to_ug(ds, species_to_convert, mw_species_list, P, T):
    '''Convert species to ug/m3 from ppb'''
    R = 8.314 #J/K/mol
    mol_per_m3= (P / (T * R)) #Pa/K/(J/K/mol) = mol/m3
    
    for spec in species_to_convert:
        attrs = ds[spec].attrs
        ds[spec] = ds[spec]*mw_species_list[spec]*mol_per_m3*1e-3 #ppb*g/mol*mol/m3*ug/ng
        ds[spec].attrs['units'] = 'μg m-3'


def exponential_decay(a, b, N):
    return a * (1-b) ** np.arange(N)

####### Plant Shutdowns #########

####### Functions #########

def individual_plant_shutdown(years_running, df, time_array, typical_shutdown_years, unique_id, min_year, var = 'BC_(g/day)'):
    ''' Shutdown a unit early. The df must have a variable 'Year_of_Commission' describing when the plant was comissioned, and 'BC_(g/day)' for BC emissions in g/day
        years_running is the number of years the plant runs
        time_array is the length of time for our simulation
        shutdown_years is the typical lifetime of a coal plant
        unique_id is the unique identifier of a unit'''
    shutdown_days = typical_shutdown_years*365
    E = np.zeros(len(time_array))
    ID_df = df.loc[df['unique_ID'] == unique_id]
    yr_offset = (ID_df['Year_of_Commission'].iloc[0] - min_year)
    test_array = np.where((time_array <= (yr_offset + years_running)*365) & (time_array >= yr_offset * 365), True, False)
    E += test_array* ID_df[var].sum()
    return(E)

## function for creating a time specific xarray data array

def np_to_xr_time_specific(C, G, E, time_init):
    '''Function to create an xarray data-array over a specified time period, with lat, lon, and s dimensions (s = time)'''
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(time_init, C.shape[0] + time_init)), 
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return(C)

