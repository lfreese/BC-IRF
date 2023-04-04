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
    """Takes the globally weighted sum of a dataset and its variable, as long as the dataset has a dz and area"""
    return (ds[variable].weighted(ds['area']*ds['dz']).sum(dim = ['lat','lon','lev']))

def global_w_sum_area(ds, variable):
    """Takes the globally weighted sum of a dataset and its variable, as long as the dataset has a dz and area"""
    return (ds[variable].weighted(ds['area']).sum(dim = ['lat','lon']))

    
def global_w_mean(ds, variable):
    """Takes the globally weighted mean of a dataset and its variable, as long as the dataset has a dz and area"""
    return (ds[variable].weighted(ds['area']*ds['dz']).mean(dim = ['lat','lon','lev']))

def global_sfc_w_mean(ds, variable):
    return (ds[variable].isel(lev = 0).weighted(ds['area']*ds['dz'].isel(lev = 0)).mean(dim = ['lat','lon']))

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

#### forcing to divide out ####

def f_(raw_f, t_p): 
    """Select our forcing at time tp"""
    return raw_f.interp({'time':t_p})



######## height #####
## from http://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids
height = pd.read_excel('GC_lev72.xlsx')
height_ds = (height.loc[height['L'].isna()].diff().set_index(np.arange(73,0,-1))['Altitude'].dropna()[::-1].to_xarray().rename({'index':'lev'})*-1e3).sortby('lev')


pressure = pd.read_excel('GC_lev72.xlsx')
pressure_ds = (pressure.loc[~pressure['L'].isna()].set_index(np.arange(72,0,-1))['Pressure']*100).to_xarray().rename({'index':'lev'}).sortby('lev')










########## Convolution #########

# #### global mean convolution ####
# #def convolve_global_mean(E, G_mean, E_len, G_len):
#   #  '''convolves a global mean G with an emissions scenario of any length'''
#   #  C = np.ndarray(((E_len+G_len+1)))
#   #  for i, tp in enumerate(np.arange(0,E_len -1)):
#    #     C[i:i+G_len] = C[i:i+G_len]+ G_mean*E[i]
#    #     C = np.nan_to_num(C, nan=0.0)
#    # return C


# def convolve_global_mean(G, E, dt, time_dim):
#     '''convolves a global mean G with an emissions scenario of any length'''
    
#     E_len = len(E)
#     #print(E_len)
#     G_len = len(G[time_dim])
#     C = np.zeros((E_len+G_len)) 
#     for i, tp in enumerate(np.arange(0,E_len)):
#         #print(E[i].values)
#         C[i:i+G_len] += G*E[i]*dt #C.loc slice or where
#         #plt.plot(C)
#     C = xr.DataArray(
#     data = C,
#     dims = ['s'],
#     coords = dict(
#         s = (['s'], np.arange(0,(E_len+G_len)))
#             )
#         )
#     return C

# #### single lev convolution ####
# def convolve_single_lev_testing(G, E, E_len, G_len): ##testing to see if we can get it with dask and xarary
#     '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
#     #C = np.zeros(((E_len+G_len), len(G.lat), len(G.lon))) #xarray the shape of tthe output I want dask.zeros as the input to xarray
#     C = xr.DataArray(
#     data = dask.array.zeros(((E_len+G_len), len(G.lat), len(G.lon))),
#     dims = ['s','lat','lon'],
#     coords = dict(
#         s = (['s'], np.arange(0,(E_len+G_len))),
#         lat = (['lat'], G.lat.values),
#         lon = (['lon'], G.lon.values)
#             )
#         )

#     for i, tp in enumerate(np.arange(0,E_len)):
#         #C[i:i+G_len] += G*E[i] #C.loc slice or where
#         C.loc[dict(s = slice(i,i + G_len))] = C.loc[dict(s = slice(i,i + G_len))]+ G*E[i]
#     return C

# def convolve_single_lev(G, E, dt):
#     '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
#     E_len = len(E)
#     G_len = len(G.s)
#     C = np.zeros(((E_len+G_len), len(G.lat), len(G.lon))) 
#     for i, tp in enumerate(np.arange(0,E_len)):
#         C[i:i+G_len] += G*E[i]*dt #C.loc slice or where
#     C = xr.DataArray(
#     data = C,
#     dims = ['s','lat','lon'],
#     coords = dict(
#         s = (['s'], np.arange(0,(E_len+G_len))),
#         lat = (['lat'], G.lat.values),
#         lon = (['lon'], G.lon.values)
#             )
#         )
#     return C

# def convolve_applied_single_lev(G, E):
#     '''applies the single lev convolution'''
#     E_len = len(E)
#     G_len = len(G.s)
#     C = np.zeros(((E_len+G_len), len(G.lat), len(G.lon)))
#     test = xr.apply_ufunc(
#         convolve_single_lev,
#         C,
#         G,
#         input_core_dims = [[],['s','lat','lon']],
#         exclude_dims = set('s'),
#         output_core_dims = [['s','lat','lon']],
#         #vectorize=True,
#         kwargs={"E": E, "E_len":E_len, "G_len":G_len},
#         )
#     test = test.assign_coords(s = np.arange(0,E_len + G_len))
#     return(test)

# #### fully resolved convolution ####
# #slowest method
# def convolve_resolved(C, G, E, E_len, G_len):
#     '''convolves a spatially resolved G with an emissions scenario of any length. This method is significantly slower depending on the number of levels included'''
#     for i, tp in enumerate(np.arange(0,E_len)):
#         C[i:i+G_len] = C[i:i+G_len]+ G*E[i]
#         C = np.nan_to_num(C, nan=0.0)
#     return C

# def convolve_applied_resolved(G, E):
#     '''applies the fully resolved convolution'''
#     E_len = len(E)
#     G_len = len(G.s)
#     C = np.ndarray(((E_len+G_len), len(G.lev), len(G.lat), len(G.lon)))
#     test = xr.apply_ufunc(
#         convolve,
#         C,
#         G,
#         input_core_dims = [[],['s','lev','lat','lon']],
#         exclude_dims = set('s'),
#         output_core_dims = [['s','lev','lat','lon']],
#         #vectorize=True,
#         kwargs={"E": E, "E_len":E_len, "G_len":G_len},
#         )
#     test = test.assign_coords(s = np.arange(0,E_len + G_len))
#     return(test)

######################## Shutdown scenarios #############################

# def shutdown_by_country(shutdown_years, time_array, country, E, df):
#     '''Shutdown a plant early based on the lender, all other plants stay on for entire time of simulation
#     inputs: shutdown_years (in years), time (array the length of simulation), country (country to shutdown early), E (dictionary for emissions),
#     df (with column named 'Year_of_Commission' and 'BC_(g/day)')'''
#     #create array
#     E[f'{country}_{shutdown_years}yrs'] = np.zeros(len(time_array))
#     #early shutdowns
#     shutdown_days = shutdown_years*365 #day
#     E[f'{country}_{shutdown_years}yrs'] += df.loc[df.Country == country]['BC_(g/day)'].sum()*(time_array<shutdown_days)
#     #rest of plants
#     shutdown_days = 0
#     E[f'{country}_{shutdown_years}yrs'] += df.loc[df.Country != country]['BC_(g/day)'].sum()*(time_array>=shutdown_days)
    
# def shutdown_by_lender(shutdown_years, time_array, lender, E, df):
#     '''Shutdown a plant early based on the lender, all other plants stay on for entire time of simulation
#     inputs: shutdown_years (in years), time (array the length of simulation), lender (lender to shutdown early), E (dictionary for emissions),
#     df (with column named 'Year_of_Commission' and 'BC_(g/day)')'''
#     #create array
#     E[f'{lender}_{shutdown_years}yrs'] = np.zeros(len(time_array))
#     #early shutdowns
#     shutdown_days = shutdown_years*365 #day
#     E[f'{lender}_{shutdown_years}yrs'] += df.loc[df.Lender == lender]['BC_(g/day)'].sum()*(time_array<shutdown_days)
#     #rest of plants
#     shutdown_days = 0
#     E[f'{lender}_{shutdown_years}yrs'] += df.loc[df.Lender != lender]['BC_(g/day)'].sum()*(time_array>=shutdown_days)
    
# def shutdown_early_commission_year(shutdown_years, time_array, year_of_commis, E, df):
#     '''Shutdown a plant early if comissioned before a certain year, all other plants stay on for entire time of simulation
#     inputs: shutdown_years (in years), time (array the length of simulation), year_of_commis (year of commission to shutdown early), E (dictionary for emissions),
#     df (with column named 'Year_of_Commission' and 'BC_(g/day)')'''
#     #######need to fix to not allow to start before built########
#     #create array
#     E[f'{year_of_commis}_{shutdown_years}yrs'] = np.zeros(len(time_array))
#     #early shutdowns
#     shutdown_days = shutdown_years*365 #day
#     E[f'{year_of_commis}_{shutdown_years}yrs'] += df.loc[df.Year_of_Commission < year_of_commis]['BC_(g/day)'].sum()*(time_array<shutdown_days)
#     #rest of plants
#     shutdown_days = 0
#     E[f'{year_of_commis}_{shutdown_years}yrs'] += df.loc[df.Year_of_Commission >= year_of_commis]['BC_(g/day)'].sum()*(time_array>=shutdown_days)