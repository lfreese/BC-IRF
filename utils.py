import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import dask

################ General functions ################
def combine_BC(ds):
    """Combines BCPI and BCPO and converts it to emissions in kg/m2/day"""
    sec_day = 86400
    ds['EmisBC_Total'] = (ds['EmisBCPI_Total'] + ds['EmisBCPO_Total'])
    ds['EmisBC_Total'].attrs = {'full_name':'total black carbon','units':'kg/m2/day'}
    ds['EmisBC_Total'] *=sec_day
    
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

#### Derivative for the pulse vs. base ####
#doing a sustained pulse, not single day pulse
def calc_δc_δt_mean(ds, conc_species):
    ds['conc_dif'] = (global_w_mean(ds.sel(run = 'delta').fillna(0), conc_species)- 
                            global_w_mean(ds.sel(run = 'base').fillna(0), conc_species))
    δc_δt = ds['conc_dif'].diff('time').fillna(0)
    return(δc_δt)

def calc_δc_δt(ds, conc_species):
    ds['conc_dif'] = (ds.sel(run = 'delta')[conc_species]- 
                            ds.sel(run = 'base')[conc_species])
    δc_δt = ds['conc_dif'].diff('time')
    return(δc_δt)

#### forcing to divide out ####

def f_(raw_f, t_p): 
    """Select our forcing at time tp"""
    return raw_f.interp({'time':t_p})

########## Convolution #########

#### global mean convolution ####
#def convolve_global_mean(E, G_mean, E_len, G_len):
  #  '''convolves a global mean G with an emissions scenario of any length'''
  #  C = np.ndarray(((E_len+G_len+1)))
  #  for i, tp in enumerate(np.arange(0,E_len -1)):
   #     C[i:i+G_len] = C[i:i+G_len]+ G_mean*E[i]
   #     C = np.nan_to_num(C, nan=0.0)
   # return C


def convolve_global_mean(G, E):
    '''convolves a global mean G with an emissions scenario of any length'''
    E_len = len(E)
    G_len = len(G.s)
    C = np.zeros((E_len+G_len)) 
    for i, tp in enumerate(np.arange(0,E_len)):
        C[i+1:i+G_len+1] += G*E[i] #C.loc slice or where
    C = xr.DataArray(
    data = C,
    dims = ['s'],
    coords = dict(
        s = (['s'], np.arange(0,(E_len+G_len)))
            )
        )
    return C

#### single lev convolution ####
def convolve_single_lev_testing(G, E, E_len, G_len):
    '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
    #C = np.zeros(((E_len+G_len), len(G.lat), len(G.lon))) #xarray the shape of tthe output I want dask.zeros as the input to xarray
    C = xr.DataArray(
    data = dask.array.zeros(((E_len+G_len), len(G.lat), len(G.lon))),
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )

    for i, tp in enumerate(np.arange(0,E_len)):
        #C[i:i+G_len] += G*E[i] #C.loc slice or where
        C.loc[dict(s = slice(i,i + G_len))] = C.loc[dict(s = slice(i,i + G_len))]+ G*E[i]
    return C

def convolve_single_lev(G, E):
    '''convolves a spatially resolved G that is mean or single level with an emissions scenario of any length'''
    E_len = len(E)
    G_len = len(G.s)
    C = np.zeros(((E_len+G_len), len(G.lat), len(G.lon))) 
    for i, tp in enumerate(np.arange(0,E_len)):
        C[i:i+G_len] += G*E[i] #C.loc slice or where
    C = xr.DataArray(
    data = C,
    dims = ['s','lat','lon'],
    coords = dict(
        s = (['s'], np.arange(0,(E_len+G_len))),
        lat = (['lat'], G.lat.values),
        lon = (['lon'], G.lon.values)
            )
        )
    return C

def convolve_applied_single_lev(G, E):
    '''applies the single lev convolution'''
    E_len = len(E)
    G_len = len(G.s)
    C = np.zeros(((E_len+G_len), len(G.lat), len(G.lon)))
    test = xr.apply_ufunc(
        convolve_single_lev,
        C,
        G,
        input_core_dims = [[],['s','lat','lon']],
        exclude_dims = set('s'),
        output_core_dims = [['s','lat','lon']],
        #vectorize=True,
        kwargs={"E": E, "E_len":E_len, "G_len":G_len},
        )
    test = test.assign_coords(s = np.arange(0,E_len + G_len))
    return(test)

#### fully resolved convolution ####
#slowest method
def convolve_resolved(C, G, E, E_len, G_len):
    '''convolves a spatially resolved G with an emissions scenario of any length. This method is significantly slower depending on the number of levels included'''
    for i, tp in enumerate(np.arange(0,E_len)):
        C[i:i+G_len] = C[i:i+G_len]+ G*E[i]
        C = np.nan_to_num(C, nan=0.0)
    return C

def convolve_applied_resolved(G, E):
    '''applies the fully resolved convolution'''
    E_len = len(E)
    G_len = len(G.s)
    C = np.ndarray(((E_len+G_len), len(G.lev), len(G.lat), len(G.lon)))
    test = xr.apply_ufunc(
        convolve,
        C,
        G,
        input_core_dims = [[],['s','lev','lat','lon']],
        exclude_dims = set('s'),
        output_core_dims = [['s','lev','lat','lon']],
        #vectorize=True,
        kwargs={"E": E, "E_len":E_len, "G_len":G_len},
        )
    test = test.assign_coords(s = np.arange(0,E_len + G_len))
    return(test)

######################## Shutdown scenarios #############################

def shutdown_by_country(shutdown_years, time_array, country, E, df):
    '''Shutdown a plant early based on the lender, all other plants stay on for entire time of simulation
    inputs: shutdown_years (in years), time (array the length of simulation), country (country to shutdown early), E (dictionary for emissions),
    df (with column named 'Year_of_Commission' and 'BC_(g/day)')'''
    #create array
    E[f'{country}_{shutdown_years}yrs'] = np.zeros(len(time_array))
    #early shutdowns
    shutdown_days = shutdown_years*365 #day
    E[f'{country}_{shutdown_years}yrs'] += df.loc[df.Country == country]['BC_(g/day)'].sum()*(time_array<shutdown_days)
    #rest of plants
    shutdown_days = 0
    E[f'{country}_{shutdown_years}yrs'] += df.loc[df.Country != country]['BC_(g/day)'].sum()*(time_array>=shutdown_days)
    
def shutdown_by_lender(shutdown_years, time_array, lender, E, df):
    '''Shutdown a plant early based on the lender, all other plants stay on for entire time of simulation
    inputs: shutdown_years (in years), time (array the length of simulation), lender (lender to shutdown early), E (dictionary for emissions),
    df (with column named 'Year_of_Commission' and 'BC_(g/day)')'''
    #create array
    E[f'{lender}_{shutdown_years}yrs'] = np.zeros(len(time_array))
    #early shutdowns
    shutdown_days = shutdown_years*365 #day
    E[f'{lender}_{shutdown_years}yrs'] += df.loc[df.Lender == lender]['BC_(g/day)'].sum()*(time_array<shutdown_days)
    #rest of plants
    shutdown_days = 0
    E[f'{lender}_{shutdown_years}yrs'] += df.loc[df.Lender != lender]['BC_(g/day)'].sum()*(time_array>=shutdown_days)
    
def shutdown_early_commission_year(shutdown_years, time_array, year_of_commis, E, df):
    '''Shutdown a plant early if comissioned before a certain year, all other plants stay on for entire time of simulation
    inputs: shutdown_years (in years), time (array the length of simulation), year_of_commis (year of commission to shutdown early), E (dictionary for emissions),
    df (with column named 'Year_of_Commission' and 'BC_(g/day)')'''
    #######need to fix to not allow to start before built########
    #create array
    E[f'{year_of_commis}_{shutdown_years}yrs'] = np.zeros(len(time_array))
    #early shutdowns
    shutdown_days = shutdown_years*365 #day
    E[f'{year_of_commis}_{shutdown_years}yrs'] += df.loc[df.Year_of_Commission < year_of_commis]['BC_(g/day)'].sum()*(time_array<shutdown_days)
    #rest of plants
    shutdown_days = 0
    E[f'{year_of_commis}_{shutdown_years}yrs'] += df.loc[df.Year_of_Commission >= year_of_commis]['BC_(g/day)'].sum()*(time_array>=shutdown_days)