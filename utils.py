import matplotlib.pyplot as plt
import xarray as xr
import numpy as np



def combine_BC(ds):
    """Combines BCPI and BCPO and converts it to emissions in seconds per day"""
    sec_day = 86400
    ds['EmisBC_Total'] = (ds['EmisBCPI_Total'] + ds['EmisBCPO_Total'])
    ds['EmisBC_Total'].attrs = {'full_name':'total black carbon','units':'kg/m2/day'}
    ds['EmisBC_Total'] *=sec_day
    
def global_w_mean(ds, variable):
    """Takes the globally weighted mean of a dataset and its variable, as long as the dataset has a dz and area"""
    return (ds[variable].weighted(ds['area']*ds['dz']).mean(dim = ['lat','lon','lev']))

def global_sfc_w_mean(ds, variable):
    return (ds[variable].isel(lev = 0).weighted(ds['area']*ds['dz'].isel(lev = 0)).mean(dim = ['lat','lon']))

def f_(raw_f, t_p): 
    """Select our forcing at time tp"""
    return raw_f.interp({'time':t_p})

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