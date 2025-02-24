#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --mem=MaxMemPerCPU
#SBATCH --cpus-per-task=16
#SBATCH --partition=edr

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import pandas as pd
from geopy.geocoders import Nominatim
from matplotlib.colors import SymLogNorm
from matplotlib.pyplot import cm

import xesmf as xe
#from pykrige.ok import OrdinaryKriging
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import dask
import utils


import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs

import geopandas


# In[3]:


## look at https://wedocs.unep.org/bitstream/handle/20.500.11822/11406/Pog_&amp;_iPog_-_Energy_and_Fuels.pdf?sequence=1&amp%3BisAllowed=

## for emissions factors scaling

## fix tott


# ## Set constants

# In[10]:


coal_year_range = np.arange(0,41)
percents = [0, 10, 20,30,40,50,60, 70,80,90, 99]
years = [0,4,10,16,20,24,30,36]
coal_year_range = np.arange(0,41)


# In[11]:


## Add time dimension
length_simulation = 50*365

time_array = np.arange(0, length_simulation)


# ## Data read in

# ### Emissions dataframe

# In[12]:


CGP_df = pd.read_csv('mod_coal_inputs/BC_limited_country_SEA_GAINS_Springer.csv')


# In[13]:


CGP_df.columns = CGP_df.columns.str.replace(' ', '_')


# In[14]:


CGP_df = CGP_df.rename(columns = {'YEAR':'Year_of_Commission', 'EMISFACTOR.PLATTS':'CO2_weighted_capacity_1000tonsperMW'})


# ### Convolution

# In[15]:


da = {}
for yr in years:
    print(yr)
    da[yr] = {}
    for pc in percents:
        print(yr, pc)
        da[yr][pc] = xr.open_mfdataset(f'Outputs/C_out_{pc}_{yr}.nc', chunks = 'auto')
        
da2 = {}
for pc in percents:
    da2[pc] = xr.concat([da[yr][pc] for yr in years], pd.Index([yr for yr in years], name = 'year_shutdown'))
    
ds = xr.concat([da2[pc] for pc in percents], pd.Index([pc for pc in percents], name = 'percent'))
ds = ds.rename({'__xarray_dataarray_variable__':'BC_conc'})
ds['BC_conc']*=1e9
ds['BC_conc'].attrs = {'units':'ppb'}


# In[16]:


def ppb_to_ug(ds, species_to_convert, mw_species_list, stp_p = 101325, stp_t = 298.):
    '''Convert species to ug/m3 from ppb'''
    R = 8.314 #J/K/mol
    ppb_ugm3 = (stp_p / stp_t / R) #Pa/K/(J/K/mol) = g/m^3

    for spec in species_to_convert:
        attrs = ds[spec].attrs
        ds[spec] = ds[spec]*mw_species_list[spec]*ppb_ugm3 #ppb*g/mol*g/m^3
        ds[spec].attrs['units'] = 'μg m-3'
    #return(ds)

mw_BC = {'BC_conc':12.011}
ppb_to_ug(ds, ['BC_conc'], mw_BC)


# In[17]:


ds['percent'] = 100- ds['percent']


# ### Area for weighting

# In[18]:


ds_area = xr.open_dataset('/net/fs11/d0/emfreese/GCrundirs/IRF_runs/stretch_2x_pulse/SEA/Jan/mod_output/GEOSChem.SpeciesConc.20160101_0000z.nc4', engine = 'netcdf4')


# In[19]:


ds_area = utils.fix_area_ij_latlon(ds_area)


# ## Country Mask

# In[20]:


country_mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
countries = ['China']#,'Indonesia','Malaysia','Vietnam','Australia', 'Cambodia','Myanmar', 'Laos','Philippines','Nepal','Bangladesh','Thailand','Bhutan']
print('countries uploaded')


# ## Dataframes by country exposure (sum and mean)

# In[21]:


percents = [100-p for p in percents]


# In[22]:


percents


# In[23]:


mask = country_mask.mask(ds, lon_name = 'lon', lat_name = 'lat')
contiguous_mask = ~np.isnan(mask)& (mask == country_mask.map_keys(countries))


# In[24]:


China_ds = ds.where(contiguous_mask).weighted(ds_area['area']).mean(dim = ['lat','lon']).sum(dim = ['s'])


# In[ ]:


print(China_ds['BC_conc'].values)


# In[ ]:


China_ds.to_netcdf(f'Outputs/China_early_shutdown_co2pct_sumBC.nc')


# In[ ]:




