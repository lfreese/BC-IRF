import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys
sys.path.insert(0, '/net/fs11/d0/emfreese/BC-IRF/')
import utils

# Import plant data
CGP_df = pd.read_csv(f'{utils.data_output_path}plants/BC_SE_Asia_all_financing_SEA_GAINS_Springer_plus_rad.csv', index_col=[0])
CGP_df.columns = CGP_df.columns.str.replace(' ', '_')

# Import country boundaries
country_df = geopandas.read_file(f'{utils.raw_data_in_path}/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
country_df = country_df.rename(columns={'SOVEREIGNT': 'country'})

# Countries to analyze
countries = ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']

def get_plant_file_list(CGP_df, countries):
    """
    Get list of all available plant files without loading them
    """
    plant_files = []
    
    plants = CGP_df.loc[(CGP_df['COUNTRY'].isin(countries)) & 
                        (CGP_df['BC_(g/yr)'] > 0)]
    
    for idx, plant in plants.iterrows():
        country = plant['COUNTRY']
        unique_id = plant['unique_ID']
        
        filepath = f'{utils.data_output_path}convolution/gridded_convolution_{country}_{unique_id}_uniqueid.nc'
        
        if os.path.exists(filepath):
            plant_files.append({
                'filepath': filepath,
                'unique_id': unique_id,
                'country': country,
                'commission_year': plant['Year_of_Commission']
            })
    
    return plant_files

def create_cumulative_video_streaming(plant_files, output_path, country_df,
                                     start_year=2000, end_year=2060,
                                     variable='BC_surface_conc', fps=10, 
                                     save_frames=True, max_frames=None):
    """
    Create cumulative video by streaming plant data (loading on-demand)
    Annual timesteps - one frame per year
    
    Parameters:
    -----------
    save_frames : bool
        If True, save individual frames as PNG files
    max_frames : int or None
        If set, only generate this many frames (for testing)
    fps : int
        Frames per second (default 10 means 10 years per second)
    """
    print(f"\nCreating cumulative video from {start_year} to {end_year}")
    print(f"Total plants: {len(plant_files)}")
    print(f"Annual timesteps (1 frame per year)")
    
    # Create frames directory if saving frames
    if save_frames:
        frames_dir = '../figures/videos/frames_cumulative'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    
    # Load one plant to get grid dimensions
    sample_ds = xr.open_dataset(plant_files[0]['filepath'])
    lat = sample_ds.lat.values
    lon = sample_ds.lon.values
    sample_ds.close()
    
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    country_df.plot(ax=ax, color='lightgray', edgecolor='black',
                    linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())
    ax.set_extent([95, 125, -10, 25], crs=ccrs.PlateCarree())
    
    # Estimate vmax by sampling a few plants and summing over the year
    print("Estimating colorbar range...")
    sample_max = 0
    for i in range(min(10, len(plant_files))):
        ds = xr.open_dataset(plant_files[i]['filepath'])
        # Sum over all days in the year
        annual_sum = ds[variable].mean(dim='s').values
        sample_max = max(sample_max, annual_sum.max() * 10)
        ds.close()
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Initialize plot
    zeros = np.zeros((len(lat), len(lon)))
    im = ax.pcolormesh(lon_grid, lat_grid, zeros, transform=ccrs.PlateCarree(),
                       cmap='bone_r', vmin=0, vmax=sample_max, shading='auto')
    
    cbar = plt.colorbar(im, ax=ax, label=f'Annual Mean (μg/m³)', shrink=0.8)
    title = ax.set_title(f'Cumulative Black Carbon Concentration\nYear {start_year}\n0 active plants',
                         fontsize=14, pad=20)
    
    # Calculate total frames (one per year)
    n_frames_full = end_year - start_year
    
    # Use max_frames if specified
    if max_frames is not None:
        n_frames = min(max_frames, n_frames_full)
        print(f"Limiting to first {n_frames} frames (out of {n_frames_full} total)")
    else:
        n_frames = n_frames_full
    
    def update(frame_idx):
        """
        Update function - loads plants on demand for each frame
        Each frame represents one year
        """
        current_year = start_year + frame_idx
        
        # Initialize cumulative concentration array
        cumulative = np.zeros((len(lat), len(lon)))
        n_active = 0
        
        # Stream through plants and add active ones
        for plant_info in plant_files:
            commission_year = plant_info['commission_year']
            
            # Check if plant is active
            if commission_year <= current_year:
                years_since_commission = current_year - commission_year
                
                if years_since_commission < 40:
                    # Load just this plant's data
                    ds = xr.open_dataset(plant_info['filepath'])
                    # Sum over all days to get annual sum concentration
                    plant_conc_annual = ds[variable].mean(dim='s').values
                    ds.close()
                    
                    cumulative += plant_conc_annual
                    n_active += 1
        
        im.set_array(cumulative.ravel())
        title.set_text(f'Cumulative Black Carbon Concentration - Annual Sum\n' + 
                      f'Year {current_year}\n' +
                      f'{n_active} active plants')
        
        # Save frame as PNG
        if save_frames:
            frame_path = f'{frames_dir}/frame_{frame_idx:03d}_year{current_year}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        if frame_idx % 5 == 0:
            print(f"    Frame {frame_idx}/{n_frames}: Year {current_year}, {n_active} active plants")
        
        return im, title
    
    print(f"Total frames: {n_frames} years")
    print(f"Video duration at {fps} fps: {n_frames/fps:.1f} seconds")
    
    print("\nGenerating and saving animation...")
    print(f"Output: {output_path.replace('.mp4', '.gif')}")
    print("Rendering frames:")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Save as GIF using PillowWriter
    output_path_gif = output_path.replace('.mp4', '.gif')
    writer = PillowWriter(fps=fps, metadata=dict(artist='Me'))
    anim.save(output_path_gif, writer=writer)
    print("\nAnimation saved successfully!")
    
    plt.close(fig)

def create_country_comparison_video_streaming(plant_files, output_path, country_df,
                                             start_year=2000, end_year=2060,
                                             variable='BC_surface_conc', fps=10,
                                             save_frames=True, max_frames=None):
    """
    Create 2x2 country comparison video with streaming
    Annual timesteps - one frame per year
    
    Parameters:
    -----------
    save_frames : bool
        If True, save individual frames as PNG files
    max_frames : int or None
        If set, only generate this many frames (for testing)
    fps : int
        Frames per second (default 10 means 10 years per second)
    """
    print(f"\nCreating country comparison video from {start_year} to {end_year}")
    
    # Create frames directory if saving frames
    if save_frames:
        frames_dir = '../figures/videos/frames_comparison'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    
    # Load one plant to get grid dimensions
    sample_ds = xr.open_dataset(plant_files[0]['filepath'])
    lat = sample_ds.lat.values
    lon = sample_ds.lon.values
    sample_ds.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()
    
    # Estimate vmax
    print("Estimating colorbar range...")
    sample_max = 0
    for i in range(min(10, len(plant_files))):
        ds = xr.open_dataset(plant_files[i]['filepath'])
        annual_sum = ds[variable].mean(dim='s').values
        sample_max = max(sample_max, annual_sum.max() * 10)
        ds.close()
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Set up subplots
    ims = []
    titles_list = []
    
    for idx, country in enumerate(countries):
        ax = axes[idx]
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        country_df.plot(ax=ax, color='lightgray', edgecolor='black',
                       linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())
        ax.set_extent([95, 125, -10, 25], crs=ccrs.PlateCarree())
        
        zeros = np.zeros((len(lat), len(lon)))
        im = ax.pcolormesh(lon_grid, lat_grid, zeros, transform=ccrs.PlateCarree(),
                          cmap='bone_r', vmin=0, vmax=sample_max, shading='auto')
        
        plt.colorbar(im, ax=ax, label=f'Annual Mean (μg/m³)', shrink=0.8)
        country_name = country.capitalize()
        title = ax.set_title(f'{country_name}\nYear {start_year}, 0 plants', fontsize=12)
        
        ims.append(im)
        titles_list.append(title)
    
    # Calculate total frames (one per year)
    n_frames_full = end_year - start_year
    
    # Use max_frames if specified
    if max_frames is not None:
        n_frames = min(max_frames, n_frames_full)
        print(f"Limiting to first {n_frames} frames (out of {n_frames_full} total)")
    else:
        n_frames = n_frames_full
    
    def update(frame_idx):
        current_year = start_year + frame_idx
        
        returns = []
        
        for idx, country in enumerate(countries):
            cumulative = np.zeros((len(lat), len(lon)))
            n_active = 0
            
            # Stream through plants from this country
            for plant_info in plant_files:
                if plant_info['country'] == country:
                    commission_year = plant_info['commission_year']
                    
                    if commission_year <= current_year:
                        years_since_commission = current_year - commission_year
                        
                        if years_since_commission < 40:
                            ds = xr.open_dataset(plant_info['filepath'])
                            # Sum over all days to get annual sum
                            plant_conc_annual = ds[variable].mean(dim='s').values
                            ds.close()
                            
                            cumulative += plant_conc_annual
                            n_active += 1
            
            ims[idx].set_array(cumulative.ravel())
            titles_list[idx].set_text(f'{country}\nYear {current_year}, {n_active} plants')
            returns.extend([ims[idx], titles_list[idx]])
        
        # Save frame as PNG
        if save_frames:
            frame_path = f'{frames_dir}/frame_{frame_idx:03d}_year{current_year}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        if frame_idx % 5 == 0:
            print(f"    Frame {frame_idx}/{n_frames}: Year {current_year}")
        
        return returns
    
    print(f"Total frames: {n_frames} years")
    
    print("\nGenerating and saving animation...")
    print(f"Output: {output_path.replace('.mp4', '.gif')}")
    print("Rendering frames:")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Save as GIF using PillowWriter
    output_path_gif = output_path.replace('.mp4', '.gif')
    writer = PillowWriter(fps=fps, metadata=dict(artist='Me'))
    anim.save(output_path_gif, writer=writer)
    print("\nAnimation saved successfully!")
    
    plt.close(fig)

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("Creating BC Pollution Videos - Annual Timesteps")
    print("="*80)
    
    # Get list of available plant files
    print("\nScanning for plant files...")
    plant_files = get_plant_file_list(CGP_df, countries)
    print(f"Found {len(plant_files)} plant files")
    
    # Create output directory
    os.makedirs('../figures/videos', exist_ok=True)
    
    # Create cumulative video (annual timesteps)
    # At 10 fps: 10 years per second, 6 seconds for 60 years
    cumulative_output = '../figures/videos/cumulative_pollution_2000_2060.mp4'
    create_cumulative_video_streaming(plant_files, cumulative_output, country_df,
                                     start_year=2000, end_year=2060, 
                                     variable='BC_surface_conc', fps=10,
                                     save_frames=True, max_frames=None)
    
    # Create country comparison video (annual timesteps)
    comparison_output = '../figures/videos/country_comparison_2000_2060.mp4'
    create_country_comparison_video_streaming(plant_files, comparison_output, country_df,
                                             start_year=2000, end_year=2060,
                                             variable='BC_surface_conc', fps=10,
                                             save_frames=True, max_frames=None)
    
    print("\n" + "="*80)
    print("All videos created successfully!")
    print("="*80)
