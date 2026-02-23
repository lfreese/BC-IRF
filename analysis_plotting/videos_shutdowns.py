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
countries = ['MALAYSIA', 'INDONESIA', 'VIETNAM']

# Shutdown rates for each country
rates = {
    'INDONESIA': [9.0, 18.0, 27.0],
    'MALAYSIA': [0.5, 1.0, 2.0],
    'VIETNAM': [1.0, 3.0, 5.0]
}

# Force closure years
force_closure_by = {
    'MALAYSIA': 2045,
    'INDONESIA': 2040,
    'VIETNAM': 2050
}

# Strategies
strategies = ['Year_of_Commission', 'MW', 'EMISFACTOR.CO2']

strategy_names = {
    'Year_of_Commission': 'Oldest First',
    'MW': 'Largest First',
    'EMISFACTOR.CO2': 'Highest CO2 First'
}

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
                'commission_year': plant['Year_of_Commission'],
                'MW': plant['MW_total'],
                'co2_intensity': plant.get('EMISFACTOR.CO2', 0)
            })
    
    return plant_files

def calculate_shutdown_schedule(plant_files, country, strategy, rate, force_closure_year, start_year=2025):
    """
    Calculate which plants are shut down in each year based on strategy and rate
    
    Returns:
    --------
    dict : {year: [list of plant unique_IDs to shut down that year]}
    """
    # Filter plants for this country
    country_plants = [p for p in plant_files if p['country'] == country]
    
    # Sort plants by strategy
    if strategy == 'Year_of_Commission':
        sorted_plants = sorted(country_plants, key=lambda x: x['commission_year'])
    elif strategy == 'MW':
        sorted_plants = sorted(country_plants, key=lambda x: x['MW'], reverse=True)
    elif strategy == 'EMISFACTOR.CO2':
        sorted_plants = sorted(country_plants, key=lambda x: x['co2_intensity'], reverse=True)
    else:
        sorted_plants = country_plants
    
    # Calculate shutdown schedule
    shutdown_schedule = {}
    total_plants = len(sorted_plants)
    plants_left = total_plants
    plant_idx = 0
    
    for year in range(start_year, force_closure_year + 1):
        years_until_force_closure = force_closure_year - year
        
        if years_until_force_closure > 0:
            plants_to_retire_this_year = int(min(rate, plants_left))
        else:
            # Force all remaining plants to close in the final year
            plants_to_retire_this_year = plants_left
        
        if plants_to_retire_this_year > 0:
            shutdown_schedule[year] = [sorted_plants[plant_idx + i]['unique_id'] 
                                      for i in range(plants_to_retire_this_year)]
            plant_idx += plants_to_retire_this_year
            plants_left -= plants_to_retire_this_year
        else:
            shutdown_schedule[year] = []
    
    return shutdown_schedule

def create_shutdown_strategy_video(plant_files, output_path, country_df, country, 
                                   strategy, rate, force_closure_year,
                                   start_year=2025, end_year=2050,
                                   variable='BC_surface_conc', fps=10, 
                                   save_frames=True, max_frames=None):
    """
    Create video showing pollution reduction under a specific shutdown strategy
    """
    print(f"\n{'='*80}")
    print(f"Creating video for {country} - {strategy_names[strategy]} - {rate} plants/year")
    print(f"{'='*80}")
    
    # Create frames directory if saving frames
    if save_frames:
        safe_strategy_name = strategy.replace('/', '_')
        frames_dir = f'../figures/videos/frames_shutdown_{country}_{safe_strategy_name}_{rate}'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    
    # Calculate shutdown schedule
    shutdown_schedule = calculate_shutdown_schedule(
        plant_files, country, strategy, rate, force_closure_year, start_year
    )
    
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
    
    # Estimate vmax
    print("Estimating colorbar range...")
    sample_max = 0
    for i in range(min(10, len([p for p in plant_files if p['country'] == country]))):
        ds = xr.open_dataset(plant_files[i]['filepath'])
        annual_sum = ds[variable].mean(dim='s').values
        sample_max = max(sample_max, annual_sum.max() * 10)
        ds.close()
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Initialize plot
    zeros = np.zeros((len(lat), len(lon)))
    im = ax.pcolormesh(lon_grid, lat_grid, zeros, transform=ccrs.PlateCarree(),
                       cmap='bone_r', vmin=0, vmax=sample_max, shading='auto')
    
    cbar = plt.colorbar(im, ax=ax, label=f'{variable} - Annual Sum (μg/m³·days)', shrink=0.8)
    title = ax.set_title(f'{country} - {strategy_names[strategy]}\n' +
                         f'Year {start_year} - All plants operating\n' +
                         f'Shutdown rate: {rate} plants/year',
                         fontsize=14, pad=20)
    
    # Calculate total frames (one per year)
    n_frames_full = end_year - start_year
    
    # Use max_frames if specified
    if max_frames is not None:
        n_frames = min(max_frames, n_frames_full)
        print(f"Limiting to first {n_frames} frames (out of {n_frames_full} total)")
    else:
        n_frames = n_frames_full
    
    # Track which plants have been shut down
    shutdown_plants = set()
    
    def update(frame_idx):
        """
        Update function - loads active plants for each year
        """
        current_year = start_year + frame_idx
        
        # Add plants that shut down this year to the set
        if current_year in shutdown_schedule:
            shutdown_plants.update(shutdown_schedule[current_year])
        
        # Initialize cumulative concentration array
        cumulative = np.zeros((len(lat), len(lon)))
        n_active = 0
        
        # Stream through plants and add active ones (not yet shutdown)
        for plant_info in plant_files:
            if plant_info['country'] != country:
                continue
                
            # Skip if plant has been shut down
            if plant_info['unique_id'] in shutdown_plants:
                continue
            
            # Check if plant is commissioned
            if plant_info['commission_year'] > current_year:
                continue
            
            # Load plant data and sum over the year
            ds = xr.open_dataset(plant_info['filepath'])
            plant_conc_annual = ds[variable].mean(dim='s').values
            ds.close()
            
            cumulative += plant_conc_annual
            n_active += 1
        
        im.set_array(cumulative.ravel())
        
        # Calculate reduction percentage
        if frame_idx == 0:
            reduction_pct = 0
        else:
            baseline = update.baseline_concentration
            if baseline > 0:
                reduction_pct = ((baseline - cumulative.mean()) / baseline) * 100
            else:
                reduction_pct = 0
        
        title.set_text(f'{country} - {strategy_names[strategy]}\n' +
                      f'Year {current_year} - {n_active} active plants\n' +
                      f'Shutdown: {len(shutdown_plants)} plants ({reduction_pct:.1f}% reduction)')
        
        # Store baseline for first frame
        if frame_idx == 0:
            update.baseline_concentration = cumulative.mean()
        
        # Save frame as PNG
        if save_frames:
            frame_path = f'{frames_dir}/frame_{frame_idx:03d}_year{current_year}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        if frame_idx % 5 == 0:
            print(f"    Frame {frame_idx}/{n_frames}: Year {current_year}, " +
                  f"{n_active} active, {len(shutdown_plants)} shut down")
        
        return im, title
    
    # Initialize baseline storage
    update.baseline_concentration = 0
    
    print(f"Total frames: {n_frames} years")
    print(f"Video duration at {fps} fps: {n_frames/fps:.1f} seconds")
    
    print("\nGenerating and saving animation...")
    print(f"Output: {output_path}")
    print("Rendering frames:")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    # Save as GIF using PillowWriter
    writer = PillowWriter(fps=fps, metadata=dict(artist='Me'))
    anim.save(output_path, writer=writer)
    print("\nAnimation saved successfully!")
    
    plt.close(fig)

def create_comparison_video(plant_files, output_path, country_df,
                           start_year=2025, end_year=2050,
                           variable='BC_surface_conc', fps=10,
                           save_frames=True, max_frames=None):
    """
    Create 3x3 grid comparing strategies and rates for all countries
    Rows = strategies, Columns = countries
    """
    print(f"\n{'='*80}")
    print(f"Creating comparison video for all countries and strategies")
    print(f"{'='*80}")
    
    if save_frames:
        frames_dir = '../figures/videos/frames_comparison_strategies'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    
    # Load one plant to get grid dimensions
    sample_ds = xr.open_dataset(plant_files[0]['filepath'])
    lat = sample_ds.lat.values
    lon = sample_ds.lon.values
    sample_ds.close()
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(24, 20),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Estimate global vmax
    print("Estimating colorbar range...")
    sample_max = 0
    for i in range(min(30, len(plant_files))):
        ds = xr.open_dataset(plant_files[i]['filepath'])
        annual_sum = ds[variable].mean(dim='s').values
        sample_max = max(sample_max, annual_sum.max() * 5)
        ds.close()
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Set up each subplot
    ims = []
    titles = []
    shutdown_schedules = []
    
    for row_idx, strategy in enumerate(strategies):
        for col_idx, country in enumerate(countries):
            ax = axes[row_idx, col_idx]
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            country_df.plot(ax=ax, color='lightgray', edgecolor='black',
                          linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())
            ax.set_extent([95, 125, -10, 25], crs=ccrs.PlateCarree())
            
            zeros = np.zeros((len(lat), len(lon)))
            im = ax.pcolormesh(lon_grid, lat_grid, zeros, transform=ccrs.PlateCarree(),
                              cmap='bone_r', vmin=0, vmax=sample_max, shading='auto')
            
            plt.colorbar(im, ax=ax, shrink=0.6)
            
            # Use middle rate for each country
            rate = rates[country][1]  # Middle rate
            
            title = ax.set_title(f'{country}\n{strategy_names[strategy]}\n' +
                               f'{rate} plants/yr',
                               fontsize=10)
            
            ims.append(im)
            titles.append(title)
            
            # Calculate shutdown schedule for this combination
            schedule = calculate_shutdown_schedule(
                plant_files, country, strategy, rate,
                force_closure_by[country], start_year
            )
            shutdown_schedules.append({
                'country': country,
                'schedule': schedule,
                'shutdown_plants': set()
            })
    
    n_frames_full = end_year - start_year
    if max_frames is not None:
        n_frames = min(max_frames, n_frames_full)
    else:
        n_frames = n_frames_full
    
    def update(frame_idx):
        current_year = start_year + frame_idx
        returns = []
        
        for idx, (im, title, sched_info) in enumerate(zip(ims, titles, shutdown_schedules)):
            country = sched_info['country']
            schedule = sched_info['schedule']
            shutdown_plants = sched_info['shutdown_plants']
            
            # Update shutdown list
            if current_year in schedule:
                shutdown_plants.update(schedule[current_year])
            
            cumulative = np.zeros((len(lat), len(lon)))
            n_active = 0
            
            for plant_info in plant_files:
                if plant_info['country'] != country:
                    continue
                if plant_info['unique_id'] in shutdown_plants:
                    continue
                if plant_info['commission_year'] > current_year:
                    continue
                
                ds = xr.open_dataset(plant_info['filepath'])
                plant_conc_annual = ds[variable].mean(dim='s').values
                ds.close()
                
                cumulative += plant_conc_annual
                n_active += 1
            
            im.set_array(cumulative.ravel())
            
            row_idx = idx // 3
            col_idx = idx % 3
            strategy = strategies[row_idx]
            rate = rates[country][1]
            
            title.set_text(f'{country}\n{strategy_names[strategy]}\n' +
                          f'{n_active} active plants')
            
            returns.extend([im, title])
        
        if save_frames:
            frame_path = f'{frames_dir}/frame_{frame_idx:03d}_year{current_year}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        if frame_idx % 5 == 0:
            print(f"    Frame {frame_idx}/{n_frames}: Year {current_year}")
        
        return returns
    
    print(f"Total frames: {n_frames} years")
    print("\nGenerating and saving animation...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    writer = PillowWriter(fps=fps, metadata=dict(artist='Me'))
    anim.save(output_path, writer=writer)
    print("\nAnimation saved successfully!")
    
    plt.close(fig)



# # Main execution
# if __name__ == "__main__":
#     print("="*80)
#     print("Creating Shutdown Strategy Videos")
#     print("="*80)
    
#     # Get list of available plant files
#     print("\nScanning for plant files...")
#     plant_files = get_plant_file_list(CGP_df, countries)
#     print(f"Found {len(plant_files)} plant files")
    
#     # Create output directory
#     os.makedirs('../figures/videos', exist_ok=True)
    
#     # Option 1: Create individual videos for each country/strategy/rate combination
#     # Uncomment to generate all combinations
#     """
#     for country in countries:
#         for strategy in strategies:
#             for rate in rates[country]:
#                 safe_strategy_name = strategy.replace('/', '_')
#                 output_path = f'../figures/videos/shutdown_{country}_{safe_strategy_name}_{rate}plants.gif'
                
#                 create_shutdown_strategy_video(
#                     plant_files, output_path, country_df,
#                     country, strategy, rate, force_closure_by[country],
#                     variable='BC_surface_conc', fps=10,
#                     save_frames=True, max_frames=None
#                 )
#     """
    
#     # Option 2: Create comparison grid video (using middle rates)
#     comparison_output = '../figures/videos/shutdown_strategies_comparison.gif'
#     create_comparison_video(
#         plant_files, comparison_output, country_df,
#         variable='BC_surface_conc', fps=10,
#         save_frames=True, max_frames=None
#     )
    
#     print("\n" + "="*80)
#     print("All videos created successfully!")
#     print("="*80)


def create_difference_from_baseline_video(plant_files, output_path, country_df,
                                         start_year=2025, end_year=2050,
                                         variable='BC_surface_conc', fps=10,
                                         save_frames=True, max_frames=None):
    """
    Create 3x3 grid showing DIFFERENCE from no-action baseline
    Shows: [All countries baseline] - [All countries with one doing early shutdown]
    Uses diverging colorbar (blue=reduction, red=increase)
    Rows = strategies, Columns = countries doing shutdown
    """
    print(f"\n{'='*80}")
    print(f"Creating difference-from-baseline video")
    print(f"Baseline: All countries, no shutdowns")
    print(f"Scenarios: All countries, with one country doing early shutdowns")
    print(f"{'='*80}")
    
    if save_frames:
        frames_dir = '../figures/videos/frames_difference_baseline'
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
    
    # Load one plant to get grid dimensions
    sample_ds = xr.open_dataset(plant_files[0]['filepath'])
    lat = sample_ds.lat.values
    lon = sample_ds.lon.values
    sample_ds.close()
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(24, 20),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Estimate vmax for difference (symmetric around 0)
    print("Estimating colorbar range...")
    sample_max = 0
    for i in range(min(30, len(plant_files))):
        ds = xr.open_dataset(plant_files[i]['filepath'])
        annual_mean = ds[variable].mean(dim='s').values
        sample_max = max(sample_max, annual_mean.max() * 5)
        ds.close()
    
    # Use symmetric limits for diverging colorbar
    # Estimate a good range for differences (smaller than absolute values)
    vmin = -sample_max * 0.3
    vmax = sample_max * 0.3
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Set up each subplot
    ims = []
    titles = []
    shutdown_schedules = []
    baseline_data = {}  # Store baseline for each year (all countries, no shutdowns)
    
    for row_idx, strategy in enumerate(strategies):
        for col_idx, shutdown_country in enumerate(countries):
            ax = axes[row_idx, col_idx]
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            country_df.plot(ax=ax, color='lightgray', edgecolor='black',
                          linewidth=0.5, alpha=0.3, transform=ccrs.PlateCarree())
            ax.set_extent([95, 125, -10, 25], crs=ccrs.PlateCarree())
            
            zeros = np.zeros((len(lat), len(lon)))
            im = ax.pcolormesh(lon_grid, lat_grid, zeros, transform=ccrs.PlateCarree(),
                              cmap='RdBu',  # Diverging: Blue=reduction, Red=increase
                              vmin=vmin, vmax=vmax, shading='auto')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label('Δ Concentration\n(μg/m³)', fontsize=8)
            
            # Use middle rate for each country
            rate = rates[shutdown_country][1]
            
            title = ax.set_title(f'{shutdown_country} shuts down\n{strategy_names[strategy]}\n' +
                               f'{rate} plants/yr',
                               fontsize=10)
            
            ims.append(im)
            titles.append(title)
            
            # Calculate shutdown schedule for this country only
            schedule = calculate_shutdown_schedule(
                plant_files, shutdown_country, strategy, rate,
                force_closure_by[shutdown_country], start_year
            )
            shutdown_schedules.append({
                'shutdown_country': shutdown_country,
                'schedule': schedule,
                'shutdown_plants': set()
            })
    
    n_frames_full = end_year - start_year
    if max_frames is not None:
        n_frames = min(max_frames, n_frames_full)
    else:
        n_frames = n_frames_full
    
    def calculate_baseline(current_year):
        """
        Calculate no-action baseline for a given year
        All countries, all plants operating (until 40 years old)
        """
        cumulative = np.zeros((len(lat), len(lon)))
        
        for plant_info in plant_files:
            if plant_info['commission_year'] > current_year:
                continue
            
            # In baseline, plants operate for 40 years from commission
            years_since_commission = current_year - plant_info['commission_year']
            if years_since_commission >= 40:
                continue
            
            ds = xr.open_dataset(plant_info['filepath'])
            plant_conc_annual = ds[variable].mean(dim='s').values
            ds.close()
            
            cumulative += plant_conc_annual
        
        return cumulative
    
    def update(frame_idx):
        current_year = start_year + frame_idx
        returns = []
        
        # Calculate baseline for this year if not already done
        # Baseline = all countries, all plants (no shutdowns)
        if current_year not in baseline_data:
            print(f"    Calculating baseline for year {current_year}...")
            baseline_data[current_year] = calculate_baseline(current_year)
        
        baseline = baseline_data[current_year]
        
        for idx, (im, title, sched_info) in enumerate(zip(ims, titles, shutdown_schedules)):
            shutdown_country = sched_info['shutdown_country']
            schedule = sched_info['schedule']
            shutdown_plants = sched_info['shutdown_plants']
            
            # Update shutdown list for the shutting-down country
            if current_year in schedule:
                shutdown_plants.update(schedule[current_year])
            
            # Calculate scenario concentration
            # All countries operating, but one country has early shutdowns
            scenario = np.zeros((len(lat), len(lon)))
            n_shutdown = 0
            n_active = 0
            
            for plant_info in plant_files:
                # Skip plants that haven't been commissioned yet
                if plant_info['commission_year'] > current_year:
                    continue
                
                # Check if this plant is from the shutting-down country and has been shut down
                if plant_info['country'] == shutdown_country and plant_info['unique_id'] in shutdown_plants:
                    n_shutdown += 1
                    continue  # Don't include this plant
                
                # Check if plant has naturally aged out (40 years)
                years_since_commission = current_year - plant_info['commission_year']
                if years_since_commission >= 40:
                    continue
                
                # Include this plant
                ds = xr.open_dataset(plant_info['filepath'])
                plant_conc_annual = ds[variable].mean(dim='s').values
                ds.close()
                
                scenario += plant_conc_annual
                n_active += 1
            
            # Calculate DIFFERENCE (baseline - scenario)
            # Positive values = reduction (good), shown in blue
            difference = baseline - scenario
            
            im.set_array(difference.ravel())
            
            row_idx = idx // 3
            col_idx = idx % 3
            strategy = strategies[row_idx]
            rate = rates[shutdown_country][1]
            
            # Calculate percentage reduction
            baseline_mean = baseline.mean()
            if baseline_mean > 0:
                reduction_pct = 100 * difference.mean() / baseline_mean
            else:
                reduction_pct = 0
            
            # Calculate max reduction (in hotspot)
            max_reduction = difference.max()
            
            country_name = shutdown_country.capitalize()

            title.set_text(f'{country_name} shuts down\n{strategy_names[strategy]}\n' +
                        f'{current_year} - {reduction_pct:.1f}% reduction, {n_shutdown} plants shut down')
            title.set_fontsize(14)

            
            returns.extend([im, title])
        
        if save_frames:
            frame_path = f'{frames_dir}/frame_{frame_idx:03d}_year{current_year}.png'
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        if frame_idx % 5 == 0:
            print(f"    Frame {frame_idx}/{n_frames}: Year {current_year}")
        
        return returns
    
    print(f"Total frames: {n_frames} years")
    print("\nGenerating and saving animation...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True)
    
    writer = PillowWriter(fps=fps, metadata=dict(artist='Me'))
    anim.save(output_path, writer=writer)
    print("\nAnimation saved successfully!")
    
    plt.close(fig)

# Add to main execution:
if __name__ == "__main__":
    print("="*80)
    print("Creating Shutdown Strategy Videos")
    print("="*80)
    
    # Get list of available plant files
    print("\nScanning for plant files...")
    plant_files = get_plant_file_list(CGP_df, countries)
    print(f"Found {len(plant_files)} plant files")
    
    # Create output directory
    os.makedirs('../figures/videos', exist_ok=True)
    
    # Create difference-from-baseline video with diverging colorbar
    difference_output = '../figures/videos/shutdown_difference_from_baseline.gif'
    create_difference_from_baseline_video(
        plant_files, difference_output, country_df,
        variable='BC_surface_conc', fps=2.5,
        save_frames=True, max_frames=None
    )
    
    print("\n" + "="*80)
    print("All videos created successfully!")
    print("="*80)