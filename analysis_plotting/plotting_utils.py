import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
def prepare_concentration_data(full_ds, map_locations, shutdown_years, 
                             variable='BC_pop_weight_mean_conc',
                             normalize=False,
                             emissions_df=None):
    """
    Prepare concentration data for mapping by merging with geodataframe.
    
    Parameters:
    -----------
    full_ds : xarray.Dataset
        Dataset containing BC concentration data
    map_locations : list
        List of locations to plot (e.g., ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM', 'all'])
    shutdown_years : array-like
        Years to plot
    variable : str, optional
        Variable to plot from the dataset (default: 'BC_pop_weight_mean_conc')
    normalize : bool, optional
        Whether to normalize by annual BC emissions (default: False)
    emissions_df : pandas.DataFrame, optional
        DataFrame containing BC emissions data (required if normalize=True)
        Should have 'COUNTRY' and 'BC_(g/yr)' columns
    
    Returns:
    --------
    dict
        Nested dictionary with structure {location: {year: geopandas.GeoDataFrame}}
    """
    if normalize and emissions_df is None:
        raise ValueError("emissions_df must be provided when normalize=True")
        
    conc_by_location = {}
    
    for loc in map_locations:
        conc_by_location[loc] = {}
        for yr in shutdown_years:
            # Get concentration data
            if loc == 'all':
                data = full_ds[variable].sel(closure_year=yr).sum(dim='unique_ID')
            else:
                data = (full_ds
                       .where(full_ds.country_emitting == loc, drop=True)
                       .sel(closure_year=yr)[variable]
                       .sum(dim='unique_ID'))
            
            # Create GeoDataFrame
            gdf_data = pd.merge(
                gdf, 
                data.to_dataframe(), 
                on='country_impacted'
            )
            
            # Normalize if requested
            if normalize:
                if loc == 'all':
                    total_emissions = emissions_df.groupby('COUNTRY').sum()['BC_(g/yr)'].sum() * yr
                else:
                    total_emissions = emissions_df.groupby('COUNTRY').sum()['BC_(g/yr)'][loc] * yr
                
                gdf_data[variable] = gdf_data[variable] / total_emissions
            
            conc_by_location[loc][yr] = gdf_data
    
    return conc_by_location


def plot_concentration_maps(conc_by_location, map_locations, shutdown_years,
                          variable='BC_pop_weight_mean_conc',
                          vmin=1e-6, vmax=1e-1,
                          figsize=(25,15),
                          save_path=None):
    """
    Create maps of BC concentrations using pre-prepared data.
    
    Parameters:
    -----------
    conc_by_location : dict
        Nested dictionary of prepared data from prepare_concentration_data()
    map_locations : list
        List of locations to plot
    shutdown_years : array-like
        Years to plot
    variable : str, optional
        Variable being plotted (for labels)
    vmin, vmax : float, optional
        Min and max values for color normalization
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        If provided, save figure to this path
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        Array of matplotlib axes
    """
    # Set up the figure
    n_years = len(shutdown_years)
    n_locations = len(map_locations)
    fig, axes = plt.subplots(n_years, n_locations, 
                            figsize=figsize, 
                            sharex=True, sharey=True, 
                            constrained_layout=True)
    
    # Define impacted countries
    all_countries = list(set(impacted_countries + ['Malaysia', 'Cambodia', 'Indonesia', 'Vietnam']))
    
    # Create plots
    for col_idx, location in enumerate(map_locations):
        for row_idx, year in enumerate(shutdown_years):
            ax = axes[row_idx, col_idx]
            
            # Plot concentrations
            conc_by_location[location][year].plot(
                ax=ax,
                column=variable,
                cmap='viridis_r',
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
            )
            
            # Plot country boundaries
            country_df[country_df['country'].isin(all_countries)].boundary.plot(
                ax=ax, 
                color='k', 
                linewidth=0.5
            )
            
            # Add hatching for emitting countries
            if location in ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']:
                country_df[country_df['country'] == location.capitalize()].boundary.plot(
                    ax=ax, 
                    color='k', 
                    linewidth=0.5, 
                    hatch="\\\\////"
                )
            elif location == 'all':
                country_df[country_df['country'].isin(['Vietnam','Indonesia','Cambodia','Malaysia'])].boundary.plot(
                    ax=ax, 
                    color='k', 
                    linewidth=0.5, 
                    hatch="\\\\////"
                )
            
            # Set titles and labels
            if row_idx == 0:
                axes[0, col_idx].set_title(f'Emissions from {location.capitalize()}', 
                                         fontsize=18)
            if col_idx == 0:
                axes[row_idx, 0].set_ylabel(f'Shutdown in {year}', 
                                          fontsize=18)
            
            # Set map extent
            ax.set_xlim(75, 155)
            ax.set_ylim(-35, 45)
    
    # Add colorbar
    cb_ax = fig.add_axes([0.2, -0.05, 0.6, 0.02])
    mpl.colorbar.ColorbarBase(
        ax=cb_ax, 
        cmap='viridis_r', 
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        orientation='horizontal', 
        label='$ng/m^3/kg/day$'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, axes


def analyze_emissions_by_var(full_ds, var):
    """
    Analyze emissions by progressively including plants sorted by commissioning year.
    
    Parameters:
    -----------
    full_ds : xarray.Dataset
        Dataset containing power plant data with 'unique_ID' and 'Year_of_Commission'
    
    Returns:
    --------
    age_ds : xarray.Dataset
        Emissions from plants commissioned up to each threshold
    bckg_age_ds : xarray.Dataset
        Emissions from plants commissioned after each threshold
    """
    # Sort the dataset in descending order
    sorted_ds = full_ds.sortby(var, ascending=False)
    
    # Get the number of plants
    n_plants = len(sorted_ds['unique_ID'])
    
    # Initialize dictionaries to store results
    age_ds = {}
    bckg_age_ds = {}
    
    # List of ARTP index variables that should use mean aggregation
    rf_vars = []
    #     'Global_10', 'Global_20', 'Global_50', 'Global_100',
    #     'N_arctic_10', 'N_arctic_20', 'N_arctic_50', 'N_arctic_100',
    #     'N_midlat_10', 'N_midlat_20', 'N_midlat_50', 'N_midlat_100',
    #     'S_subtropics_10', 'S_subtropics_20', 'S_subtropics_50', 'S_subtropics_100',
    #     'Tropics_10', 'Tropics_20', 'Tropics_50', 'Tropics_100', 'drf', 'snowrf', 'dt_drf', 'dt_snowrf'
    # ]
    
    # Create datasets for each threshold
    for idx in range(1, n_plants + 1):  # Start from 1 to n_plants
        # Oldest plants up to idx
        # For ARTP index variables, use mean instead of sum
        if var in rf_vars:
            age_ds[idx] = sorted_ds.isel(unique_ID=slice(0, idx)).mean(dim='unique_ID')
            bckg_age_ds[idx] = sorted_ds.isel(unique_ID=slice(idx, n_plants)).mean(dim='unique_ID')
        else:
            age_ds[idx] = sorted_ds.isel(unique_ID=slice(0, idx)).sum(dim='unique_ID')
            bckg_age_ds[idx] = sorted_ds.isel(unique_ID=slice(idx, n_plants)).sum(dim='unique_ID')

    # Concatenate along a new dimension representing the threshold
    age_ds = xr.concat([age_ds[i] for i in range(1, n_plants + 1)], 
                       pd.Index(range(1, n_plants + 1), name='plants_open'))
    
    bckg_age_ds = xr.concat([bckg_age_ds[i] for i in range(1, n_plants + 1)], 
                           pd.Index(range(1, n_plants + 1), name='plants_open'))
    
    return age_ds, bckg_age_ds

def plot_variable_by_country(dataset, variable, country=None, emitting_country=None, scenario='main', 
                           contour_variable=None, levels=10,
                           figsize=(10, 6), ax=None, target_year=2040, target_co2=10):
    """
    Plot a variable for a specific country and scenario, with optional contour overlay.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        The dataset containing the variable to plot
    variable : str
        The variable to plot (e.g., 'BC_surface_mean_conc', 'co2_emissions')
    country : str or None
        Country receiving impacts to select (None for variables without country dimension)
    emitting_country : str or None
        Country emitting pollutants to filter by (None for all emitting countries)
    scenario : str
        Scenario to plot (default: 'main')
    contour_variable : str or None
        Optional second variable to plot as contours
    levels : int
        Number of contour levels to plot
    figsize : tuple
        Figure size as (width, height)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    target_year : int
        Year to draw vertical reference line (default: 2040)
    target_co2 : float
        CO2 emissions target in GtCO2 for horizontal reference line (default: 10)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Filter by emitting country if specified
    if emitting_country is not None:
        # Get the unique_IDs for plants in the emitting country
        plant_ids = dataset.where(dataset.country_emitting == emitting_country, drop=True).unique_ID
        
        # If we found plants, filter the dataset to only include those plants
        if len(plant_ids) > 0:
            dataset = dataset.sel(unique_ID=plant_ids)
    
    # Select data based on provided parameters
    data = dataset[variable]
    
    # Apply country selection if applicable (for impacted country)
    if country is not None and 'country_impacted' in data.dims:
        data = data.sel(country_impacted=country)
    
    # Apply scenario selection
    if 'scenario' in data.dims:
        data = data.sel(scenario=scenario)
    
    # Plot the main data
    im = data.plot(ax=ax, cmap='cividis', add_colorbar=False)
    
    # Add colorbar with custom label
    cbar = plt.colorbar(im, ax=ax)
    if variable == 'BC_surface_mean_conc':
        cbar.set_label('BC Surface Concentration (ng m$^{-3}$)')
    elif variable == 'co2_emissions':
        cbar.set_label('CO$_2$ Emissions (Gt yr$^{-1}$)')
    elif variable == 'BC_pop_weight_mean_conc':
        cbar.set_label('BC Population Weighted Surface Concentration (ng m$^{-3}$)')
    elif variable == 'BC_column_mean_conc':
        cbar.set_label('BC Column Concentration (ng m$^{-3}$)')

    # Add contour plot if requested
    if contour_variable is not None:
        contour_data = dataset[contour_variable]
        
        # Apply same selections to contour data
        if country is not None and 'country_impacted' in contour_data.dims:
            contour_data = contour_data.sel(country_impacted=country)
        if 'scenario' in contour_data.dims:
            contour_data = contour_data.sel(scenario=scenario)
            
        # Convert CO2 emissions to GtCO2 if that's the contour variable
        if contour_variable == 'co2_emissions':
            contour_data = contour_data / 1e9  # Convert to GtCO2
            contour_units = 'GtCO₂'
            
            # Find the y-value (number of plants) that corresponds to target CO2 at target year
            if 'closure_year' in contour_data.dims:
                # Get the data for the target year
                year_data = contour_data.sel(closure_year=target_year)
                if len(year_data) > 0:
                    # Convert to numpy array for easier manipulation
                    year_values = year_data.values
                    plants_values = year_data.plants_open.values
                    
                    # Find the closest number of plants that gives target CO2
                    idx = np.abs(year_values - target_co2).argmin()
                    plants_for_target = plants_values[idx]
                    
                    # Get current axis limits
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    
                    # Plot vertical line from bottom to target plants
                    ax.plot([target_year, target_year], 
                           [ylim[0], plants_for_target],
                           color='white', linestyle='--', alpha=1, linewidth=2)
                    
                    # Plot horizontal line from left to target year
                    ax.plot([xlim[0], target_year],
                           [plants_for_target, plants_for_target],
                           color='white', linestyle='--', alpha=1, linewidth=2)
        else:
            contour_units = ''
            
        # Create contour plot
        cs = contour_data.plot.contour(ax=ax, levels=levels, cmap='gist_heat_r', add_colorbar=False)
        ax.clabel(cs, inline=True, fontsize=8)
    
    # Set axis labels
    ax.set_ylabel('Number of Plants Open')
    ax.set_xlabel('Year of Closure')
    
    # Add secondary y-axis for MW capacity
    ax2 = ax.twinx()
    
    # Get the MW values for each number of plants
    if 'MW' in dataset:
        # Get the MW values (already summed)
        mw_data = dataset['MW'].sel(plants_open=slice(None, None))
        # Convert to GW for better readability
        mw_data = mw_data / 1000  # Convert to GW
        
        # Get the maximum MW value for setting axis limits
        max_mw = mw_data.max().values
        
        # Set the secondary axis limits to match the primary axis
        ax2.set_ylim(ax.get_ylim())
        
        # Use actual MW values for ticks
        n_ticks = 8  # Number of ticks to show
        mw_ticks = np.linspace(0, max_mw, n_ticks)
        # Find the closest plant numbers for these MW values
        plant_ticks = []
        for mw in mw_ticks:
            idx = np.abs(mw_data.values - mw).argmin()
            plant_ticks.append(mw_data.plants_open.values[idx])
        
        # Format the tick labels
        mw_labels = [f'{mw:.0f}' for mw in mw_ticks]
        
        # Set the secondary axis ticks and labels
        ax2.set_yticks(plant_ticks)
        ax2.set_yticklabels(mw_labels)
        ax2.set_ylabel('Cumulative Capacity (GW)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Store the MW data for later use
        ax2.mw_data = mw_data
    
    # Create title with appropriate information
    variable_name = variable.replace('_', ' ').title()
    country_info = f" in {country}" if country is not None else ""
    emitting_info = f" from {emitting_country}" if emitting_country is not None else ""
    scenario_info = f" ({scenario} scenario)" if 'scenario' in data.dims else ""
    
    title = f"{variable_name}{country_info}{emitting_info}{scenario_info}"
    if contour_variable is not None:
        title += f" with {contour_variable.replace('_', ' ').title()} contours"
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if ax is None:  # Only apply tight_layout if we created the figure
        plt.tight_layout()
    return fig, ax

def plot_comparison(age_ds, mw_ds, emis_intens_ds, variable, country=None, emitting_country=None, 
                   scenario='main', contour_variable=None, levels=10, figsize=(30, 8), 
                   target_year=2040, target_co2=10):
    """
    Plot three side-by-side plots comparing the same variable across different datasets.
    
    Parameters:
    -----------
    age_ds : xarray.Dataset
        Dataset sorted by age
    mw_ds : xarray.Dataset
        Dataset sorted by plant size
    emis_intens_ds : xarray.Dataset
        Dataset sorted by emission intensity
    variable : str
        The variable to plot
    country : str or None
        Country receiving impacts to select
    emitting_country : str or None
        Country emitting pollutants to filter by (None for all emitting countries)
    scenario : str
        Scenario to plot
    contour_variable : str or None
        Optional second variable to plot as contours
    levels : int
        Number of contour levels
    figsize : tuple
        Figure size as (width, height)
    target_year : int
        Year to draw vertical reference line (default: 2040)
    target_co2 : float
        CO2 emissions target in GtCO2 for horizontal reference line (default: 10)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        Array of matplotlib axes
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    # Plot for age-sorted dataset
    plot_variable_by_country(age_ds, variable, country, emitting_country, scenario, 
                           contour_variable, levels, ax=axes[0],
                           target_year=target_year, target_co2=target_co2)
    axes[0].set_title("Sorted by Age (oldest first)")
    
    # Plot for size-sorted dataset
    plot_variable_by_country(mw_ds, variable, country, emitting_country, scenario, 
                           contour_variable, levels, ax=axes[1],
                           target_year=target_year, target_co2=target_co2)
    axes[1].set_title("Sorted by Plant Size (largest first)")
    
    # Plot for emission intensity-sorted dataset
    plot_variable_by_country(emis_intens_ds, variable, country, emitting_country, scenario, 
                           contour_variable, levels, ax=axes[2],
                           target_year=target_year, target_co2=target_co2)
    axes[2].set_title("Sorted by Emission Intensity (highest first)")
    
    plt.tight_layout()
    return fig, axes



def plot_emissions_and_temperature(CGP_df, start_year=2000, end_year=2060, tcr=1.65, tcr_uncertainty=0.4, 
                                 breakdown_by=None, show_lifetime_uncertainty=True, lifetime_uncertainty=5,
                                 figsize=(12, 8)):
    """
    Plot cumulative CO2 emissions and temperature response over time.
    
    Parameters:
    -----------
    CGP_df : pandas.DataFrame
        DataFrame containing power plant data
    start_year : int
        Starting year for the analysis (default: 2000)
    end_year : int
        Ending year for the analysis (default: 2060)
    tcr : float
        Transient Climate Response factor in °C per 1000 GtCO2 (default: 1.65)
    tcr_uncertainty : float
        Uncertainty in TCR value (default: 0.4)
    breakdown_by : str, optional
        Column to break down emissions by ('country' or 'plant_type')
    show_lifetime_uncertainty : bool
        Whether to show shading for lifetime uncertainty (default: True)
    lifetime_uncertainty : int
        Number of years of uncertainty in plant lifetime (default: 5)
    figsize : tuple
        Figure size as (width, height)
    """
    # Set up time array
    years = end_year - start_year
    time_array = np.arange(start_year, end_year)
    
    # Initialize arrays for emissions
    total_emissions = np.zeros(years)
    if show_lifetime_uncertainty:
        min_emissions = np.zeros(years)
        max_emissions = np.zeros(years)
    
    # If breaking down emissions, create separate arrays for each category
    if breakdown_by:
        categories = CGP_df[breakdown_by].unique()
        category_emissions = {cat: np.zeros(years) for cat in categories}
        if show_lifetime_uncertainty:
            category_min_emissions = {cat: np.zeros(years) for cat in categories}
            category_max_emissions = {cat: np.zeros(years) for cat in categories}
    else:
        categories = None
        category_emissions = None

    for unique_id in CGP_df['unique_ID'].values:
        # Get plant data
        plant_data = CGP_df.loc[CGP_df['unique_ID'] == unique_id]
        
        # Get annual CO2 emissions (in GtCO2)
        annual_co2 = float(plant_data['ANNUALCO2']) / 1e9  # Convert to GtCO2 and ensure scalar
        
        # Add to total emissions
        yr_offset = int(plant_data['Year_of_Commission'].iloc[0] - start_year)  # Ensure integer offset
        if yr_offset >= 0:  # Only include plants commissioned after start_year
            # Calculate operating years
            operating_years = int(min(40, end_year - plant_data['Year_of_Commission'].iloc[0]))  # Ensure integer
            if operating_years > 0:
                # Add emissions for each operating year
                for yr in range(operating_years):
                    if yr_offset + yr < years:
                        total_emissions[yr_offset + yr] += annual_co2
                        
                        # Add to category emissions if breaking down
                        if breakdown_by:
                            category = plant_data[breakdown_by].iloc[0]
                            category_emissions[category][yr_offset + yr] += annual_co2
                
                # Calculate lifetime uncertainty bounds if requested
                if show_lifetime_uncertainty:
                    # Minimum lifetime (shorter operation)
                    min_operating_years = int(max(0, operating_years - lifetime_uncertainty))
                    for yr in range(min_operating_years):
                        if yr_offset + yr < years:
                            min_emissions[yr_offset + yr] += annual_co2
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                category_min_emissions[category][yr_offset + yr] += annual_co2
                    
                    # Maximum lifetime (longer operation)
                    max_operating_years = int(min(operating_years + lifetime_uncertainty, 
                                            end_year - plant_data['Year_of_Commission'].iloc[0]))
                    for yr in range(max_operating_years):
                        if yr_offset + yr < years:
                            max_emissions[yr_offset + yr] += annual_co2
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                category_max_emissions[category][yr_offset + yr] += annual_co2

    # Calculate cumulative emissions
    cumulative_emissions = np.cumsum(total_emissions)
    if show_lifetime_uncertainty:
        min_cumulative = np.cumsum(min_emissions)
        max_cumulative = np.cumsum(max_emissions)
    
    # Calculate temperature response with uncertainty
    co2_temp_response = cumulative_emissions * (tcr / 1000)  # Temperature in °C
    co2_temp_upper = cumulative_emissions * ((tcr + tcr_uncertainty) / 1000)
    co2_temp_lower = cumulative_emissions * ((tcr - tcr_uncertainty) / 1000)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot CO2 emissions
    if breakdown_by and category_emissions:
        # Plot stacked area for each category
        yearly_category_emissions = {cat: np.cumsum(em) for cat, em in category_emissions.items()}
        if show_lifetime_uncertainty:
            # Plot uncertainty bounds
            yearly_category_min = {cat: np.cumsum(em) for cat, em in category_min_emissions.items()}
            yearly_category_max = {cat: np.cumsum(em) for cat, em in category_max_emissions.items()}
            for cat in categories:
                ax1.fill_between(time_array, 
                               yearly_category_min[cat], 
                               yearly_category_max[cat],
                               alpha=0.1, color='gray')
        ax1.stackplot(time_array, 
                     [yearly_category_emissions[cat] for cat in categories],
                     labels=categories)
        ax1.legend(loc='upper left')
    else:
        if show_lifetime_uncertainty:
            # Plot uncertainty bounds
            ax1.fill_between(time_array, min_cumulative, max_cumulative,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax1.plot(time_array, cumulative_emissions, 
                color='tab:blue', linewidth=2, label='Cumulative Emissions')
    
    ax1.set_ylabel('Cumulative CO2 Emissions (GtCO2)')
    ax1.grid(True, alpha=0.3)
    if show_lifetime_uncertainty:
        ax1.legend(loc='upper left')
    
    # Bottom-left: CO2 temperature response with both TCR and lifetime uncertainty
    ax_co2_temp = axs[1, 0]
    
    # First plot the lifetime uncertainty (if enabled)
    if show_lifetime_uncertainty:
        # Calculate temperature response for min/max emissions with central TCR
        min_temp_response = min_cumulative * (tcr / 1000)
        max_temp_response = max_cumulative * (tcr / 1000)
        
        # Plot lifetime uncertainty band
        ax_co2_temp.fill_between(time_array, min_temp_response, max_temp_response,
                        color='gray', alpha=0.2, label='Lifetime Uncertainty')
    
    # Then plot the TCR uncertainty on the central emissions estimate
    ax_co2_temp.fill_between(time_array, co2_temp_lower, co2_temp_upper,
                    color='tab:red', alpha=0.2, label='TCR Uncertainty')
    
    # If both uncertainties are enabled, also show the combined uncertainty
    if show_lifetime_uncertainty:
        # Calculate the full range of uncertainty combining both factors
        min_combined = min_cumulative * ((tcr - tcr_uncertainty) / 1000)  # Min lifetime, min TCR
        max_combined = max_cumulative * ((tcr + tcr_uncertainty) / 1000)  # Max lifetime, max TCR
        
        # Plot outline of the combined uncertainty range
        ax_co2_temp.plot(time_array, min_combined, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_co2_temp.plot(time_array, max_combined, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot the central estimate
    ax_co2_temp.plot(time_array, co2_temp_response,
            color='tab:red', linewidth=2, label='CO2 Temperature Response')
    
    ax_co2_temp.set_xlabel('Year')
    ax_co2_temp.set_ylabel('Temperature Response (°C)')
    ax_co2_temp.set_title('CO2 Temperature Impact', fontweight='bold')
    ax_co2_temp.grid(True, alpha=0.3)
    ax_co2_temp.legend(loc='upper left', fontsize='small')
    
    plt.suptitle(f'CO2 Emissions and Temperature Response (TCR = {tcr}±{tcr_uncertainty}°C/1000GtCO2)')
    plt.tight_layout()
    
    return cumulative_emissions, co2_temp_response

def plot_emissions_and_temperature_with_bc(CGP_df, start_year=2000, end_year=2060, tcr=1.65, tcr_uncertainty=0.4, 
                                         breakdown_by=None, show_lifetime_uncertainty=True, lifetime_uncertainty=5,
                                         figsize=(14, 10)):
    """
    Plot cumulative CO2 emissions and temperature response over time, including BC-related temperature responses.
    Uses a 2x2 layout with CO2 on the left side and BC on the right side.
    
    Parameters:
    -----------
    CGP_df : pandas.DataFrame
        DataFrame containing power plant data
    start_year : int
        Starting year for the analysis (default: 2000)
    end_year : int
        Ending year for the analysis (default: 2060)
    tcr : float
        Transient Climate Response factor in °C per 1000 GtCO2 (default: 1.65)
    tcr_uncertainty : float
        Uncertainty in TCR value (default: 0.4)
    breakdown_by : str, optional
        Column to break down emissions by (e.g., 'COUNTRY' or 'plant_type')
    show_lifetime_uncertainty : bool
        Whether to show shading for lifetime uncertainty (default: True)
    lifetime_uncertainty : int
        Number of years of uncertainty in plant lifetime (default: 5)
    figsize : tuple
        Figure size as (width, height)
    """
    # Set up time array
    years = end_year - start_year
    time_array = np.arange(start_year, end_year)
    
    # Initialize arrays for emissions and temperature responses
    total_emissions = np.zeros(years)
    total_bc_emissions = np.zeros(years)  # BC emissions in g/yr
    total_dt_drf = np.zeros(years)
    total_dt_snowrf = np.zeros(years)
    
    if show_lifetime_uncertainty:
        min_emissions = np.zeros(years)
        max_emissions = np.zeros(years)
        min_bc_emissions = np.zeros(years)
        max_bc_emissions = np.zeros(years)
        min_dt_drf = np.zeros(years)
        max_dt_drf = np.zeros(years)
        min_dt_snowrf = np.zeros(years)
        max_dt_snowrf = np.zeros(years)
    
    # If breaking down emissions, create separate arrays for each category
    if breakdown_by:
        categories = CGP_df[breakdown_by].unique()
        category_emissions = {cat: np.zeros(years) for cat in categories}
        category_bc_emissions = {cat: np.zeros(years) for cat in categories}
        category_dt_drf = {cat: np.zeros(years) for cat in categories}
        category_dt_snowrf = {cat: np.zeros(years) for cat in categories}
        
        if show_lifetime_uncertainty:
            category_min_emissions = {cat: np.zeros(years) for cat in categories}
            category_max_emissions = {cat: np.zeros(years) for cat in categories}
            category_min_bc_emissions = {cat: np.zeros(years) for cat in categories}
            category_max_bc_emissions = {cat: np.zeros(years) for cat in categories}
            category_min_dt_drf = {cat: np.zeros(years) for cat in categories}
            category_max_dt_drf = {cat: np.zeros(years) for cat in categories}
            category_min_dt_snowrf = {cat: np.zeros(years) for cat in categories}
            category_max_dt_snowrf = {cat: np.zeros(years) for cat in categories}
    else:
        categories = None
        category_emissions = None
        category_bc_emissions = None
        category_dt_drf = None
        category_dt_snowrf = None

    for unique_id in CGP_df['unique_ID'].values:
        # Get plant data
        plant_data = CGP_df.loc[CGP_df['unique_ID'] == unique_id]
        
        # Get annual values
        annual_co2 = float(plant_data['ANNUALCO2']) / 1e9  # Convert to GtCO2
        annual_bc = float(plant_data['BC_(g/yr)'])  # BC emissions in g/yr
        annual_dt_drf = float(plant_data['dt_drf'])
        annual_dt_snowrf = float(plant_data['dt_snowrf'])
        
        # Add to total emissions
        yr_offset = int(plant_data['Year_of_Commission'].iloc[0] - start_year)
        if yr_offset >= 0:
            operating_years = int(min(40, end_year - plant_data['Year_of_Commission'].iloc[0]))
            if operating_years > 0:
                # Add values for each operating year
                for yr in range(operating_years):
                    if yr_offset + yr < years:
                        total_emissions[yr_offset + yr] += annual_co2
                        total_bc_emissions[yr_offset + yr] += annual_bc
                        total_dt_drf[yr_offset + yr] += annual_dt_drf
                        total_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                        
                        if breakdown_by:
                            category = plant_data[breakdown_by].iloc[0]
                            category_emissions[category][yr_offset + yr] += annual_co2
                            category_bc_emissions[category][yr_offset + yr] += annual_bc
                            category_dt_drf[category][yr_offset + yr] += annual_dt_drf
                            category_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                
                if show_lifetime_uncertainty:
                    min_operating_years = int(max(0, operating_years - lifetime_uncertainty))
                    max_operating_years = int(min(operating_years + lifetime_uncertainty, 
                                            end_year - plant_data['Year_of_Commission'].iloc[0]))
                    
                    for yr in range(min_operating_years):
                        if yr_offset + yr < years:
                            min_emissions[yr_offset + yr] += annual_co2
                            min_bc_emissions[yr_offset + yr] += annual_bc
                            min_dt_drf[yr_offset + yr] += annual_dt_drf
                            min_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                category_min_emissions[category][yr_offset + yr] += annual_co2
                                category_min_bc_emissions[category][yr_offset + yr] += annual_bc
                                category_min_dt_drf[category][yr_offset + yr] += annual_dt_drf
                                category_min_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                    
                    for yr in range(max_operating_years):
                        if yr_offset + yr < years:
                            max_emissions[yr_offset + yr] += annual_co2
                            max_bc_emissions[yr_offset + yr] += annual_bc
                            max_dt_drf[yr_offset + yr] += annual_dt_drf
                            max_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                category_max_emissions[category][yr_offset + yr] += annual_co2
                                category_max_bc_emissions[category][yr_offset + yr] += annual_bc
                                category_max_dt_drf[category][yr_offset + yr] += annual_dt_drf
                                category_max_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf

    # Calculate cumulative values
    cumulative_emissions = np.cumsum(total_emissions)
    cumulative_bc_emissions = np.cumsum(total_bc_emissions)
    cumulative_dt_drf = np.cumsum(total_dt_drf)
    cumulative_dt_snowrf = np.cumsum(total_dt_snowrf)
    
    if show_lifetime_uncertainty:
        min_cumulative = np.cumsum(min_emissions)
        max_cumulative = np.cumsum(max_emissions)
        min_cumulative_bc = np.cumsum(min_bc_emissions)
        max_cumulative_bc = np.cumsum(max_bc_emissions)
        min_cumulative_dt_drf = np.cumsum(min_dt_drf)
        max_cumulative_dt_drf = np.cumsum(max_dt_drf)
        min_cumulative_dt_snowrf = np.cumsum(min_dt_snowrf)
        max_cumulative_dt_snowrf = np.cumsum(max_dt_snowrf)
    
    # Calculate temperature responses
    co2_temp_response = cumulative_emissions * (tcr / 1000)
    co2_temp_upper = cumulative_emissions * ((tcr + tcr_uncertainty) / 1000)
    co2_temp_lower = cumulative_emissions * ((tcr - tcr_uncertainty) / 1000)
    
    # Calculate total BC temperature response
    total_bc_temp_response = cumulative_dt_drf + cumulative_dt_snowrf
    if show_lifetime_uncertainty:
        total_bc_temp_upper = max_cumulative_dt_drf + max_cumulative_dt_snowrf
        total_bc_temp_lower = min_cumulative_dt_drf + min_cumulative_dt_snowrf
    
    # Total temperature response (CO2 + BC)
    total_temp_response = co2_temp_response + total_bc_temp_response
    if show_lifetime_uncertainty:
        total_temp_upper = co2_temp_upper + total_bc_temp_upper
        total_temp_lower = co2_temp_lower + total_bc_temp_lower

    # Create the 2x2 plot layout
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # ----- Left side (CO2) -----
    # Top-left: CO2 emissions
    ax_co2_emissions = axs[0, 0]
    if breakdown_by and category_emissions:
        # Plot stacked area for each category
        yearly_category_emissions = {cat: np.cumsum(em) for cat, em in category_emissions.items()}
        if show_lifetime_uncertainty:
            yearly_category_min = {cat: np.cumsum(em) for cat, em in category_min_emissions.items()}
            yearly_category_max = {cat: np.cumsum(em) for cat, em in category_max_emissions.items()}
            for cat in categories:
                ax_co2_emissions.fill_between(time_array, 
                               yearly_category_min[cat], 
                               yearly_category_max[cat],
                               alpha=0.1, color='gray')
        ax_co2_emissions.stackplot(time_array, 
                     [yearly_category_emissions[cat] for cat in categories],
                     labels=categories)
        ax_co2_emissions.legend(loc='upper left', fontsize='small')
    else:
        if show_lifetime_uncertainty:
            ax_co2_emissions.fill_between(time_array, min_cumulative, max_cumulative,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax_co2_emissions.plot(time_array, cumulative_emissions, 
                color='tab:blue', linewidth=2, label='Cumulative CO2 Emissions')
    
    ax_co2_emissions.set_ylabel('Cumulative CO2 Emissions (GtCO2)')
    ax_co2_emissions.set_title('CO2 Emissions', fontweight='bold')
    ax_co2_emissions.grid(True, alpha=0.3)
    if show_lifetime_uncertainty and not breakdown_by:
        ax_co2_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-left: CO2 temperature response with both TCR and lifetime uncertainty
    ax_co2_temp = axs[1, 0]
    
    # First plot the lifetime uncertainty (if enabled)
    if show_lifetime_uncertainty:
        # Calculate the full range of uncertainty combining both factors
        min_combined = min_cumulative * ((tcr - tcr_uncertainty) / 1000)  # Min lifetime, min TCR
        max_combined = max_cumulative * ((tcr + tcr_uncertainty) / 1000)  # Max lifetime, max TCR
        
        # Plot outline of the combined uncertainty range
        ax_co2_temp.plot(time_array, min_combined, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_co2_temp.plot(time_array, max_combined, color='black', linestyle='--', linewidth=1, alpha=0.5)
    

        # Calculate temperature response for min/max emissions with central TCR
        min_temp_response = min_cumulative * (tcr / 1000)
        max_temp_response = max_cumulative * (tcr / 1000)
        
        # Plot lifetime uncertainty band
        ax_co2_temp.fill_between(time_array, min_temp_response, max_temp_response,
                        color='gray', alpha=0.2, label='Lifetime Uncertainty')

            # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, min_temp_response, min_combined,
                        color='tab:red', alpha=0.2)
            # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, max_temp_response, max_combined,
                        color='tab:red', alpha=0.2, label='TCR Uncertainty')
          
    elif not show_lifetime_uncertainty :
    # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, co2_temp_lower, co2_temp_upper,
                    color='tab:red', alpha=0.2, label='TCR Uncertainty')
    
   
    # Plot the central estimate
    ax_co2_temp.plot(time_array, co2_temp_response,
            color='tab:red', linewidth=2, label='CO2 Temperature Response')
    
    ax_co2_temp.set_xlabel('Year')
    ax_co2_temp.set_ylabel('Temperature Response (°C)')
    ax_co2_temp.set_title('CO2 Temperature Impact', fontweight='bold')
    ax_co2_temp.grid(True, alpha=0.3)
    ax_co2_temp.legend(loc='upper left', fontsize='small')
    
    # ----- Right side (BC) -----
    # Top-right: BC emissions
    ax_bc_emissions = axs[0, 1]
    
    if breakdown_by and category_bc_emissions:
        # Plot stacked area for each category
        yearly_category_bc_emissions = {cat: np.cumsum(em) for cat, em in category_bc_emissions.items()}
        if show_lifetime_uncertainty:
            yearly_category_min_bc = {cat: np.cumsum(em) for cat, em in category_min_bc_emissions.items()}
            yearly_category_max_bc = {cat: np.cumsum(em) for cat, em in category_max_bc_emissions.items()}
            for cat in categories:
                ax_bc_emissions.fill_between(time_array, 
                               yearly_category_min_bc[cat], 
                               yearly_category_max_bc[cat],
                               alpha=0.1, color='gray')
        ax_bc_emissions.stackplot(time_array, 
                     [yearly_category_bc_emissions[cat] for cat in categories],
                     labels=categories)
        ax_bc_emissions.legend(loc='upper left', fontsize='small')
    else:
        if show_lifetime_uncertainty:
            ax_bc_emissions.fill_between(time_array, min_cumulative_bc, max_cumulative_bc,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax_bc_emissions.plot(time_array, cumulative_bc_emissions, 
                color='tab:green', linewidth=2, label='Cumulative BC Emissions')
    
    # Format y-axis for better readability (e.g., using scientific notation or appropriate units)
    ax_bc_emissions.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    ax_bc_emissions.set_ylabel('Cumulative BC Emissions (g)')
    ax_bc_emissions.set_title('BC Emissions', fontweight='bold')
    ax_bc_emissions.grid(True, alpha=0.3)
    if show_lifetime_uncertainty and not breakdown_by:
        ax_bc_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-right: BC temperature responses with both lifetime uncertainty and proper combined uncertainty
    ax_bc_temp = axs[1, 1]
    
    # First plot the individual components for clarity
    ax_bc_temp.plot(time_array, cumulative_dt_drf,
            color='tab:green', linewidth=2, label='BC DRF')
    ax_bc_temp.plot(time_array, cumulative_dt_snowrf,
            color='tab:purple', linewidth=2, label='BC Snow RF')
    
    # Show lifetime uncertainty on the total BC temperature
    if show_lifetime_uncertainty:
        # Calculate temperature response for min/max BC emissions with central values
        min_bc_temp_response = min_cumulative_dt_drf + min_cumulative_dt_snowrf
        max_bc_temp_response = max_cumulative_dt_drf + max_cumulative_dt_snowrf
        
        # Plot lifetime uncertainty band
        ax_bc_temp.fill_between(time_array, min_bc_temp_response, max_bc_temp_response,
                        color='gray', alpha=0.2, label='Lifetime Uncertainty')
    
    # Plot the total BC temperature response
    ax_bc_temp.plot(time_array, total_bc_temp_response,
            color='tab:blue', linewidth=2, linestyle='--', label='Total BC')
    
    # If both uncertainties are enabled, also show the combined uncertainty outline
    if show_lifetime_uncertainty:
        # Calculate the full range of uncertainty combining factors
        min_combined_bc = total_bc_temp_lower  # Already calculated in main code
        max_combined_bc = total_bc_temp_upper  # Already calculated in main code
        
        # Plot outline of the combined uncertainty range
        ax_bc_temp.plot(time_array, min_combined_bc, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_bc_temp.plot(time_array, max_combined_bc, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax_bc_temp.set_xlabel('Year')
    ax_bc_temp.set_ylabel('Cumulative Temperature Response (°C)')
    ax_bc_temp.set_title('BC Temperature Impact', fontweight='bold')
    ax_bc_temp.grid(True, alpha=0.3)
    ax_bc_temp.legend(loc='upper left', fontsize='small')
    
    # Add a figure title
    plt.suptitle(f'Climate Impacts of Coal Power Plants (TCR = {tcr}±{tcr_uncertainty}°C/1000GtCO2)', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    return (cumulative_emissions, co2_temp_response, 
            cumulative_bc_emissions, cumulative_dt_drf, cumulative_dt_snowrf, 
            total_bc_temp_response, total_temp_response)