import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from matplotlib.lines import Line2D

import cmocean
import cmocean.cm as cmo
import xarray as xr
import numpy as np
import pandas as pd

import seaborn as sns


def prepare_concentration_data(full_ds, gdf, map_locations, shutdown_years, 
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
    reg_data_by_location = {}
    for loc in map_locations:
        conc_by_location[loc] = {}
        for yr in shutdown_years:
            # Get concentration data
            if loc == 'all':
                data = full_ds[variable].sel(scenario_year=yr).sum(dim='unique_ID')
            else:
                data = (full_ds
                       .where(full_ds.country_emitting == loc, drop=True)
                       .sel(scenario_year=yr)[variable]
                       .sum(dim='unique_ID'))

            if normalize:
                if loc == 'all':
                    norm_factor = (full_ds['BC_(g/yr)']
                                    .sum(dim = 'unique_ID').sel(scenario_year = yr))
                else:
                    norm_factor = (full_ds.where(full_ds.country_emitting == loc, drop = True)['BC_(g/yr)']
                                    .sum(dim = 'unique_ID').sel(scenario_year = yr))
                #print(data)
                #print(norm_factor)
                data = data / norm_factor.values   

            if hasattr(data, 'ndim') and data.ndim == 0:
                data_prep = gdf.copy()
                data_prep[variable] = data.values
                data = data_prep.drop(columns = ['geometry'])
            else:
                data = data.to_dataframe()

            # Create GeoDataFrame
            gdf_data = pd.merge(
                gdf, 
                data, 
                on='country_impacted'
            )
            
            # # Normalize if requested
            # if normalize:
            #     if loc == 'all':
            #         total_emissions = emissions_df.groupby('COUNTRY').sum()['BC_(g/yr)'].sum() 
            #     else:
            #         total_emissions = emissions_df.groupby('COUNTRY').sum()['BC_(g/yr)'][loc]
                
            #     gdf_data[variable] = gdf_data[variable] / total_emissions
            
            conc_by_location[loc][yr] = gdf_data
    
    return conc_by_location


def plot_concentration_maps(conc_by_location, map_locations, shutdown_years,
                            impacted_countries,
                            country_df,
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
                cmap=cmo.matter,
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
                country_df[country_df['country'] == location.title()].boundary.plot(
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
                axes[0, col_idx].set_title(f'Emissions from {location.title()}', 
                                         fontsize=18)
            if col_idx == 0:
                axes[row_idx, 0].set_ylabel(f'Shutdown in {year}', 
                                          fontsize=18)
            
            # Set map extent
            ax.set_xlim(75, 155)
            ax.set_ylim(-35, 45)
            
            # Add subplot lettering
            subplot_letter = chr(ord('a') + row_idx * n_locations + col_idx)
            ax.text(0.04, 0.04, f'{subplot_letter})', transform=ax.transAxes, 
                    fontsize=16)
            
    # Add colorbar
    cb_ax = fig.add_axes([0.2, -0.05, 0.6, 0.02])
    mpl.colorbar.ColorbarBase(
        ax=cb_ax, 
        cmap=cmo.matter, 
        norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        orientation='horizontal', 
        label='$ng/m^3/g/yr$'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, axes

def plot_single_year_concentration_maps(conc_by_location, norm_conc_by_location, 
                                       map_locations, year,
                                       impacted_countries, country_df,
                                       variable='BC_pop_weight_mean_conc',
                                       vmin_regular=1e-6, vmax_regular=1e-1,
                                       vmin_norm=None, vmax_norm=None,
                                       figsize=(20, 12),
                                       save_path=None):
    """
    Create maps of BC concentrations for a single year showing both regular and normalized data.
    
    Parameters:
    -----------
    conc_by_location : dict
        Nested dictionary of prepared regular data from prepare_concentration_data()
    norm_conc_by_location : dict
        Nested dictionary of prepared normalized data from prepare_concentration_data()
    map_locations : list
        List of locations to plot
    year : int
        Year to plot
    impacted_countries : list
        List of countries to highlight as impacted
    country_df : GeoDataFrame
        GeoDataFrame containing country boundaries
    variable : str, optional
        Variable being plotted (for labels)
    vmin_regular, vmax_regular : float, optional
        Min and max values for color normalization of regular data
    vmin_norm, vmax_norm : float, optional
        Min and max values for color normalization of normalized data
        If None, will be automatically determined from the data
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
    n_locations = len(map_locations)
    fig, axes = plt.subplots(2, n_locations, 
                           figsize=figsize, 
                           sharex=True, sharey=True, 
                           constrained_layout=True)
    
    # Define impacted countries
    all_countries = list(set(impacted_countries + ['Malaysia', 'Cambodia', 'Indonesia', 'Vietnam']))
    
    # If vmin_norm or vmax_norm not provided, calculate from data
    if vmin_norm is None or vmax_norm is None:
        # Find min and max values across all locations
        all_values = []
        for location in map_locations:
            values = norm_conc_by_location[location][year][variable].values
            all_values.extend(values[~np.isnan(values)])
        
        if vmin_norm is None:
            vmin_norm = np.percentile(all_values, 5)  # 5th percentile to avoid outliers
        if vmax_norm is None:
            vmax_norm = np.percentile(all_values, 95)  # 95th percentile to avoid outliers
    
    # Create regular data plots (top row)
    for col_idx, location in enumerate(map_locations):
        ax = axes[0, col_idx]
        
        # Plot concentrations
        cmap = cmo.matter
        conc_by_location[location][year].plot(
            ax=ax,
            column=variable,
            cmap=cmap,
            norm=matplotlib.colors.LogNorm(vmin=vmin_regular, vmax=vmax_regular)
        )
        
        # Plot country boundaries
        country_df[country_df['country'].isin(all_countries)].boundary.plot(
            ax=ax, 
            color='k', 
            linewidth=0.5
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Add hatching for emitting countries
        if location in ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']:
            country_df[country_df['country'] == location.title()].boundary.plot(
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
        ax.set_title(f'{location.title()}', 
                   fontsize=16)
        
        # Set map extent
        ax.set_xlim(75, 155)
        ax.set_ylim(-35, 45)
    
        # Add subplot lettering
        subplot_letter = chr(ord('a') + col_idx)
        ax.text(0.04, 0.04, f'{subplot_letter})', transform=ax.transAxes, 
                fontsize=16)

    # Create normalized data plots (bottom row)
    for col_idx, location in enumerate(map_locations):
        ax = axes[1, col_idx]
        
        # Plot normalized concentrations
        norm_conc_by_location[location][year].plot(
            ax=ax,
            column=variable,
            cmap=cmap,
            norm=matplotlib.colors.LogNorm(vmin=vmin_norm, vmax=vmax_norm)
        )
        
        # Plot country boundaries
        country_df[country_df['country'].isin(all_countries)].boundary.plot(
            ax=ax, 
            color='k', 
            linewidth=0.5
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Add hatching for emitting countries
        if location in ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']:
            country_df[country_df['country'] == location.title()].boundary.plot(
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
        ax.set_title(f'{location.title()}', 
                   fontsize=16)
        
        # Set map extent
        ax.set_xlim(75, 155)
        ax.set_ylim(-35, 45)

        # Add subplot lettering
        subplot_letter = chr(ord('f') + col_idx)
        ax.text(0.04, 0.04, f'{subplot_letter})', transform=ax.transAxes, 
                fontsize=16)

    
    # Add row labels
    fig.text(-0.03, 0.75, f'Cumulative Concentration \n(ng/m³)', 
            fontsize=16, va='center', ha='center', rotation=90)
    fig.text(-0.03, 0.25, f'Cumulative Concentration normalized \nby Total Emissions (ng/m³/g/yr)', 
            fontsize=16, va='center', ha='center', rotation=90)
    
    # Add colorbars
    # Regular data colorbar
    cb_ax1 = fig.add_axes([0.25, 0.50, 0.5, 0.02])
    cb1 = mpl.colorbar.ColorbarBase(
        ax=cb_ax1, 
        cmap=cmap, 
        norm=matplotlib.colors.LogNorm(vmin=vmin_regular, vmax=vmax_regular),
        orientation='horizontal', 
    )
    cb1.ax.tick_params(labelsize=14)  
    cb1.set_label('Concentration (ng/m³)', fontsize=16) 

    # Normalized data colorbar
    cb_ax2 = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    cb2 = mpl.colorbar.ColorbarBase(
        ax=cb_ax2, 
        cmap=cmap, 
        norm=matplotlib.colors.LogNorm(vmin=vmin_norm, vmax=vmax_norm),
        orientation='horizontal', 
    )
    cb2.ax.tick_params(labelsize=14)  
    cb2.set_label('Normalized concentration (ng/m³/g/yr)', fontsize=16) 
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig, axes
def prep_matrix_from_scenario_ds(scenario_ds, year, variable='BC_pop_weight_mean_conc', normalized = False):
    """
    Create a heatmap matrix directly from scenario_ds by iterating through each emitting country
    """
    # Select data for the specified year
    year_data = scenario_ds.sel(scenario_year=year)
    
    # Get all unique emitting countries
    emitting_countries = year_data['country_emitting'].values
    unique_emitting_countries = list(set(emitting_countries))
    
    # Get all impacted countries for the matrix index
    impacted_countries = year_data['country_impacted'].values
    
    # Initialize empty matrix DataFrame
    matrix_df = pd.DataFrame(
        index=impacted_countries,
        columns=unique_emitting_countries,
        dtype=float
    ).fillna(0.0)
    
    if normalized:
        for emitting_country in unique_emitting_countries:
                # Get data for this emitting country and sum across all plants
            matrix_df[emitting_country] = (year_data.where(
                            year_data['country_emitting'] == emitting_country, 
                            drop=True
                        ).sum(dim='unique_ID')[variable]/
                                    year_data.where(
                                    year_data['country_emitting'] ==  emitting_country, 
                                    drop=True
                                ).sum(dim='unique_ID')['BC_(g/yr)']).to_pandas()
    # Iterate through each emitting country
    elif not normalized:
        for emitting_country in unique_emitting_countries:
            # Get data for this emitting country and sum across all plants
            matrix_df[emitting_country] = year_data.where(
                year_data['country_emitting'] == emitting_country, 
                drop=True
            ).sum(dim='unique_ID')[variable].to_pandas()
        
    return matrix_df

def plot_single_matrix(matrix_data, ax, variable, title_suffix=''):
    """
    Plot a single heatmap matrix using seaborn
    """
    # Format country names
    matrix_formatted = matrix_data.copy()
    matrix_formatted.index = [country.title() for country in matrix_data.index]
    matrix_formatted.columns = [country.title() for country in matrix_data.columns]
    
      
    # Sort the index (y-axis, impacted countries) alphabetically
    matrix_formatted = matrix_formatted.sort_index(ascending=True)
    
    # Set minimum value for log scale
    vmin = np.min(matrix_data.values[matrix_data.values > 0])
    vmax = np.max(matrix_data.values)
    
    # Create heatmap
    sns.heatmap(matrix_formatted, 
                ax=ax,
                cmap=cmocean.cm.matter,
                norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                cbar_kws={'orientation': 'horizontal', 'pad': 0.05, 'shrink': 0.8})
    
    # Move x-axis labels to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(),  ha='center')
    
    # Custom colorbar label
    cbar = ax.collections[0].colorbar
    
    # Custom colorbar label based on variable and normalization
    if 'pop_weight' in variable:
        base_label = 'Population-weighted \nBlack Carbon Concentration'
        if 'Normalized' in title_suffix:
            colorbar_label = f'Normalized {base_label} (ng/m³/g/yr)'
        else:
            colorbar_label = f'{base_label} (ng/m³)'
    elif 'surface' in variable:
        base_label = 'Surface \n Black Carbon Concentration'
        if 'Normalized' in title_suffix:
            colorbar_label = f'Normalized {base_label} (ng/m³/g/yr)'
        else:
            colorbar_label = f'{base_label} (ng/m³)'
    elif 'column' in variable:
        base_label = 'Column \n Black Carbon Concentration'
        if 'Normalized' in title_suffix:
            colorbar_label = f'Normalized {base_label} (ng/m³/g/yr)'
        else:
            colorbar_label = f'{base_label} (ng/m³)'
    else:
        colorbar_label = f'{variable}'
    
    cbar.set_label(colorbar_label, fontsize=14)

def create_sankey_from_map_data(regular_data, norm_data=None, temp_data=None, temp_norm_data=None,
                              year=2040, location='all', 
                              variable='BC_pop_weight_mean_conc', temp_variable='temperature_impact_aamaas_10',
                              top_n=10, allow_loops=False):
    """
    Create a 2x2 grid of Sankey diagrams showing concentration and temperature data,
    with both regular and normalized versions of each.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    
    # Define a colormap for source countries
    source_colors = {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }
    
    # Check if the location is valid
    if location not in regular_data:
        raise ValueError(f"Location '{location}' not found in data. Available: {list(regular_data.keys())}")
    
    # Determine which diagrams to show
    has_norm_data = norm_data is not None and location in norm_data
    has_temp_data = temp_data is not None and location in temp_data
    has_temp_norm_data = temp_norm_data is not None and location in temp_norm_data
    
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "a) Concentration (ng/m³·person)", 
            "b) Concentration Normalized by Emissions (ng/m³·person)/(g/yr)" if has_norm_data else None,
            "c) Temperature Impact (°C)" if has_temp_data else None,
            "d) Temperature Impact Normalized by Emissions (°C/g/yr)" if has_temp_norm_data else None
        ),
        specs=[
            [{"type": "sankey"}, {"type": "sankey"}],
            [{"type": "sankey"}, {"type": "sankey"}]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    # Process each dataset and create Sankey diagrams
    datasets = [
        (regular_data, variable, 1, 1),  # regular concentration, top-left
        (norm_data, variable, 1, 2),     # normalized concentration, top-right
        (temp_data, temp_variable, 2, 1), # temperature data, bottom-left
        (temp_norm_data, temp_variable, 2, 2)  # normalized temperature data, bottom-right
    ]
    
    # Dictionary to store all the Sankey data
    sankey_data = {}

    # Calculate top_receptors combining BC and temperature data
    top_receptors = set()
    
    # Get top receptors from regular BC data
    try:
        bc_data_df = regular_data[location][year].copy()
        if isinstance(bc_data_df, pd.DataFrame):
            bc_grouped_df = bc_data_df.groupby(bc_data_df.index)[variable].sum().reset_index()
            bc_grouped_df.columns = ['country_impacted', variable]
        else:
            bc_grouped_df = bc_data_df.reset_index()
            bc_grouped_df = bc_grouped_df.groupby('country_impacted')[variable].sum().reset_index()
        
        bc_grouped_df = bc_grouped_df.dropna(subset=[variable])
        if not bc_grouped_df.empty:
            top_receptors_bc = bc_grouped_df.sort_values(variable, ascending=False).head(top_n)
            top_receptors.update(top_receptors_bc['country_impacted'].tolist())
    except (KeyError, Exception) as e:
        print(f"Could not get BC top receptors: {e}")
    
    # Get top receptors from temperature data
    if temp_data is not None and location in temp_data:
        try:
            temp_data_df = temp_data[location][year].copy()
            if isinstance(temp_data_df, pd.DataFrame):
                temp_grouped_df = temp_data_df.groupby(temp_data_df.index)[temp_variable].sum().reset_index()
                temp_grouped_df.columns = ['country_impacted', temp_variable]
            else:
                temp_grouped_df = temp_data_df.reset_index()
                temp_grouped_df = temp_grouped_df.groupby('country_impacted')[temp_variable].sum().reset_index()
            
            temp_grouped_df = temp_grouped_df.dropna(subset=[temp_variable])
            if not temp_grouped_df.empty:
                top_receptors_temp = temp_grouped_df.sort_values(temp_variable, ascending=False).head(top_n)
                top_receptors.update(top_receptors_temp['country_impacted'].tolist())
        except (KeyError, Exception) as e:
            print(f"Could not get temperature top receptors: {e}")
    
    print(f"Combined top receptors: {top_receptors}")
    top_receptors = list(top_receptors)
    for data_idx, (data_dict, viz_variable, idx_row, idx_col) in enumerate(datasets):
        # Skip if data is not provided or location not in data
        if data_dict is None or location not in data_dict:
            continue
        
        # Create a key for this dataset
        if data_dict is regular_data:
            if viz_variable == variable:
                data_key = 'regular_concentration'
            else:
                data_key = 'regular_temperature'
        elif data_dict is norm_data:
            data_key = 'normalized_concentration'
        elif data_dict is temp_data:
            data_key = 'temperature'
        elif data_dict is temp_norm_data:
            data_key = 'normalized_temperature'
        else:
            data_key = f'dataset_{data_idx}'
        
        try:
            # Get data for the specified year and location
            data_df = data_dict[location][year].copy()
            
            # Group by country_impacted and sum the values
            if isinstance(data_df, pd.DataFrame):
                grouped_df = data_df.groupby(data_df.index)[viz_variable].sum().reset_index()
                grouped_df.columns = ['country_impacted', viz_variable]
            else:
                # If it's already in the right format, just ensure we have the right columns
                grouped_df = data_df.reset_index()
                grouped_df = grouped_df.groupby('country_impacted')[viz_variable].sum().reset_index()
        except KeyError:
            print(f"Year {year} not found in data for row {idx_row}, col {idx_col}. Available: {list(data_dict[location].keys())}")
            continue
        
        # Remove NaN values
        grouped_df = grouped_df.dropna(subset=[viz_variable])
        
        if grouped_df.empty:
            print(f"No data for {location} in {year} for variable {viz_variable}")
            continue
        

        # For 'all', we need to split by source country
        if location == 'all':

            # Get source countries from the map_locations list
            source_countries = ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']
            
            # Create separate DataFrames for each source country
            country_dfs = {}
            for country in source_countries:
                if country in data_dict:
                    country_data = data_dict[country][year].copy()
                   
                    # Group by country_impacted and sum
                    country_grouped = country_data.groupby(country_data.index)[viz_variable].mean().reset_index()
                    country_grouped.columns = ['country_impacted', viz_variable]
                
                    # Filter to include top receptor countries and source countries if needed
                    target_list = top_receptors
                    if not allow_loops:
                        # Add source countries as potential targets
                        target_list += [c.title() for c in source_countries]
                    country_grouped = country_grouped[country_grouped['country_impacted'].isin(target_list)]
                    
                    if not country_grouped.empty:
                        country_dfs[country] = country_grouped
                    
            # Prepare data for the Sankey diagram
            all_sources = []
            all_targets = []
            all_values = []
            all_labels = []
            all_colors = []
            
            # Create nodes list
            source_nodes = [country.title() for country in country_dfs.keys()]
            
            # Create target nodes list (including source countries that appear as targets)
            target_nodes = []
            for _, df in country_dfs.items():
                for receptor in df['country_impacted'].unique():
                    if receptor not in target_nodes:
                        print(receptor)
                        target_nodes.append(receptor)
            
            # Create a mapping for nodes that would cause self-loops
            # For source countries that are also targets, create a separate "copy" node
            dual_role_nodes = {}
            for country in source_countries:
                cap_country = country.title()
                if cap_country in target_nodes:
                    # Always create target-specific labels since we don't allow loops
                    dual_role_nodes[cap_country] = f"{cap_country} (recipient)"
            
            # Add regular target nodes
            all_nodes = source_nodes.copy()
            
            # Add target nodes (using the dual role mapping where needed)
            for node in target_nodes:
                if node in [c.title() for c in source_nodes]:
                    # This is a source country that's also a target - use the mapped version
                    all_nodes.append(dual_role_nodes.get(node, node))
                elif node not in all_nodes:
                    all_nodes.append(node)
            
            # Create node index mapping
            node_indices = {}
            for i, node in enumerate(all_nodes):
                if '(recipient)' in node:
                    # Map both the display name and the internal name
                    original_name = node.split(' (recipient)')[0]
                    node_indices[original_name] = i
                    node_indices[node] = i
                elif node.upper() in source_countries:
                    node_indices[node.upper()] = i
                    node_indices[node] = i  # Also map capitalized version
                else:
                    node_indices[node] = i
            
            # Calculate total values for each node (incoming + outgoing)
            node_values = {node: 0.0 for node in all_nodes}
            
            # First pass to calculate total values
            for source_country, df in country_dfs.items():
                # Sum outgoing values for source countries
                source_total = df[viz_variable].sum()
                source_title = source_country.title()
                node_values[source_title] += source_total
                
                # Sum incoming values for target countries
                for _, row in df.iterrows():
                    receptor = row['country_impacted']
                    value = row[viz_variable]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0]
                        
                    if np.isnan(value) or value <= 0:
                        continue
                    
                    # Use the dual role node for self-loops
                    if receptor.upper() == source_country or receptor == source_title:
                        node_key = dual_role_nodes.get(receptor, receptor)
                    else:
                        node_key = receptor
                    
                    node_values[node_key] += value
            
            # Create links
            for source_country, df in country_dfs.items():
                source_idx = node_indices[source_country]
                for _, row in df.iterrows():
                    receptor = row['country_impacted']
                    value = row[viz_variable]
                    if isinstance(value, pd.Series):
                        value = value.iloc[0]
                        
                    if np.isnan(value) or value <= 0:
                        continue
                    
                    # Check if this would be a self-loop
                    source_title = source_country.title()
                    if receptor.upper() == source_country or receptor == source_title:
                        # Always use the dual role node (never allow loops)
                        target_idx = node_indices[dual_role_nodes.get(receptor, receptor)]
                    else:
                        # Regular target
                        target_idx = node_indices[receptor]
                    
                    # Double check to ensure no self-loops
                    if source_idx == target_idx:
                        continue
                        
                    all_sources.append(source_idx)
                    all_targets.append(target_idx)
                    all_values.append(value)
                    
                    # Format the label 
                    all_labels.append(f"{source_title} → {receptor}: {value:.2e}")
                    
                    # Add color based on source country (transparent version)
                    source_color = source_colors[source_country]
                    # Convert to rgba with transparency
                    rgba_color = f"rgba{tuple(int(source_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.5,)}"
                    all_colors.append(rgba_color)
            
            # Create node colors and labels with values included
            node_colors = []
            node_labels = []
            
            # Format node labels to include country names and values
            for node in all_nodes:
                # Get the display name and value
                if '(recipient)' in node:
                    display_name = node.split(' (recipient)')[0]
                    node_value = node_values[node]
                    node_labels.append(f"{display_name}") #<br>{node_value:.2e}
                else:
                    node_value = node_values[node]
                    node_labels.append(f"{node}") #<br>{node_value:.2e}
                
                # Set the color
                if node.upper() in source_colors or node.split(' ')[0].upper() in source_colors:
                    country = node.upper() if node.upper() in source_colors else node.split(' ')[0].upper()
                    node_colors.append(source_colors[country])
                else:
                    node_colors.append("gray")
            
            # Store the data for this Sankey diagram
            sankey_data[data_key] = {
                'nodes': {
                    'labels': node_labels,
                    'colors': node_colors,
                    'values': node_values,
                    'all_nodes': all_nodes
                },
                'links': {
                    'source': all_sources,
                    'target': all_targets,
                    'value': all_values,
                    'labels': all_labels,
                    'colors': all_colors
                },
                'raw_data': {
                    'country_dfs': country_dfs,
                    'top_receptors': top_receptors,
                    'source_countries': source_countries
                },
                'metadata': {
                    'year': year,
                    'location': location,
                    'variable': viz_variable,
                    'top_n': top_n
                }
            }
                    
            # Create Sankey diagram with labels included
            sankey = go.Sankey(
                arrangement = 'snap',
                node = dict(
                    pad = 15,
                    thickness = 25,
                    line = dict(color = "black", width = 0.5),
                    label = node_labels,  # Now use the formatted labels
                    color = node_colors,
                    customdata = node_labels,  # Include labels in hover data too
                    hovertemplate = '%{customdata}'
                ),
                link = dict(
                    source = all_sources,
                    target = all_targets,
                    value = all_values,
                    label = all_labels,
                    color = all_colors
                )
            )
            
            # Add the sankey diagram to the appropriate subplot
            fig.add_trace(
                sankey,
                row=idx_row, 
                col=idx_col
            )

    # Update layout
    fig.update_layout(
        font_size=12,
        height=800,  # Increased height for 2x2 layout
        width=1200,
        margin=dict(l=50, r=50, b=25, t=50),
    )
    
    return fig, sankey_data


# def create_multi_source_sankey(regular_data, year=2040, variable='BC_pop_weight_mean_conc', top_n=10):
#     """
#     Create four side-by-side Sankey diagrams, one for each source country,
#     showing flows to all recipient countries except itself.
    
#     Parameters:
#     -----------
#     regular_data : dict
#         Output from prepare_concentration_data function for regular data
#     year : int
#         Year to visualize
#     variable : str
#         Variable to visualize (e.g., 'BC_pop_weight_mean_conc')
#     top_n : int
#         Number of top receptor countries to include for each source
    
#     Returns:
#     --------
#     plotly.graph_objects.Figure
#         Figure containing four Sankey diagrams
#     """
#     import plotly.graph_objects as go
#     from plotly.subplots import make_subplots
#     import numpy as np
    
#     # Define a colormap for source countries
#     source_colors = {
#         'MALAYSIA': '#0072B2',  
#         'CAMBODIA': '#009E73',  
#         'INDONESIA': '#D55E00', 
#         'VIETNAM': '#CC79A7'    
#     }
    
#     # Source countries to create diagrams for
#     source_countries = ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']
    
#     # Create subplots - one for each source country
#     fig = make_subplots(
#         rows=1, cols=4,
#         #subplot_titles=[f"From {country.title()}" for country in source_countries],
#         specs=[[{"type": "sankey"} for _ in range(4)]],
#         horizontal_spacing=0.4
#     )
    
#     # Process each source country
#     for col_idx, source_country in enumerate(source_countries):
#         # Check if the source country is in the data
#         if source_country not in regular_data:
#             continue
            
#         # Get data for the specified year and source country
#         try:
#             data_df = regular_data[source_country][year].copy()
            
#             # Group by country_impacted and sum the values
#             if isinstance(data_df, pd.DataFrame):
#                 grouped_df = data_df.groupby(data_df.index)[variable].sum().reset_index()
#                 grouped_df.columns = ['country_impacted', variable]
#             else:
#                 grouped_df = data_df.reset_index()
#                 grouped_df = grouped_df.groupby('country_impacted')[variable].sum().reset_index()
            
#         except KeyError:
#             continue
        
#         # Remove NaN values and the source country itself
#         grouped_df = grouped_df.dropna(subset=[variable])
#         grouped_df = grouped_df[grouped_df['country_impacted'] != source_country.title()]
        
#         if grouped_df.empty:
#             continue
        
#         # Sort by impact and get top receptors
#         top_receptors = grouped_df.sort_values(variable, ascending=False).head(top_n)
        
#         # Prepare data for Sankey diagram
#         # Single source node
#         source_node = source_country.title()
#         target_nodes = top_receptors['country_impacted'].tolist()
        
#         # Create node list
#         all_nodes = [source_node] + target_nodes
        
#         # Create node index mapping
#         node_indices = {node: i for i, node in enumerate(all_nodes)}
        
#         # Create links
#         sources = []
#         targets = []
#         values = []
#         labels = []
#         colors = []
        
#         source_idx = 0  # Source is always the first node
        
#         for _, row in top_receptors.iterrows():
#             receptor = row['country_impacted']
#             value = row[variable]
            
#             if np.isnan(value) or value <= 0:
#                 continue
                
#             target_idx = node_indices[receptor]
            
#             sources.append(source_idx)
#             targets.append(target_idx)
#             values.append(value)
            
#             # Format label
#             labels.append(f"{source_node} → {receptor}: {value:.2e}")
            
#             # Add color based on source country
#             source_color = source_colors[source_country]
#             # Convert to rgba with transparency
#             rgba_color = f"rgba{tuple(int(source_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}"
#             colors.append(rgba_color)
        
#         # Create node colors and labels
#         node_colors = []
#         node_labels = []
        
#         for node in all_nodes:
#             node_labels.append(node)
            
#             if node == source_node:
#                 node_colors.append(source_colors[source_country])
#             else:
#                 node_colors.append("gray")
        
#         # Create Sankey diagram
#         sankey = go.Sankey(
#             arrangement = 'snap',
#             node = dict(
#                 pad = 10,
#                 thickness = 20,
#                 line = dict(color = "black", width = 0.5),
#                 label = node_labels,
#                 color = node_colors
#             ),
#             link = dict(
#                 source = sources,
#                 target = targets,
#                 value = values,
#                 label = labels,
#                 color = colors
#             )
#         )
        
#         # Add to subplot
#         fig.add_trace(
#             sankey,
#             row=1, 
#             col=col_idx + 1
#         )
    
#     # Update layout
#     fig.update_layout(
#         font_size=10,
#         height=500,
#         width=1200,
#         margin=dict(l=10, r=10, b=10, t=40)
#     )
    
#     return fig
def create_multi_source_sankey(regular_data, norm_data=None, temp_data=None, temp_norm_data=None,
                              year=2040, variable='BC_pop_weight_mean_conc', 
                              temp_variable='dt_sum', top_n=10):
    """
    Create a 2x2 grid of Sankey diagrams:
    - Top row: BC concentration (regular and normalized) showing flows to receptor countries
    - Bottom row: Temperature (regular and normalized) showing flows to single "Global Impact" node
    
    Parameters:
    -----------
    regular_data : dict
        BC concentration data (regular)
    norm_data : dict, optional
        BC concentration data (normalized)
    temp_data : dict, optional
        Temperature data (regular)
    temp_norm_data : dict, optional
        Temperature data (normalized)
    year : int
        Year to visualize
    variable : str
        Variable for BC (e.g., 'BC_pop_weight_mean_conc')
    temp_variable : str
        Variable for temperature (e.g., 'dt_sum')
    top_n : int
        Number of top receptor countries for BC diagrams
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Figure containing 2x2 grid of Sankey diagrams
    sankey_data : dict
        Dictionary containing the data used to create each Sankey diagram
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd
    
    # Define a colormap for source countries
    source_colors = {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }
    
    source_countries = ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']
    
    # Initialize data dictionary
    sankey_data = {
        'bc_regular': {},
        'bc_normalized': {},
        'temp_regular': {},
        'temp_normalized': {}
    }
    
    # Determine which diagrams to show
    has_norm_data = norm_data is not None
    has_temp_data = temp_data is not None
    has_temp_norm_data = temp_norm_data is not None
    
    # Create 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "a) Concentration (ng/m³·person)", 
            "b) Concentration Normalized by Emissions (ng/m³·person)/(g/yr)" if has_norm_data else None,
            "c) Temperature Impact (°C)" if has_temp_data else None,
            "d) Temperature Impact Normalized by Emissions (°C/g/yr)" if has_temp_norm_data else None
        ),
        specs=[
            [{"type": "sankey"}, {"type": "sankey"}],
            [{"type": "sankey"}, {"type": "sankey"}]
        ],
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )
    # ========== TOP ROW: BC CONCENTRATION (regular and normalized) ==========
    for col_idx, (bc_data, col_num, data_key) in enumerate([
        (regular_data, 1, 'bc_regular'), 
        (norm_data, 2, 'bc_normalized')
    ]):
        if bc_data is None:
            continue
        
        try:
            # Get top receptors from 'all' data
            if 'all' not in bc_data:
                continue
                
            all_data_df = bc_data['all'][year].copy()
            if isinstance(all_data_df, pd.DataFrame):
                all_grouped_df = all_data_df.groupby(all_data_df.index)[variable].mean().reset_index()
                all_grouped_df.columns = ['country_impacted', variable]
            else:
                all_grouped_df = all_data_df.reset_index()
                all_grouped_df = all_grouped_df.groupby('country_impacted')[variable].mean().reset_index()
            
            all_grouped_df = all_grouped_df.dropna(subset=[variable])
            
            # Get top receptor countries (NOT excluding source countries)
            top_receptors = all_grouped_df.sort_values(variable, ascending=False).head(top_n)
            receptor_countries = top_receptors['country_impacted'].tolist()
            
            # Now collect data for each source country
            all_sources = []
            all_targets = []
            all_values = []
            all_labels = []
            all_colors = []
            
            # Create nodes list - source countries
            source_nodes = [c.title() for c in source_countries if c in bc_data]
            
            # Identify which receptor countries are also source countries (need dual role nodes)
            dual_role_nodes = {}
            for country in source_countries:
                cap_country = country.title()
                if cap_country in receptor_countries:
                    # Create a separate recipient node for this source country
                    dual_role_nodes[cap_country] = f"{cap_country} (recipient)"
            
            # Build all_nodes list
            all_nodes = source_nodes.copy()
            
            # Add receptor nodes (using dual role mapping where needed)
            for receptor in receptor_countries:
                if receptor in [c.title() for c in source_countries]:
                    # This is a source country that's also a receptor - use the mapped version
                    all_nodes.append(dual_role_nodes.get(receptor, receptor))
                elif receptor not in all_nodes:
                    all_nodes.append(receptor)
            
            # Create node index mapping
            node_indices = {}
            for i, node in enumerate(all_nodes):
                if '(recipient)' in node:
                    # Map both the display name and the internal name
                    original_name = node.split(' (recipient)')[0]
                    node_indices[original_name] = i
                    node_indices[node] = i
                elif node.upper() in source_countries:
                    node_indices[node.upper()] = i
                    node_indices[node] = i  # Also map capitalized version
                else:
                    node_indices[node] = i
            
            # Calculate total values for each node (incoming + outgoing)
            node_values = {node: 0.0 for node in all_nodes}
            
            # First pass to calculate total values
            for source_country in source_countries:
                if source_country not in bc_data:
                    continue
                    
                country_data = bc_data[source_country][year].copy()
                
                # Group by country_impacted
                if isinstance(country_data, pd.DataFrame):
                    country_grouped = country_data.groupby(country_data.index)[variable].mean().reset_index()
                    country_grouped.columns = ['country_impacted', variable]
                else:
                    country_grouped = country_data.reset_index()
                    country_grouped = country_grouped.groupby('country_impacted')[variable].mean().reset_index()
                
                # Filter to top receptors
                country_grouped = country_grouped[country_grouped['country_impacted'].isin(receptor_countries)]
                
                # Sum outgoing values for source countries
                source_title = source_country.title()
                source_total = country_grouped[variable].sum()
                node_values[source_title] += source_total
                
                # Sum incoming values for target countries
                for _, row in country_grouped.iterrows():
                    receptor = row['country_impacted']
                    value = row[variable]
                    
                    if np.isnan(value) or value <= 0:
                        continue
                    
                    # Use the dual role node if this receptor is also a source
                    if receptor.upper() == source_country or receptor == source_title:
                        # This would be a self-loop - use dual role node
                        node_key = dual_role_nodes.get(receptor, receptor)
                    else:
                        node_key = receptor
                    
                    node_values[node_key] += value
            
            # Create links
            for source_country in source_countries:
                if source_country not in bc_data:
                    continue
                    
                source_idx = node_indices[source_country]
                country_data = bc_data[source_country][year].copy()
                
                # Group by country_impacted
                if isinstance(country_data, pd.DataFrame):
                    country_grouped = country_data.groupby(country_data.index)[variable].mean().reset_index()
                    country_grouped.columns = ['country_impacted', variable]
                else:
                    country_grouped = country_data.reset_index()
                    country_grouped = country_grouped.groupby('country_impacted')[variable].mean().reset_index()
                
                # Filter to top receptors
                country_grouped = country_grouped[country_grouped['country_impacted'].isin(receptor_countries)]
                
                for _, row in country_grouped.iterrows():
                    receptor = row['country_impacted']
                    value = row[variable]
                    
                    if np.isnan(value) or value <= 0:
                        continue
                    
                    # Check if this would be a self-loop
                    source_title = source_country.title()
                    if receptor.upper() == source_country or receptor == source_title:
                        # Use the dual role node (never allow loops)
                        target_idx = node_indices[dual_role_nodes.get(receptor, receptor)]
                    else:
                        # Regular target
                        target_idx = node_indices[receptor]
                    
                    # Double check to ensure no self-loops
                    if source_idx == target_idx:
                        continue
                    
                    all_sources.append(source_idx)
                    all_targets.append(target_idx)
                    all_values.append(value)
                    all_labels.append(f"{source_title} → {receptor}: {value:.2e}")
                    
                    # Color based on source
                    source_color = source_colors[source_country]
                    rgba_color = f"rgba{tuple(int(source_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}"
                    all_colors.append(rgba_color)
            
            # Create node colors and labels
            node_colors = []
            node_labels = []
            
            for node in all_nodes:
                # Get the display name
                if '(recipient)' in node:
                    display_name = node.split(' (recipient)')[0]
                    node_labels.append(f"{display_name}")
                else:
                    node_labels.append(f"{node}")
                
                # Set the color
                if node.upper() in source_colors or node.split(' ')[0].upper() in source_colors:
                    country = node.upper() if node.upper() in source_colors else node.split(' ')[0].upper()
                    node_colors.append(source_colors[country])
                else:
                    node_colors.append("gray")
            
            # Store data
            sankey_data[data_key] = {
                'nodes': all_nodes,
                'node_labels': node_labels,
                'node_colors': node_colors,
                'sources': all_sources,
                'targets': all_targets,
                'values': all_values,
                'labels': all_labels,
                'colors': all_colors
            }
            
            # Create Sankey
            sankey = go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors
                ),
                link=dict(
                    source=all_sources,
                    target=all_targets,
                    value=all_values,
                    label=all_labels,
                    color=all_colors
                )
            )
            
            fig.add_trace(sankey, row=1, col=col_num)
            
        except Exception as e:
            print(f"Error creating BC Sankey (col {col_num}): {e}")
            import traceback
            traceback.print_exc()
    
    # ========== BOTTOM ROW: TEMPERATURE (regular and normalized) ==========
    for col_idx, (t_data, col_num, data_key) in enumerate([
        (temp_data, 1, 'temp_regular'), 
        (temp_norm_data, 2, 'temp_normalized')
    ]):
        if t_data is None:
            continue
        
        try:
            # Collect temperature contributions from each emitter
            emitter_contributions = {}
            
            for source_country in source_countries:
                if source_country not in t_data:
                    continue
                
                data_df = t_data[source_country][year].copy()
                
                # Group and take mean (global value)
                if isinstance(data_df, pd.DataFrame):
                    grouped_df = data_df.groupby(data_df.index)[temp_variable].mean().reset_index()
                    grouped_df.columns = ['country_impacted', temp_variable]
                else:
                    grouped_df = data_df.reset_index()
                    grouped_df = grouped_df.groupby('country_impacted')[temp_variable].mean().reset_index()
                
                grouped_df = grouped_df.dropna(subset=[temp_variable])
                
                if not grouped_df.empty:
                    mean_temp = grouped_df[temp_variable].mean()
                    emitter_contributions[source_country] = mean_temp
            
            # Create nodes: Source countries + Global Impact
            source_nodes = [c.title() for c in source_countries if c in emitter_contributions]
            global_node = ['Global Temperature Impact']
            all_nodes = source_nodes + global_node
            
            node_indices = {node: i for i, node in enumerate(all_nodes)}
            global_idx = len(source_nodes)
            
            total_global_temp = sum(emitter_contributions.values())
            
            # Create links: Emitters → Global Impact
            sources = []
            targets = []
            values = []
            labels = []
            colors = []
            
            for source_country in source_nodes:
                country_upper = source_country.upper()
                if country_upper in emitter_contributions:
                    source_idx = node_indices[source_country]
                    temp_contribution = emitter_contributions[country_upper]
                    
                    sources.append(source_idx)
                    targets.append(global_idx)
                    values.append(temp_contribution)
                    labels.append(f"{source_country} → Global: {temp_contribution:.2e} K")
                    
                    source_color = source_colors[country_upper]
                    rgba_color = f"rgba{tuple(int(source_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.6,)}"
                    colors.append(rgba_color)
            
            # Store data
            sankey_data[data_key] = {
                'nodes': all_nodes,
                'links': {
                    'sources': [all_nodes[s] for s in sources],
                    'targets': [all_nodes[t] for t in targets],
                    'values': values
                },
                'contributions': emitter_contributions,
                'total_global_temp': total_global_temp
            }
            
            # Node colors
            node_colors = []
            for node in all_nodes:
                if node.upper() in source_colors:
                    node_colors.append(source_colors[node.upper()])
                elif node == 'Global Temperature Impact':
                    node_colors.append('#CC2936')  # Red
                else:
                    node_colors.append('gray')
            
            # Create Sankey
            sankey = go.Sankey(
                arrangement='snap',
                node=dict(
                    pad=15,
                    thickness=30,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=labels,
                    color=colors
                )
            )
            
            fig.add_trace(sankey, row=2, col=col_num)
            
        except Exception as e:
            print(f"Error creating Temperature Sankey (col {col_num}): {e}")
    
    # Update layout
    fig.update_layout(
        font_size=12,
        height=900,
        width=1400,
        margin=dict(l=50, r=50, b=50, t=80)
    )
    
    return fig, sankey_data

def analyze_emissions_by_var(full_ds, var, country= None):
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
    if country:
        # Filter by country if specified
        sorted_ds = sorted_ds.where(sorted_ds['country_emitting'] == country, drop=True)
    
    # Get the number of plants
    n_plants = len(sorted_ds['unique_ID'])
    
    # Initialize dictionaries to store results
    age_ds = {}
    bckg_age_ds = {}
    
    # Create datasets for each threshold
    for idx in range(1, n_plants + 1):  # Start from 1 to n_plants
        # Oldest plants up to idx
        age_ds[idx] = sorted_ds.isel(unique_ID=slice(0, idx)).sum(dim='unique_ID')
        bckg_age_ds[idx] = sorted_ds.isel(unique_ID=slice(idx, n_plants)).sum(dim='unique_ID')

    # Concatenate along a new dimension representing the threshold
    age_ds = xr.concat([age_ds[i] for i in range(1, n_plants + 1)], 
                       pd.Index(range(1, n_plants + 1), name='plants_open'))
    
    bckg_age_ds = xr.concat([bckg_age_ds[i] for i in range(1, n_plants + 1)], 
                           pd.Index(range(1, n_plants + 1), name='plants_open'))
    
    return age_ds, bckg_age_ds

def plot_variable_by_country(dataset, variable, country=None,
                           contour_variable=None, levels=10,
                           figsize=(10, 6), ax=None, target_year=2040, target_co2=0.001,
                           xlim=None, ylim=None, vmin=None, vmax=None, 
                           cmap='cividis', colorbar_label=None, flip_y_axis = True):
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
    xlim : tuple or None
        Custom x-axis limits as (min, max)
    ylim : tuple or None
        Custom y-axis limits as (min, max)
    vmin, vmax : float or None
        Custom colorbar limits
    cmap : str
        Colormap name
    colorbar_label : str or None
        Custom colorbar label. If None, uses default based on variable.
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Select data based on provided parameters
    data = dataset[variable]
    
    # Apply country selection if applicable (for impacted country)
    if country is not None and 'country_impacted' in data.dims:
        data = data.sel(country_impacted=country)
    
    # Plot the main data with custom color limits if provided
    kwargs = {'cmap': cmap, 'add_colorbar': False}
    if vmin is not None:
        kwargs['vmin'] = vmin
    if vmax is not None:
        kwargs['vmax'] = vmax
    
    im = data.plot(ax=ax, **kwargs)
    
    # Add colorbar with custom label
    cbar = plt.colorbar(im, ax=ax)
    
    # Set colorbar label based on variable or custom label
    if colorbar_label:
        cbar.set_label(colorbar_label)
    elif variable == 'BC_surface_mean_conc':
        cbar.set_label('BC Surface Concentration (ng m$^{-3}$)')
    elif variable == 'co2_emissions':
        cbar.set_label('CO$_2$ Emissions (Gt yr$^{-1}$)')
    elif variable == 'BC_pop_weight_mean_conc':
        cbar.set_label('BC Population Weighted Surface Concentration (ng m$^{-3}$)')
    elif variable == 'BC_column_mean_conc':
        cbar.set_label('BC Column Concentration (ng m$^{-3}$)')
    else:
        cbar.set_label(variable.replace('_', ' ').title())

    # Add contour plot if requested
    if contour_variable is not None:
        contour_data = dataset[contour_variable]
        
        # Apply same selections to contour data
        if country is not None and 'country_impacted' in contour_data.dims:
            contour_data = contour_data.sel(country_impacted=country)
    
            
        # Convert CO2 emissions to GtCO2 if that's the contour variable
        if contour_variable == 'co2_emissions':
            contour_data = contour_data / 1e9  # Convert to GtCO2
            contour_units = 'GtCO₂'
            
            # Find the y-value (number of plants) that corresponds to target CO2 at target year
            if 'scenario_year' in contour_data.dims:
                # Get the data for the target year
                year_data = contour_data.sel(scenario_year=target_year)
            elif 'year' in contour_data.dims:
                # Get the data for the target year
                year_data = contour_data.sel(year=target_year)
            if 'scenario_year' or 'year' in contour_data.dims:
                if len(year_data) > 0:
                    # Convert to numpy array for easier manipulation
                    year_values = year_data.values
                    plants_values = year_data.plants_open.values
                    
                    # Find the closest number of plants that gives target CO2
                    idx = np.abs(year_values - target_co2).argmin()
                    plants_for_target = plants_values[idx]
                    
                    # Get current axis limits
                    current_xlim = ax.get_xlim()
                    current_ylim = ax.get_ylim()
                    
                    # Plot vertical line from bottom to target plants
                    ax.plot([target_year, target_year], 
                           [current_ylim[0], plants_for_target],
                           color='white', linestyle='--', alpha=1, linewidth=2)
                    
                    # Plot horizontal line from left to target year
                    ax.plot([current_xlim[0], target_year],
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

    total_plants = len(dataset['plants_open'])
    # Apply y-axis flipping if requested
    if flip_y_axis and 'plants_open' in data.dims:
        # Create a new coordinate for plants_closed
        plants_closed = total_plants - data['plants_open']
        
        # Reindex the data with the new coordinate
        data = data.assign_coords(plants_closed=('plants_open', plants_closed.values))
        
        # Swap the dimension
        data = data.swap_dims({'plants_open': 'plants_closed'})
        
        # Sort by the new dimension to ensure proper ordering
        data = data.sortby('plants_closed')
        
        # Remove the old coordinate to avoid confusion
        data = data.drop('plants_open')

    # Set axis labels based on whether we flipped the y-axis
    if flip_y_axis:
        ax.set_ylabel('Number of Plants Closed')
    else:
        ax.set_ylabel('Number of Plants Open')
    
    ax.set_xlabel('Year')

        
    # Set custom axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    

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
        if flip_y_axis:
            # Reverse the tick positions and labels
            yticks = ax2.get_yticks()
            ax2.set_yticks(yticks)
            
            # Extract text from each label and convert to float
            try:
                # Get the current tick positions and their corresponding values
                ytick_labels = [float(label.get_text()) if label.get_text() else 0 
                                for label in ax2.get_yticklabels()]
                
                # Create new labels showing capacity lost
                new_labels = [f'{max_mw - val:.0f}' if val <= max_mw else '0' 
                            for val in ytick_labels]
                
                ax2.set_yticklabels(new_labels)
            except ValueError:
                # Fallback if there's an issue with the labels
                ax2.set_yticklabels([f'{max_mw - i*(max_mw/len(yticks)):.0f}' 
                                for i in range(len(yticks))])
            
            ax2.set_ylabel('Capacity Lost (GW)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
        else:
            ax2.set_yticks(plant_ticks)
            ax2.set_yticklabels(mw_labels)
            ax2.set_ylabel('Cumulative Capacity (GW)', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
        # Store the MW data for later use
        ax2.mw_data = mw_data
    
    # Create title with appropriate information
    variable_name = variable.replace('_', ' ').title()
    country_info = f" in {country}" if country is not None else ""
    
    title = f"{variable_name}{country_info}{scenario_info}"
    if contour_variable is not None:
        title += f" with {contour_variable.replace('_', ' ').title()} contours"
    
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if ax is None:  # Only apply tight_layout if we created the figure
        plt.tight_layout()
    return fig, ax

def plot_comparison(age_ds, mw_ds, emis_intens_ds, variable, country=None,  
                  contour_variable=None, levels=10, figsize=(30, 8), 
                   target_year=2040, target_co2=10, flip_y_axis = True):
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
    plot_variable_by_country( age_ds, variable, country, 
                           contour_variable, levels, ax=axes[0],
                           target_year=target_year, target_co2=target_co2)
    axes[0].set_title("Sorted by Age (oldest first)")
    
    # Plot for size-sorted dataset
    plot_variable_by_country(mw_ds, variable, country, 
                           contour_variable, levels, ax=axes[1],
                           target_year=target_year, target_co2=target_co2)
    axes[1].set_title("Sorted by Plant Size (largest first)")
    
    # Plot for emission intensity-sorted dataset
    plot_variable_by_country(emis_intens_ds, variable, country, 
                           contour_variable, levels, ax=axes[2],
                           target_year=target_year, target_co2=target_co2)
    axes[2].set_title("Sorted by Emission Intensity (highest first)")
    
    plt.tight_layout()
    return fig, axes


def plot_emissions_and_temperature(CGP_df, start_year=2000, end_year=2060, tcr=1.65, tcr_uncertainty=0.4, 
                                 breakdown_by=None, show_lifetime_uncertainty=True, lifetime_uncertainty=5,
                                 figsize=(12, 8), colors_in=None):
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
    colors : dict, optional
        Dictionary mapping categories to colors
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    tcr = np.round(float(tcr), 2)

    # Define default colors for countries if not provided
    default_colors =  {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }
    
    # Use provided colors or default to the predefined set
    colors = colors_in or default_colors
    
    # Function to format country names
    def format_country_name(name):
        if isinstance(name, str):
            if name == 'US':
                return 'United States'
            else:
                return name.title()
        return name
    
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
        # Create dictionaries for formatted category names
        formatted_categories = {}
        category_colors = {}
        
        for idx, cat in enumerate(categories):
            formatted_name = format_country_name(cat)
            formatted_categories[cat] = formatted_name
            if colors_in:
                category_colors[cat] = colors.get(idx, 'gray')  # Use predefined colors or default to gray
            else:
                category_colors[cat] = colors.get(cat, 'gray')
        category_emissions = {cat: np.zeros(years) for cat in categories}
        if show_lifetime_uncertainty:
            category_min_emissions = {cat: np.zeros(years) for cat in categories}
            category_max_emissions = {cat: np.zeros(years) for cat in categories}
    else:
        categories = None
        formatted_categories = None
        category_colors = None
        category_emissions = None

    for unique_id in CGP_df['unique_ID'].values:
        # Get plant data
        plant_data = CGP_df.loc[CGP_df['unique_ID'] == unique_id]
        
        # Get annual CO2 emissions (in GtCO2)
        annual_co2 = float(plant_data['co2_emissions']) / 1e9  # Convert to GtCO2 and ensure scalar
        
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
        
        # Prepare data for stackplot with custom colors and labels
        stack_data = []
        stack_colors = []
        stack_labels = []
        
        for cat in categories:
            stack_data.append(yearly_category_emissions[cat])
            stack_colors.append(category_colors[cat])
            stack_labels.append(formatted_categories[cat])
            
            # Add uncertainty shading if enabled
            if show_lifetime_uncertainty:
                yearly_category_min = np.cumsum(category_min_emissions[cat])
                yearly_category_max = np.cumsum(category_max_emissions[cat])
                ax1.fill_between(time_array, 
                               yearly_category_min, 
                               yearly_category_max,
                               alpha=0.1, color=category_colors[cat])
        
        # Create stackplot with custom colors and labels
        ax1.stackplot(time_array, stack_data, labels=stack_labels, colors=stack_colors)
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
    if show_lifetime_uncertainty and not breakdown_by:
        ax1.legend(loc='upper left')
    ax1.set_title(r'a) CO$_2$ Emissions')

    # Bottom-left: CO2 temperature response with both TCR and lifetime uncertainty
    ax_co2_temp = ax2
    
    co2_temp_color = "#B4204C"
    
    # Plot the central estimate
    ax_co2_temp.plot(time_array, co2_temp_response,
            color=co2_temp_color, linewidth=2, label = f'TCR = {tcr:.2f}$\circ$C/1000GtCO2')
    
    # First plot the lifetime uncertainty (if enabled)
    if show_lifetime_uncertainty:
        # Calculate the full range of uncertainty combining both factors
        min_combined = min_cumulative * ((tcr - tcr_uncertainty) / 1000)  # Min lifetime, min TCR
        max_combined = max_cumulative * ((tcr + tcr_uncertainty) / 1000)  # Max lifetime, max TCR
        
        # Plot outline of the combined uncertainty range
        ax_co2_temp.plot(time_array, min_combined, color= "#206AB4", linestyle='--', linewidth=2, label = f'TCR = {(tcr - tcr_uncertainty):.2f}$\circ$C/1000GtCO2')
        ax_co2_temp.plot(time_array, max_combined, color= "#20AFB4", linestyle='--', linewidth=2,  label = f'TCR = {(tcr + tcr_uncertainty):.2f}$\circ$C/1000GtCO2')

        # Calculate temperature response for min/max emissions with central TCR
        min_temp_response = min_cumulative * (tcr / 1000)
        max_temp_response = max_cumulative * (tcr / 1000)
        
        # Plot lifetime uncertainty band
        ax_co2_temp.fill_between(time_array, min_temp_response, max_temp_response,
                        color='gray', alpha=0.2, label='Lifetime Uncertainty')

        # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, min_temp_response, min_combined,
                        color=co2_temp_color, alpha=0.2)
        ax_co2_temp.fill_between(time_array, max_temp_response, max_combined,
                        color=co2_temp_color, alpha=0.2, label='TCR Uncertainty')
          
    elif not show_lifetime_uncertainty:
        # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, co2_temp_lower, co2_temp_upper,
                    color=co2_temp_color, alpha=0.2, label='TCR Uncertainty')
    


    ax_co2_temp.set_xlabel('Year')
    ax_co2_temp.set_ylabel('Temperature Response (°C)')
    ax_co2_temp.set_title(r'b) CO$_2$ Temperature Impact')
    ax_co2_temp.grid(True, alpha=0.3)
    ax_co2_temp.legend(loc='upper left', fontsize='small')
    
    plt.tight_layout()
    
    return cumulative_emissions, co2_temp_response

def plot_emissions_and_temperature_with_bc_artp(CGP_df, start_year=2000, end_year=2060, tcr=1.65, tcr_uncertainty=0.4, 
                                         breakdown_by=None, show_lifetime_uncertainty=True, lifetime_uncertainty=5,
                                         figsize=(14, 10)):
    """
    Plot cumulative CO2 emissions and temperature response over time, using Global ARTP values for BC impact.
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
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Define color mapping for countries
    source_colors = {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }
    
    # Function to format country names
    def format_country_name(name):
        if isinstance(name, str):
            return name.title()
        return name
    
    tcr = np.round(float(tcr), 2)

    # Set up time array
    years = end_year - start_year
    time_array = np.arange(start_year, end_year)
    
    # Initialize arrays for emissions and temperature responses
    total_emissions = np.zeros(years)
    total_bc_emissions = np.zeros(years)  # BC emissions in g/yr
    
    # ARTP temperature impact arrays
    total_global_10 = np.zeros(years)
    total_global_20 = np.zeros(years)
    total_global_50 = np.zeros(years)
    total_global_100 = np.zeros(years)
    
    if show_lifetime_uncertainty:
        min_emissions = np.zeros(years)
        max_emissions = np.zeros(years)
        min_bc_emissions = np.zeros(years)
        max_bc_emissions = np.zeros(years)
        
        # ARTP uncertainty arrays
        min_global_10 = np.zeros(years)
        max_global_10 = np.zeros(years)
        min_global_20 = np.zeros(years)
        max_global_20 = np.zeros(years)
        min_global_50 = np.zeros(years)
        max_global_50 = np.zeros(years)
        min_global_100 = np.zeros(years)
        max_global_100 = np.zeros(years)
    
    # If breaking down emissions, create separate arrays for each category
    if breakdown_by:
        categories = CGP_df[breakdown_by].unique()
        # Create dictionaries for formatted category names and colors
        formatted_categories = {}
        category_colors = {}
        
        for cat in categories:
            formatted_name = format_country_name(cat)
            formatted_categories[cat] = formatted_name
            category_colors[cat] = source_colors.get(cat, 'gray')  # Use predefined colors or default to gray
        
        category_emissions = {cat: np.zeros(years) for cat in categories}
        category_bc_emissions = {cat: np.zeros(years) for cat in categories}
        
        # ARTP category arrays
        category_global_10 = {cat: np.zeros(years) for cat in categories}
        category_global_20 = {cat: np.zeros(years) for cat in categories}
        category_global_50 = {cat: np.zeros(years) for cat in categories}
        category_global_100 = {cat: np.zeros(years) for cat in categories}
        
        if show_lifetime_uncertainty:
            category_min_emissions = {cat: np.zeros(years) for cat in categories}
            category_max_emissions = {cat: np.zeros(years) for cat in categories}
            category_min_bc_emissions = {cat: np.zeros(years) for cat in categories}
            category_max_bc_emissions = {cat: np.zeros(years) for cat in categories}
            
            # ARTP category uncertainty arrays
            category_min_global_10 = {cat: np.zeros(years) for cat in categories}
            category_max_global_10 = {cat: np.zeros(years) for cat in categories}
            category_min_global_20 = {cat: np.zeros(years) for cat in categories}
            category_max_global_20 = {cat: np.zeros(years) for cat in categories}
            category_min_global_50 = {cat: np.zeros(years) for cat in categories}
            category_max_global_50 = {cat: np.zeros(years) for cat in categories}
            category_min_global_100 = {cat: np.zeros(years) for cat in categories}
            category_max_global_100 = {cat: np.zeros(years) for cat in categories}
    else:
        categories = None
        formatted_categories = None
        category_colors = None
        category_emissions = None
        category_bc_emissions = None
        category_global_10 = None
        category_global_20 = None
        category_global_50 = None
        category_global_100 = None

    for unique_id in CGP_df['unique_ID'].values:
        # Get plant data
        plant_data = CGP_df.loc[CGP_df['unique_ID'] == unique_id]
        
        # Get annual values
        annual_co2 = float(plant_data['co2_emissions']) / 1e9  # Convert to GtCO2
        annual_bc = float(plant_data['BC_(g/yr)'])  # BC emissions in g/yr
        
        # Get ARTP temperature responses
        annual_global_10 = float(plant_data['Global_10'])
        annual_global_20 = float(plant_data['Global_20'])
        annual_global_50 = float(plant_data['Global_50'])
        annual_global_100 = float(plant_data['Global_100'])
        
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
                        
                        # Add ARTP temperature impacts
                        total_global_10[yr_offset + yr] += annual_global_10
                        total_global_20[yr_offset + yr] += annual_global_20
                        total_global_50[yr_offset + yr] += annual_global_50
                        total_global_100[yr_offset + yr] += annual_global_100
                        
                        if breakdown_by:
                            category = plant_data[breakdown_by].iloc[0]
                            category_emissions[category][yr_offset + yr] += annual_co2
                            category_bc_emissions[category][yr_offset + yr] += annual_bc
                            
                            # Add ARTP temperature impacts by category
                            category_global_10[category][yr_offset + yr] += annual_global_10
                            category_global_20[category][yr_offset + yr] += annual_global_20
                            category_global_50[category][yr_offset + yr] += annual_global_50
                            category_global_100[category][yr_offset + yr] += annual_global_100
                
                if show_lifetime_uncertainty:
                    min_operating_years = int(max(0, operating_years - lifetime_uncertainty))
                    max_operating_years = int(min(operating_years + lifetime_uncertainty, 
                                            end_year - plant_data['Year_of_Commission'].iloc[0]))
                    
                    for yr in range(min_operating_years):
                        if yr_offset + yr < years:
                            min_emissions[yr_offset + yr] += annual_co2
                            min_bc_emissions[yr_offset + yr] += annual_bc
                            
                            # Add ARTP temperature impacts with min lifetime
                            min_global_10[yr_offset + yr] += annual_global_10
                            min_global_20[yr_offset + yr] += annual_global_20
                            min_global_50[yr_offset + yr] += annual_global_50
                            min_global_100[yr_offset + yr] += annual_global_100
                            
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                category_min_emissions[category][yr_offset + yr] += annual_co2
                                category_min_bc_emissions[category][yr_offset + yr] += annual_bc
                                
                                # Add ARTP temperature impacts by category with min lifetime
                                category_min_global_10[category][yr_offset + yr] += annual_global_10
                                category_min_global_20[category][yr_offset + yr] += annual_global_20
                                category_min_global_50[category][yr_offset + yr] += annual_global_50
                                category_min_global_100[category][yr_offset + yr] += annual_global_100
                    
                    for yr in range(max_operating_years):
                        if yr_offset + yr < years:
                            max_emissions[yr_offset + yr] += annual_co2
                            max_bc_emissions[yr_offset + yr] += annual_bc
                            
                            # Add ARTP temperature impacts with max lifetime
                            max_global_10[yr_offset + yr] += annual_global_10
                            max_global_20[yr_offset + yr] += annual_global_20
                            max_global_50[yr_offset + yr] += annual_global_50
                            max_global_100[yr_offset + yr] += annual_global_100
                            
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                category_max_emissions[category][yr_offset + yr] += annual_co2
                                category_max_bc_emissions[category][yr_offset + yr] += annual_bc
                                
                                # Add ARTP temperature impacts by category with max lifetime
                                category_max_global_10[category][yr_offset + yr] += annual_global_10
                                category_max_global_20[category][yr_offset + yr] += annual_global_20
                                category_max_global_50[category][yr_offset + yr] += annual_global_50
                                category_max_global_100[category][yr_offset + yr] += annual_global_100

    # Calculate cumulative values
    cumulative_emissions = np.cumsum(total_emissions)
    cumulative_bc_emissions = np.cumsum(total_bc_emissions)
    
    # Calculate cumulative ARTP temperature impacts
    cumulative_global_10 = np.cumsum(total_global_10)
    cumulative_global_20 = np.cumsum(total_global_20)
    cumulative_global_50 = np.cumsum(total_global_50)
    cumulative_global_100 = np.cumsum(total_global_100)
    
    if show_lifetime_uncertainty:
        min_cumulative = np.cumsum(min_emissions)
        max_cumulative = np.cumsum(max_emissions)
        min_cumulative_bc = np.cumsum(min_bc_emissions)
        max_cumulative_bc = np.cumsum(max_bc_emissions)
        
        # Calculate cumulative ARTP temperature impacts
        min_cumulative_global_20 = np.cumsum(min_global_20)
        max_cumulative_global_20 = np.cumsum(max_global_20)

    # Calculate temperature responses
    co2_temp_response = cumulative_emissions * (tcr / 1000)
    co2_temp_upper = cumulative_emissions * ((tcr + tcr_uncertainty) / 1000)
    co2_temp_lower = cumulative_emissions * ((tcr - tcr_uncertainty) / 1000)
    
    # Total temperature response (CO2 + BC)
    total_temp_response = co2_temp_response + cumulative_global_20

    # Create the 2x2 plot layout
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # ----- Left side (CO2) -----
    # Top-left: CO2 emissions
    ax_co2_emissions = axs[0, 0]
    if breakdown_by and category_emissions:
        # Plot stacked area for each category
        yearly_category_emissions = {cat: np.cumsum(em) for cat, em in category_emissions.items()}
        
        # Prepare data for stackplot with custom colors and labels
        stack_data = []
        stack_colors = []
        stack_labels = []
        
        for cat in categories:
            stack_data.append(yearly_category_emissions[cat])
            stack_colors.append(category_colors[cat])
            stack_labels.append(formatted_categories[cat])
            
            # Add uncertainty shading if enabled
            if show_lifetime_uncertainty:
                yearly_category_min = np.cumsum(category_min_emissions[cat])
                yearly_category_max = np.cumsum(category_max_emissions[cat])
                ax_co2_emissions.fill_between(time_array, 
                                            yearly_category_min, 
                                            yearly_category_max,
                                            alpha=0.1, color=category_colors[cat])
        
        # Create stackplot with custom colors
        ax_co2_emissions.stackplot(time_array, stack_data, labels=stack_labels, colors=stack_colors)
        ax_co2_emissions.legend(loc='upper left', fontsize='small')
    else:
        if show_lifetime_uncertainty:
            ax_co2_emissions.fill_between(time_array, min_cumulative, max_cumulative,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax_co2_emissions.plot(time_array, cumulative_emissions, 
                color='tab:blue', linewidth=2, label='Cumulative CO2 Emissions')
    
    ax_co2_emissions.set_ylabel('Cumulative CO2 Emissions (GtCO2)')
    ax_co2_emissions.set_title(r'a) CO$_2$ Emissions')
    ax_co2_emissions.grid(True, alpha=0.3)
    if show_lifetime_uncertainty and not breakdown_by:
        ax_co2_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-left: CO2 temperature response with both TCR and lifetime uncertainty
    ax_co2_temp = axs[1, 0]
    co2_temp_color = "#B4204C"

    # Plot the central estimate
    ax_co2_temp.plot(time_array, co2_temp_response,
            color=co2_temp_color, linewidth=2, label = f'TCR = {tcr:.2f}$\circ$C/1000GtCO2')
    
    # First plot the lifetime uncertainty (if enabled)
    if show_lifetime_uncertainty:
        # Calculate the full range of uncertainty combining both factors
        min_combined = min_cumulative * ((tcr - tcr_uncertainty) / 1000)  # Min lifetime, min TCR
        max_combined = max_cumulative * ((tcr + tcr_uncertainty) / 1000)  # Max lifetime, max TCR
        
        # Plot outline of the combined uncertainty range
        ax_co2_temp.plot(time_array, min_combined, color= "#206AB4", linestyle='--', linewidth=2, label = f'TCR = {(tcr - tcr_uncertainty):.2f}$\circ$C/1000GtCO2')
        ax_co2_temp.plot(time_array, max_combined, color= "#20AFB4", linestyle='--', linewidth=2,  label = f'TCR = {tcr + tcr_uncertainty}$\circ$C/1000GtCO2')

        # Calculate temperature response for min/max emissions with central TCR
        min_temp_response = min_cumulative * (tcr / 1000)
        max_temp_response = max_cumulative * (tcr / 1000)
        
        # Plot lifetime uncertainty band
        ax_co2_temp.fill_between(time_array, min_temp_response, max_temp_response,
                        color='gray', alpha=0.2, label='Lifetime Uncertainty')

        # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, min_temp_response, min_combined,
                        color=co2_temp_color, alpha=0.2)
        ax_co2_temp.fill_between(time_array, max_temp_response, max_combined,
                        color=co2_temp_color, alpha=0.2, label='TCR Uncertainty')
          
    elif not show_lifetime_uncertainty:
        # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, co2_temp_lower, co2_temp_upper,
                    color=co2_temp_color, alpha=0.2, label='TCR Uncertainty')
    

    ax_co2_temp.set_xlabel('Year')
    ax_co2_temp.set_ylabel('Temperature Response (°C)')
    ax_co2_temp.set_title(r'c) CO$_2$ Temperature Impact')
    ax_co2_temp.grid(True, alpha=0.3)
    ax_co2_temp.legend(loc='upper left', fontsize='small')
    
    # ----- Right side (BC) -----
    # Top-right: BC emissions
    ax_bc_emissions = axs[0, 1]
    
    if breakdown_by and category_bc_emissions:
        # Plot stacked area for each category
        yearly_category_bc_emissions = {cat: np.cumsum(em) for cat, em in category_bc_emissions.items()}
        
        # Prepare data for stackplot with custom colors and labels
        stack_data = []
        stack_colors = []
        stack_labels = []
        
        for cat in categories:
            stack_data.append(yearly_category_bc_emissions[cat])
            stack_colors.append(category_colors[cat])
            stack_labels.append(formatted_categories[cat])
            
            # Add uncertainty shading if enabled
            if show_lifetime_uncertainty:
                yearly_category_min_bc = np.cumsum(category_min_bc_emissions[cat])
                yearly_category_max_bc = np.cumsum(category_max_bc_emissions[cat])
                ax_bc_emissions.fill_between(time_array, 
                                           yearly_category_min_bc, 
                                           yearly_category_max_bc,
                                           alpha=0.1, color=category_colors[cat])
        
        # Create stackplot with custom colors
        ax_bc_emissions.stackplot(time_array, stack_data, labels=stack_labels, colors=stack_colors)
        ax_bc_emissions.legend(loc='upper left', fontsize='small')
    else:
        if show_lifetime_uncertainty:
            ax_bc_emissions.fill_between(time_array, min_cumulative_bc, max_cumulative_bc,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax_bc_emissions.plot(time_array, cumulative_bc_emissions, 
                color='tab:green', linewidth=2, label='Cumulative Black Carbon Emissions')
    
    # Format y-axis for better readability (e.g., using scientific notation or appropriate units)
    ax_bc_emissions.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    ax_bc_emissions.set_ylabel('Cumulative Black Carbon Emissions (g)')
    ax_bc_emissions.set_title('b) Black Carbon Emissions')
    ax_bc_emissions.grid(True, alpha=0.3)
    if show_lifetime_uncertainty and not breakdown_by:
        ax_bc_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-right: BC temperature responses (Global ARTP) with time horizon uncertainty
    ax_bc_temp = axs[1, 1]
    
    bc_temp_color =  "#B82A0A"
    # Plot Global_20 as the main temperature impact
    ax_bc_temp.plot(time_array, cumulative_global_20, 
            color= bc_temp_color , linewidth=2, label='20-year Global ARTP')
    
    # Plot other time horizons for context
    ax_bc_temp.plot(time_array, cumulative_global_10,
            color= "#066F8F", linewidth=2, linestyle='--', label='10-year Global ARTP')
    ax_bc_temp.plot(time_array, cumulative_global_50,
            color= "#039391", linewidth=2, linestyle='--',  label='50-year Global ARTP')
    ax_bc_temp.plot(time_array, cumulative_global_100,
            color=  "#8865A5", linewidth=2, linestyle='--',  label='100-year Global ARTP')
    
    # Fill between the time horizon uncertainty (10-year to 100-year)
    ax_bc_temp.fill_between(time_array, max_cumulative_global_20, cumulative_global_10,
                    color=bc_temp_color, alpha=0.2, label='Time Horizon Uncertainty')
    ax_bc_temp.fill_between(time_array, cumulative_global_100, min_cumulative_global_20,
                    color=bc_temp_color, alpha=0.2)
    
    # Show lifetime uncertainty on the Global_20 ARTP temperature
    if show_lifetime_uncertainty:
        # Plot lifetime uncertainty band for Global_20
        ax_bc_temp.fill_between(time_array, min_cumulative_global_20, max_cumulative_global_20,
                        color='gray', alpha=0.2, label='Lifetime Uncertainty')
    
    ax_bc_temp.set_xlabel('Year')
    ax_bc_temp.set_ylabel('Cumulative Temperature Response (°C)')
    ax_bc_temp.set_title('d) Black Carbon Temperature Impact')
    ax_bc_temp.grid(True, alpha=0.3)
    ax_bc_temp.legend(loc='upper left', fontsize='small')
    
    for ax in [ax_co2_emissions, ax_co2_temp, ax_bc_emissions, ax_bc_temp]: 
        ax.set_xlim(start_year, end_year)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    return (cumulative_emissions, co2_temp_response, 
            cumulative_bc_emissions, cumulative_global_20, 
            {'global_10': cumulative_global_10, 
             'global_50': cumulative_global_50, 
             'global_100': cumulative_global_100},
            total_temp_response)
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
    import numpy as np
    import matplotlib.pyplot as plt
    
    tcr = np.round(float(tcr), 2)

    # Define color mapping for countries
    source_colors = {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }
    
    # Function to format country names
    def format_country_name(name):
        if isinstance(name, str):
            return name.title()
        return name
    
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
        # Create dictionaries for formatted category names and colors
        formatted_categories = {}
        category_colors = {}
        
        for cat in categories:
            formatted_name = format_country_name(cat)
            formatted_categories[cat] = formatted_name
            category_colors[cat] = source_colors.get(cat, 'gray')  # Use predefined colors or default to gray
            
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
        formatted_categories = None
        category_colors = None
        category_emissions = None
        category_bc_emissions = None
        category_dt_drf = None
        category_dt_snowrf = None

    for unique_id in CGP_df['unique_ID'].values:
        # Get plant data
        plant_data = CGP_df.loc[CGP_df['unique_ID'] == unique_id]
        
        # Get annual values
        annual_co2 = float(plant_data['co2_emissions']) / 1e9  # Convert to GtCO2
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
        
        # Prepare data for stackplot with custom colors and labels
        stack_data = []
        stack_colors = []
        stack_labels = []
        
        for cat in categories:
            stack_data.append(yearly_category_emissions[cat])
            stack_colors.append(category_colors[cat])
            stack_labels.append(formatted_categories[cat])
            
            # Add uncertainty shading if enabled
            if show_lifetime_uncertainty:
                yearly_category_min = np.cumsum(category_min_emissions[cat])
                yearly_category_max = np.cumsum(category_max_emissions[cat])
                ax_co2_emissions.fill_between(time_array, 
                                            yearly_category_min, 
                                            yearly_category_max,
                                            alpha=0.1, color=category_colors[cat])
        
        # Create stackplot with custom colors
        ax_co2_emissions.stackplot(time_array, stack_data, labels=stack_labels, colors=stack_colors)
        ax_co2_emissions.legend(loc='upper left', fontsize='small')
    else:
        if show_lifetime_uncertainty:
            ax_co2_emissions.fill_between(time_array, min_cumulative, max_cumulative,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax_co2_emissions.plot(time_array, cumulative_emissions, 
                color='tab:blue', linewidth=2, label='Cumulative CO2 Emissions')
    
    ax_co2_emissions.set_ylabel('Cumulative CO2 Emissions (GtCO2)')
    ax_co2_emissions.set_title(r'a) CO$_2$ Emissions')
    ax_co2_emissions.grid(True, alpha=0.3)
    if show_lifetime_uncertainty and not breakdown_by:
        ax_co2_emissions.legend(loc='upper left', fontsize='small')
    
    
    # Bottom-left: CO2 temperature response with both TCR and lifetime uncertainty
    ax_co2_temp = axs[1, 0]
    co2_temp_color = "#B4204C"

    # Plot the central estimate
    ax_co2_temp.plot(time_array, co2_temp_response,
            color=co2_temp_color, linewidth=2, label = f'TCR = {tcr:.2f}$\circ$C/1000GtCO2')
    
    # First plot the lifetime uncertainty (if enabled)
    if show_lifetime_uncertainty:
        # Calculate the full range of uncertainty combining both factors
        min_combined = min_cumulative * ((tcr - tcr_uncertainty) / 1000)  # Min lifetime, min TCR
        max_combined = max_cumulative * ((tcr + tcr_uncertainty) / 1000)  # Max lifetime, max TCR
        
        # Plot outline of the combined uncertainty range
        ax_co2_temp.plot(time_array, min_combined, color= "#206AB4", linestyle='--', linewidth=2, label = f'TCR = {(tcr - tcr_uncertainty):.2f}$\circ$C/1000GtCO2')
        ax_co2_temp.plot(time_array, max_combined, color= "#20AFB4", linestyle='--', linewidth=2,  label = f'TCR = {tcr + tcr_uncertainty}$\circ$C/1000GtCO2')

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
          
    elif not show_lifetime_uncertainty:
        # Then plot the TCR uncertainty on the central emissions estimate
        ax_co2_temp.fill_between(time_array, co2_temp_lower, co2_temp_upper,
                    color='tab:red', alpha=0.2, label='TCR Uncertainty')
    
    # Plot the central estimate
    ax_co2_temp.plot(time_array, co2_temp_response,
            color='tab:red', linewidth=2, label=r'CO$_2$ Temperature Response')
    

    
    ax_co2_temp.set_xlabel('Year')
    ax_co2_temp.set_ylabel('Temperature Response (°C)')
    ax_co2_temp.set_title(r'c) CO$_2$ Temperature Impact')
    ax_co2_temp.grid(True, alpha=0.3)
    ax_co2_temp.legend(loc='upper left', fontsize='small')
    
    # ----- Right side (BC) -----
    # Top-right: BC emissions
    ax_bc_emissions = axs[0, 1]
    
    if breakdown_by and category_bc_emissions:
        # Plot stacked area for each category
        yearly_category_bc_emissions = {cat: np.cumsum(em) for cat, em in category_bc_emissions.items()}
        
        # Prepare data for stackplot with custom colors and labels
        stack_data = []
        stack_colors = []
        stack_labels = []
        
        for cat in categories:
            stack_data.append(yearly_category_bc_emissions[cat])
            stack_colors.append(category_colors[cat])
            stack_labels.append(formatted_categories[cat])
            
            # Add uncertainty shading if enabled
            if show_lifetime_uncertainty:
                yearly_category_min_bc = np.cumsum(category_min_bc_emissions[cat])
                yearly_category_max_bc = np.cumsum(category_max_bc_emissions[cat])
                ax_bc_emissions.fill_between(time_array, 
                                           yearly_category_min_bc, 
                                           yearly_category_max_bc,
                                           alpha=0.1, color=category_colors[cat])
        
        # Create stackplot with custom colors
        ax_bc_emissions.stackplot(time_array, stack_data, labels=stack_labels, colors=stack_colors)
        ax_bc_emissions.legend(loc='upper left', fontsize='small')
    else:
        if show_lifetime_uncertainty:
            ax_bc_emissions.fill_between(time_array, min_cumulative_bc, max_cumulative_bc,
                           color='gray', alpha=0.2, label='Lifetime Uncertainty')
        ax_bc_emissions.plot(time_array, cumulative_bc_emissions, 
                color='tab:green', linewidth=2, label='Cumulative Black Carbon Emissions')
    
    # Format y-axis for better readability (e.g., using scientific notation or appropriate units)
    ax_bc_emissions.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    ax_bc_emissions.set_ylabel('Cumulative Black Carbon Emissions (g)')
    ax_bc_emissions.set_title('b) Black Carbon Emissions')
    ax_bc_emissions.grid(True, alpha=0.3)
    if show_lifetime_uncertainty and not breakdown_by:
        ax_bc_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-right: BC temperature responses with both lifetime uncertainty and proper combined uncertainty
    ax_bc_temp = axs[1, 1]

    bc_temp_color =  "#B82A0A"

    # First plot the individual components for clarity
    ax_bc_temp.plot(time_array, cumulative_dt_drf,
            color= "#EA8228", linewidth=2, linestyle = '--', label='Direct Radiative Forcing')
    ax_bc_temp.plot(time_array, cumulative_dt_snowrf,
            color= "#C32567", linewidth=2, linestyle = '--', label='Snow Radiative Forcing')
    
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
            color=bc_temp_color, linewidth=2, label='Direct + Snow Radiative Forcing')
    
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
    ax_bc_temp.set_title('d) Black Carbon Temperature Impact')
    ax_bc_temp.grid(True, alpha=0.3)
    ax_bc_temp.legend(loc='upper left', fontsize='small')

    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    return (cumulative_emissions, co2_temp_response, 
            cumulative_bc_emissions, cumulative_dt_drf, cumulative_dt_snowrf, 
            total_bc_temp_response, total_temp_response, max_temp_response, min_temp_response, min_bc_temp_response, max_bc_temp_response, min_combined, max_combined)

def plot_emissions_and_temperature_with_early_shutdown(CGP_df, shutdown_year=2030, start_year=2000, end_year=2060, 
                                                     shutdown_age=None, fraction_to_shutdown=None,
                                                     tcr=1.65, tcr_uncertainty=0.4, 
                                                     breakdown_by=None, show_lifetime_uncertainty=True, 
                                                     lifetime_uncertainty=5, figsize=(14, 10)):
    """
    Plot cumulative CO2 and BC emissions and temperature responses over time,
    with the option to shut down the oldest plants by a specified year.
    
    Parameters:
    -----------
    CGP_df : pandas.DataFrame
        DataFrame containing power plant data
    shutdown_year : int
        Year to shut down the oldest plants (default: 2030)
    start_year : int
        Starting year for the analysis (default: 2000)
    end_year : int
        Ending year for the analysis (default: 2060)
    shutdown_age : int, optional
        Age of plants to shut down (e.g., shut down all plants older than X years by shutdown_year)
        If None, uses fraction_to_shutdown instead
    fraction_to_shutdown : float, optional
        Fraction of oldest plants to shut down (0-1)
        Only used if shutdown_age is None
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
    
    Returns:
    --------
    tuple
        Contains arrays for emissions and temperature impacts for both baseline and early shutdown scenarios
    """

    tcr = np.round(float(tcr), 2)

    # Set up time array
    years = end_year - start_year
    time_array = np.arange(start_year, end_year)
    
    # Create a sorted copy of the dataframe to identify the oldest plants
    sorted_df = CGP_df.copy()
    sorted_df['plant_age_at_shutdown'] = shutdown_year - sorted_df['Year_of_Commission']
    
    # Determine which plants to shut down
    if shutdown_age is not None:
        # Shut down plants older than shutdown_age by the shutdown year
        plants_to_shutdown = sorted_df[sorted_df['plant_age_at_shutdown'] >= shutdown_age]['unique_ID'].values
    elif fraction_to_shutdown is not None:
        # Shut down the oldest fraction of plants
        sorted_df = sorted_df.sort_values('Year_of_Commission')
        num_plants_to_shutdown = int(len(sorted_df) * fraction_to_shutdown)
        plants_to_shutdown = sorted_df.head(num_plants_to_shutdown)['unique_ID'].values
    else:
        # Default: shut down plants older than 20 years by the shutdown year
        plants_to_shutdown = sorted_df[sorted_df['plant_age_at_shutdown'] >= 20]['unique_ID'].values
    
    # Initialize arrays for emissions and temperature responses (baseline scenario)
    baseline_emissions = np.zeros(years)
    baseline_bc_emissions = np.zeros(years)
    baseline_dt_drf = np.zeros(years)
    baseline_dt_snowrf = np.zeros(years)
    
    # Initialize arrays for emissions and temperature responses (shutdown scenario)
    shutdown_emissions = np.zeros(years)
    shutdown_bc_emissions = np.zeros(years)
    shutdown_dt_drf = np.zeros(years)
    shutdown_dt_snowrf = np.zeros(years)
    
    if show_lifetime_uncertainty:
        # Baseline scenario
        baseline_min_emissions = np.zeros(years)
        baseline_max_emissions = np.zeros(years)
        baseline_min_bc_emissions = np.zeros(years)
        baseline_max_bc_emissions = np.zeros(years)
        baseline_min_dt_drf = np.zeros(years)
        baseline_max_dt_drf = np.zeros(years)
        baseline_min_dt_snowrf = np.zeros(years)
        baseline_max_dt_snowrf = np.zeros(years)
        
        # Shutdown scenario
        shutdown_min_emissions = np.zeros(years)
        shutdown_max_emissions = np.zeros(years)
        shutdown_min_bc_emissions = np.zeros(years)
        shutdown_max_bc_emissions = np.zeros(years)
        shutdown_min_dt_drf = np.zeros(years)
        shutdown_max_dt_drf = np.zeros(years)
        shutdown_min_dt_snowrf = np.zeros(years)
        shutdown_max_dt_snowrf = np.zeros(years)
    
    # If breaking down emissions, create separate arrays for each category
    if breakdown_by:
        categories = CGP_df[breakdown_by].unique()
        
        # Baseline scenario
        baseline_category_emissions = {cat: np.zeros(years) for cat in categories}
        baseline_category_bc_emissions = {cat: np.zeros(years) for cat in categories}
        baseline_category_dt_drf = {cat: np.zeros(years) for cat in categories}
        baseline_category_dt_snowrf = {cat: np.zeros(years) for cat in categories}
        
        # Shutdown scenario
        shutdown_category_emissions = {cat: np.zeros(years) for cat in categories}
        shutdown_category_bc_emissions = {cat: np.zeros(years) for cat in categories}
        shutdown_category_dt_drf = {cat: np.zeros(years) for cat in categories}
        shutdown_category_dt_snowrf = {cat: np.zeros(years) for cat in categories}
        
        if show_lifetime_uncertainty:
            # Baseline scenario
            baseline_category_min_emissions = {cat: np.zeros(years) for cat in categories}
            baseline_category_max_emissions = {cat: np.zeros(years) for cat in categories}
            baseline_category_min_bc_emissions = {cat: np.zeros(years) for cat in categories}
            baseline_category_max_bc_emissions = {cat: np.zeros(years) for cat in categories}
            baseline_category_min_dt_drf = {cat: np.zeros(years) for cat in categories}
            baseline_category_max_dt_drf = {cat: np.zeros(years) for cat in categories}
            baseline_category_min_dt_snowrf = {cat: np.zeros(years) for cat in categories}
            baseline_category_max_dt_snowrf = {cat: np.zeros(years) for cat in categories}
            
            # Shutdown scenario
            shutdown_category_min_emissions = {cat: np.zeros(years) for cat in categories}
            shutdown_category_max_emissions = {cat: np.zeros(years) for cat in categories}
            shutdown_category_min_bc_emissions = {cat: np.zeros(years) for cat in categories}
            shutdown_category_max_bc_emissions = {cat: np.zeros(years) for cat in categories}
            shutdown_category_min_dt_drf = {cat: np.zeros(years) for cat in categories}
            shutdown_category_max_dt_drf = {cat: np.zeros(years) for cat in categories}
            shutdown_category_min_dt_snowrf = {cat: np.zeros(years) for cat in categories}
            shutdown_category_max_dt_snowrf = {cat: np.zeros(years) for cat in categories}
    else:
        categories = None
        baseline_category_emissions = None
        shutdown_category_emissions = None

    # Calculate the shutdown year offset
    shutdown_year_offset = shutdown_year - start_year
    
    # Process each plant
    for unique_id in CGP_df['unique_ID'].values:
        # Get plant data
        plant_data = CGP_df.loc[CGP_df['unique_ID'] == unique_id]
        
        # Get annual values
        annual_co2 = float(plant_data['co2_emissions']) / 1e9  # Convert to GtCO2
        annual_bc = float(plant_data['BC_(g/yr)'])  # BC emissions in g/yr
        annual_dt_drf = float(plant_data['dt_drf'])
        annual_dt_snowrf = float(plant_data['dt_snowrf'])
        
        # Check if this plant is in the shutdown list
        is_shutdown_plant = unique_id in plants_to_shutdown
        
        # Add to baseline emissions (all plants operate until end of life)
        yr_offset = int(plant_data['Year_of_Commission'].iloc[0] - start_year)
        if yr_offset >= 0:
            # Standard operating life (40 years or until end of simulation)
            operating_years = int(min(40, end_year - plant_data['Year_of_Commission'].iloc[0]))
            
            if operating_years > 0:
                # Add values for each operating year (baseline scenario)
                for yr in range(operating_years):
                    if yr_offset + yr < years:
                        baseline_emissions[yr_offset + yr] += annual_co2
                        baseline_bc_emissions[yr_offset + yr] += annual_bc
                        baseline_dt_drf[yr_offset + yr] += annual_dt_drf
                        baseline_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                        
                        if breakdown_by:
                            category = plant_data[breakdown_by].iloc[0]
                            baseline_category_emissions[category][yr_offset + yr] += annual_co2
                            baseline_category_bc_emissions[category][yr_offset + yr] += annual_bc
                            baseline_category_dt_drf[category][yr_offset + yr] += annual_dt_drf
                            baseline_category_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                
                # Add values for each operating year (shutdown scenario)
                # If plant is to be shutdown early, only operate until shutdown year
                shutdown_operating_years = operating_years
                if is_shutdown_plant and shutdown_year_offset > yr_offset:
                    shutdown_operating_years = min(operating_years, shutdown_year_offset - yr_offset)
                
                for yr in range(shutdown_operating_years):
                    if yr_offset + yr < years:
                        shutdown_emissions[yr_offset + yr] += annual_co2
                        shutdown_bc_emissions[yr_offset + yr] += annual_bc
                        shutdown_dt_drf[yr_offset + yr] += annual_dt_drf
                        shutdown_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                        
                        if breakdown_by:
                            category = plant_data[breakdown_by].iloc[0]
                            shutdown_category_emissions[category][yr_offset + yr] += annual_co2
                            shutdown_category_bc_emissions[category][yr_offset + yr] += annual_bc
                            shutdown_category_dt_drf[category][yr_offset + yr] += annual_dt_drf
                            shutdown_category_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                
                # Calculate lifetime uncertainty bounds
                if show_lifetime_uncertainty:
                    # Baseline scenario
                    baseline_min_operating_years = int(max(0, operating_years - lifetime_uncertainty))
                    baseline_max_operating_years = int(min(operating_years + lifetime_uncertainty, 
                                                   end_year - plant_data['Year_of_Commission'].iloc[0]))
                    
                    # Shutdown scenario
                    shutdown_min_operating_years = baseline_min_operating_years
                    shutdown_max_operating_years = baseline_max_operating_years
                    if is_shutdown_plant and shutdown_year_offset > yr_offset:
                        shutdown_min_operating_years = min(shutdown_min_operating_years, shutdown_year_offset - yr_offset)
                        shutdown_max_operating_years = min(shutdown_max_operating_years, shutdown_year_offset - yr_offset)
                    
                    # Minimum lifetime - baseline
                    for yr in range(baseline_min_operating_years):
                        if yr_offset + yr < years:
                            baseline_min_emissions[yr_offset + yr] += annual_co2
                            baseline_min_bc_emissions[yr_offset + yr] += annual_bc
                            baseline_min_dt_drf[yr_offset + yr] += annual_dt_drf
                            baseline_min_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                baseline_category_min_emissions[category][yr_offset + yr] += annual_co2
                                baseline_category_min_bc_emissions[category][yr_offset + yr] += annual_bc
                                baseline_category_min_dt_drf[category][yr_offset + yr] += annual_dt_drf
                                baseline_category_min_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                    
                    # Maximum lifetime - baseline
                    for yr in range(baseline_max_operating_years):
                        if yr_offset + yr < years:
                            baseline_max_emissions[yr_offset + yr] += annual_co2
                            baseline_max_bc_emissions[yr_offset + yr] += annual_bc
                            baseline_max_dt_drf[yr_offset + yr] += annual_dt_drf
                            baseline_max_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                baseline_category_max_emissions[category][yr_offset + yr] += annual_co2
                                baseline_category_max_bc_emissions[category][yr_offset + yr] += annual_bc
                                baseline_category_max_dt_drf[category][yr_offset + yr] += annual_dt_drf
                                baseline_category_max_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                    
                    # Minimum lifetime - shutdown
                    for yr in range(shutdown_min_operating_years):
                        if yr_offset + yr < years:
                            shutdown_min_emissions[yr_offset + yr] += annual_co2
                            shutdown_min_bc_emissions[yr_offset + yr] += annual_bc
                            shutdown_min_dt_drf[yr_offset + yr] += annual_dt_drf
                            shutdown_min_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                shutdown_category_min_emissions[category][yr_offset + yr] += annual_co2
                                shutdown_category_min_bc_emissions[category][yr_offset + yr] += annual_bc
                                shutdown_category_min_dt_drf[category][yr_offset + yr] += annual_dt_drf
                                shutdown_category_min_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf
                    
                    # Maximum lifetime - shutdown
                    for yr in range(shutdown_max_operating_years):
                        if yr_offset + yr < years:
                            shutdown_max_emissions[yr_offset + yr] += annual_co2
                            shutdown_max_bc_emissions[yr_offset + yr] += annual_bc
                            shutdown_max_dt_drf[yr_offset + yr] += annual_dt_drf
                            shutdown_max_dt_snowrf[yr_offset + yr] += annual_dt_snowrf
                            if breakdown_by:
                                category = plant_data[breakdown_by].iloc[0]
                                shutdown_category_max_emissions[category][yr_offset + yr] += annual_co2
                                shutdown_category_max_bc_emissions[category][yr_offset + yr] += annual_bc
                                shutdown_category_max_dt_drf[category][yr_offset + yr] += annual_dt_drf
                                shutdown_category_max_dt_snowrf[category][yr_offset + yr] += annual_dt_snowrf

    # Calculate cumulative values - baseline
    baseline_cumulative_emissions = np.cumsum(baseline_emissions)
    baseline_cumulative_bc_emissions = np.cumsum(baseline_bc_emissions)
    baseline_cumulative_dt_drf = np.cumsum(baseline_dt_drf)
    baseline_cumulative_dt_snowrf = np.cumsum(baseline_dt_snowrf)
    
    # Calculate cumulative values - shutdown
    shutdown_cumulative_emissions = np.cumsum(shutdown_emissions)
    shutdown_cumulative_bc_emissions = np.cumsum(shutdown_bc_emissions)
    shutdown_cumulative_dt_drf = np.cumsum(shutdown_dt_drf)
    shutdown_cumulative_dt_snowrf = np.cumsum(shutdown_dt_snowrf)
    
    if show_lifetime_uncertainty:
        # Baseline scenario
        baseline_min_cumulative = np.cumsum(baseline_min_emissions)
        baseline_max_cumulative = np.cumsum(baseline_max_emissions)
        baseline_min_cumulative_bc = np.cumsum(baseline_min_bc_emissions)
        baseline_max_cumulative_bc = np.cumsum(baseline_max_bc_emissions)
        baseline_min_cumulative_dt_drf = np.cumsum(baseline_min_dt_drf)
        baseline_max_cumulative_dt_drf = np.cumsum(baseline_max_dt_drf)
        baseline_min_cumulative_dt_snowrf = np.cumsum(baseline_min_dt_snowrf)
        baseline_max_cumulative_dt_snowrf = np.cumsum(baseline_max_dt_snowrf)
        
        # Shutdown scenario
        shutdown_min_cumulative = np.cumsum(shutdown_min_emissions)
        shutdown_max_cumulative = np.cumsum(shutdown_max_emissions)
        shutdown_min_cumulative_bc = np.cumsum(shutdown_min_bc_emissions)
        shutdown_max_cumulative_bc = np.cumsum(shutdown_max_bc_emissions)
        shutdown_min_cumulative_dt_drf = np.cumsum(shutdown_min_dt_drf)
        shutdown_max_cumulative_dt_drf = np.cumsum(shutdown_max_dt_drf)
        shutdown_min_cumulative_dt_snowrf = np.cumsum(shutdown_min_dt_snowrf)
        shutdown_max_cumulative_dt_snowrf = np.cumsum(shutdown_max_dt_snowrf)
    
    # Calculate temperature responses - baseline
    baseline_co2_temp_response = baseline_cumulative_emissions * (tcr / 1000)
    baseline_co2_temp_upper = baseline_cumulative_emissions * ((tcr + tcr_uncertainty) / 1000)
    baseline_co2_temp_lower = baseline_cumulative_emissions * ((tcr - tcr_uncertainty) / 1000)
    
    # Calculate temperature responses - shutdown
    shutdown_co2_temp_response = shutdown_cumulative_emissions * (tcr / 1000)
    shutdown_co2_temp_upper = shutdown_cumulative_emissions * ((tcr + tcr_uncertainty) / 1000)
    shutdown_co2_temp_lower = shutdown_cumulative_emissions * ((tcr - tcr_uncertainty) / 1000)
    
    # Calculate total BC temperature response - baseline
    baseline_bc_temp_response = baseline_cumulative_dt_drf + baseline_cumulative_dt_snowrf
    if show_lifetime_uncertainty:
        baseline_bc_temp_upper = baseline_max_cumulative_dt_drf + baseline_max_cumulative_dt_snowrf
        baseline_bc_temp_lower = baseline_min_cumulative_dt_drf + baseline_min_cumulative_dt_snowrf
    
    # Calculate total BC temperature response - shutdown
    shutdown_bc_temp_response = shutdown_cumulative_dt_drf + shutdown_cumulative_dt_snowrf
    if show_lifetime_uncertainty:
        shutdown_bc_temp_upper = shutdown_max_cumulative_dt_drf + shutdown_max_cumulative_dt_snowrf
        shutdown_bc_temp_lower = shutdown_min_cumulative_dt_drf + shutdown_min_cumulative_dt_snowrf
    
    # Total temperature response (CO2 + BC) - baseline
    baseline_total_temp_response = baseline_co2_temp_response + baseline_bc_temp_response
    if show_lifetime_uncertainty:
        baseline_total_temp_upper = baseline_co2_temp_upper + baseline_bc_temp_upper
        baseline_total_temp_lower = baseline_co2_temp_lower + baseline_bc_temp_lower
    
    # Total temperature response (CO2 + BC) - shutdown
    shutdown_total_temp_response = shutdown_co2_temp_response + shutdown_bc_temp_response
    if show_lifetime_uncertainty:
        shutdown_total_temp_upper = shutdown_co2_temp_upper + shutdown_bc_temp_upper
        shutdown_total_temp_lower = shutdown_co2_temp_lower + shutdown_bc_temp_lower

    # Create the 2x2 plot layout
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True)
    
    # ----- Left side (CO2) -----
    # Top-left: CO2 emissions
    ax_co2_emissions = axs[0, 0]
    
    # Plot baseline emissions
    ax_co2_emissions.plot(time_array, baseline_cumulative_emissions, 
                        color='tab:blue', linewidth=2, linestyle='--', 
                        label='Baseline Scenario')
    
    # Plot shutdown emissions
    ax_co2_emissions.plot(time_array, shutdown_cumulative_emissions, 
                        color='tab:blue', linewidth=2, 
                        label=f'Early Shutdown ({shutdown_year})')
    
    # Add uncertainty if enabled
    if show_lifetime_uncertainty:
        # Baseline uncertainty
        ax_co2_emissions.fill_between(time_array, baseline_min_cumulative, baseline_max_cumulative,
                                    color='tab:blue', alpha=0.1)
        
        # Shutdown uncertainty
        ax_co2_emissions.fill_between(time_array, shutdown_min_cumulative, shutdown_max_cumulative,
                                    color='tab:blue', alpha=0.2)
    
    # Add vertical line at shutdown year
    ax_co2_emissions.axvline(x=shutdown_year, color='red', linestyle='--', alpha=0.7)
    
    ax_co2_emissions.set_ylabel(r'Cumulative CO$_2$ Emissions (GtCO$_2$)')
    ax_co2_emissions.set_title(r'CO$_2$ Emissions')
    ax_co2_emissions.grid(True, alpha=0.3)
    ax_co2_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-left: CO2 temperature response
    ax_co2_temp = axs[1, 0]
    
    # Plot baseline temperature response
    ax_co2_temp.plot(time_array, baseline_co2_temp_response,
                   color='tab:red', linewidth=2, linestyle='--',
                   label='Baseline Scenario')
    
    # Plot shutdown temperature response
    ax_co2_temp.plot(time_array, shutdown_co2_temp_response,
                   color='tab:red', linewidth=2,
                   label=f'Early Shutdown ({shutdown_year})')
    
    # Add uncertainty if enabled
    if show_lifetime_uncertainty:
        # Baseline uncertainty (lifetime + TCR)
        baseline_min_combined = baseline_min_cumulative * ((tcr - tcr_uncertainty) / 1000)
        baseline_max_combined = baseline_max_cumulative * ((tcr + tcr_uncertainty) / 1000)
        
        ax_co2_temp.fill_between(time_array, baseline_min_combined, baseline_max_combined,
                               color='tab:red', alpha=0.1)
        
        # Shutdown uncertainty (lifetime + TCR)
        shutdown_min_combined = shutdown_min_cumulative * ((tcr - tcr_uncertainty) / 1000)
        shutdown_max_combined = shutdown_max_cumulative * ((tcr + tcr_uncertainty) / 1000)
        
        ax_co2_temp.fill_between(time_array, shutdown_min_combined, shutdown_max_combined,
                               color='tab:red', alpha=0.2)
    
    # Add vertical line at shutdown year
    ax_co2_temp.axvline(x=shutdown_year, color='red', linestyle='--', alpha=0.7)
    
    ax_co2_temp.set_xlabel('Year')
    ax_co2_temp.set_ylabel('Temperature Response (°C)')
    ax_co2_temp.set_title(r'CO$_2$ Temperature Impact')
    ax_co2_temp.grid(True, alpha=0.3)
    ax_co2_temp.legend(loc='upper left', fontsize='small')
    
    # ----- Right side (BC) -----
    # Top-right: BC emissions
    ax_bc_emissions = axs[0, 1]
    
    # Plot baseline BC emissions
    ax_bc_emissions.plot(time_array, baseline_cumulative_bc_emissions, 
                       color='tab:green', linewidth=2, linestyle='--',
                       label='Baseline Scenario')
    
    # Plot shutdown BC emissions
    ax_bc_emissions.plot(time_array, shutdown_cumulative_bc_emissions, 
                       color='tab:green', linewidth=2,
                       label=f'Early Shutdown ({shutdown_year})')
    
    # Add uncertainty if enabled
    if show_lifetime_uncertainty:
        # Baseline uncertainty
        ax_bc_emissions.fill_between(time_array, baseline_min_cumulative_bc, baseline_max_cumulative_bc,
                                   color='tab:green', alpha=0.1)
        
        # Shutdown uncertainty
        ax_bc_emissions.fill_between(time_array, shutdown_min_cumulative_bc, shutdown_max_cumulative_bc,
                                   color='tab:green', alpha=0.2)
    
    # Add vertical line at shutdown year
    ax_bc_emissions.axvline(x=shutdown_year, color='red', linestyle='--', alpha=0.7)
    
    # Format y-axis for better readability
    ax_bc_emissions.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    ax_bc_emissions.set_ylabel('Cumulative Black Carbon Emissions (g)')
    ax_bc_emissions.set_title('Black Carbon Emissions')
    ax_bc_emissions.grid(True, alpha=0.3)
    ax_bc_emissions.legend(loc='upper left', fontsize='small')
    
    # Bottom-right: BC temperature responses
    ax_bc_temp = axs[1, 1]
    
    # Plot baseline BC temperature components
    ax_bc_temp.plot(time_array, baseline_cumulative_dt_drf,
                  color='tab:olive', linewidth=1, linestyle='--',
                  label='Baseline BC DRF')
    ax_bc_temp.plot(time_array, baseline_cumulative_dt_snowrf,
                  color='tab:purple', linewidth=1, linestyle='--',
                  label='Baseline BC Snow RF')
    
    # Plot shutdown BC temperature components
    ax_bc_temp.plot(time_array, shutdown_cumulative_dt_drf,
                  color='tab:olive', linewidth=1,
                  label='Shutdown BC DRF')
    ax_bc_temp.plot(time_array, shutdown_cumulative_dt_snowrf,
                  color='tab:purple', linewidth=1,
                  label='Shutdown BC Snow RF')
    
    # Plot total BC temperature impact
    ax_bc_temp.plot(time_array, baseline_bc_temp_response,
                  color='tab:blue', linewidth=2, linestyle='--',
                  label='Baseline Total BC')
    ax_bc_temp.plot(time_array, shutdown_bc_temp_response,
                  color='tab:blue', linewidth=2,
                  label='Shutdown Total BC')
    
    # Add uncertainty if enabled
    if show_lifetime_uncertainty:
        # Baseline uncertainty
        ax_bc_temp.fill_between(time_array, baseline_bc_temp_lower, baseline_bc_temp_upper,
                              color='tab:blue', alpha=0.1)
        
        # Shutdown uncertainty
        ax_bc_temp.fill_between(time_array, shutdown_bc_temp_lower, shutdown_bc_temp_upper,
                              color='tab:blue', alpha=0.2)
    
    # Add vertical line at shutdown year
    ax_bc_temp.axvline(x=shutdown_year, color='red', linestyle='--', alpha=0.7)
    
    ax_bc_temp.set_xlabel('Year')
    ax_bc_temp.set_ylabel('Cumulative Temperature Response (°C)')
    ax_bc_temp.set_title('BC Temperature Impact')
    ax_bc_temp.grid(True, alpha=0.3)
    ax_bc_temp.legend(loc='upper left', fontsize='small')
    
    # Add a figure title
    plt.suptitle(f'Climate Impacts of Early Coal Plant Shutdown ({shutdown_year})', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    return {
        'baseline': {
            'co2_emissions': baseline_cumulative_emissions,
            'bc_emissions': baseline_cumulative_bc_emissions,
            'co2_temp': baseline_co2_temp_response,
            'bc_temp_drf': baseline_cumulative_dt_drf,
            'bc_temp_snow': baseline_cumulative_dt_snowrf,
            'bc_temp_total': baseline_bc_temp_response,
            'total_temp': baseline_total_temp_response
        },
        'shutdown': {
            'co2_emissions': shutdown_cumulative_emissions,
            'bc_emissions': shutdown_cumulative_bc_emissions,
            'co2_temp': shutdown_co2_temp_response,
            'bc_temp_drf': shutdown_cumulative_dt_drf,
            'bc_temp_snow': shutdown_cumulative_dt_snowrf,
            'bc_temp_total': shutdown_bc_temp_response,
            'total_temp': shutdown_total_temp_response
        },
        'avoided': {
            'co2_emissions': baseline_cumulative_emissions - shutdown_cumulative_emissions,
            'bc_emissions': baseline_cumulative_bc_emissions - shutdown_cumulative_bc_emissions,
            'co2_temp': baseline_co2_temp_response - shutdown_co2_temp_response,
            'bc_temp_total': baseline_bc_temp_response - shutdown_bc_temp_response,
            'total_temp': baseline_total_temp_response - shutdown_total_temp_response
        }
    }



def compare_individual_strategies_and_rates(scenario_ds, rates=[1.0, 2.0, 4.0], 
                                        force_closure_by=2040,
                                        strategies=['Year_of_Commission', 'MW', 'CO2_weighted_capacity_1000tonsperMW'],
                                        impact_var='BC_pop_weight_mean_conc',
                                        impacted_country='Vietnam',
                                        country='VIETNAM',
                                        figsize=(18, 12)):
    """
    Compare different retirement strategies and rates for Vietnam's power plants in one comprehensive plot.
    Shows all strategies side by side with forced closure by a specified year.
    
    Parameters:
    -----------
    scenario_ds : xarray.Dataset
        Dataset containing power plant data
    rates : list
        List of retirement rates to compare (plants per year)
    force_closure_by : int
        Year by which all plants must be closed, regardless of rate
    strategies : list
        List of strategies to compare (e.g., 'Year_of_Commission', 'MW')
    impact_var : str
        Variable to analyze (e.g., 'BC_pop_weight_mean_conc', 'co2_emissions')
    impacted_country : str
        Country receiving impacts to analyze
    country : str
        Source country for emissions (default: 'VIETNAM')
    figsize : tuple
        Figure size for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the comparison plots
    axs : matplotlib.axes.Axes
        Axes of the figure
    co2_data : dict
        Dictionary containing CO2 emissions data for each strategy and rate
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create figure with 3 rows (capacity, BC concentration, cumulative CO2)
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Strategy names and styles
    strategy_names = {
        'Year_of_Commission': 'Oldest First',
        'MW': 'Largest First', 
        'EMISFACTOR.CO2': 'Most Polluting First'
    }
    
    strategy_styles = {
        'Year_of_Commission': {'linestyle': '-'},
        'MW': {'linestyle': '--'}, 
        'EMISFACTOR.CO2': {'linestyle': ':'}
    }
    
    # Create colormap for rates
    colors = plt.cm.Paired(np.linspace(0, 1, len(rates)))
    
    # Dictionary to store CO2 data
    co2_data = {}
    
    # Get the Vietnam dataset
    vietnam_ds = scenario_ds.where(scenario_ds.country_emitting == country, drop=True)
    
    # Get total number of plants
    total_plants = len(vietnam_ds.unique_ID)
    
    # Define timeline from 2025 to force_closure_by
    timeline = np.arange(2025, force_closure_by + 1)
    
    # For each strategy
    for s_idx, strategy in enumerate(strategies):
        # Create dict to store data for this strategy
        co2_data[strategy] = {}
        
        # Sort plants by strategy
        if strategy == 'Year_of_Commission':
            sorted_ds = vietnam_ds.sortby(strategy)  # Ascending for age (oldest first)
        else:
            sorted_ds = vietnam_ds.sortby(strategy, ascending=False)  # Descending for size/emissions
        
        # Generate trajectories for different retirement rates
        for r_idx, rate in enumerate(rates):
            # Calculate how many years it takes to retire all plants at this rate
            years_to_retire = int(np.ceil(total_plants / rate))
            
            # Track capacity, BC concentration, and CO2 emissions over time
            capacity = np.zeros(len(timeline))
            bc_conc = np.zeros(len(timeline))
            co2_emis = np.zeros(len(timeline))
            cumulative_co2 = np.zeros(len(timeline))
            
            # Initialize with all plants operating
            capacity[0] = sorted_ds['MW'].sum().values / 1000  # Convert to GW
            
            # Get BC concentration for the impacted country
            # Handle impacted_country parameter - convert string to list for consistent processing
            if isinstance(impacted_country, str):
                impacted_countries = [impacted_country]
            else:
                impacted_countries = list(impacted_country)
            
            # Get BC concentration for the impacted country/countries
            bc_conc[0] = 0  # Initialize
            for imp_country in impacted_countries:
                if imp_country in scenario_ds.country_impacted.values:
                    print(timeline)
                    bc_data = sorted_ds.sel(country_impacted=imp_country).sel(scenario_year = timeline[0])[impact_var]
                    bc_conc[0] += bc_data.sum().values

  
            # Get CO2 emissions
            co2_emis[0] = sorted_ds['co2_emissions'].sel(scenario_year = timeline[0]).sum().values / 1e9  # Convert to giga tons
            cumulative_co2[0] = co2_emis[0]
            
            # Simulate retirement over time
            plants_left = total_plants
            
            for i, year in enumerate(timeline[1:], 1):
                # Calculate years until forced closure
                years_until_force_closure = force_closure_by - year
                
                # If we're approaching the force_closure_by year, calculate plants that must be retired
                if years_until_force_closure > 0:
                    plants_to_retire_this_year = min(rate, plants_left)
                else:
                    # Force all remaining plants to close in the final year
                    plants_to_retire_this_year = plants_left
                
                # Retire plants
                plants_left -= plants_to_retire_this_year
                
                # Get remaining plants
                remaining_ds = sorted_ds.isel(unique_ID=slice(int(total_plants - plants_left), total_plants))
                
                # Calculate remaining capacity
                capacity[i] = remaining_ds['MW'].sum().values / 1000 if plants_left > 0 else 0
                
                # Calculate remaining BC concentration
                bc_conc[i] = 0  # Initialize
                if plants_left > 0:
                    for imp_country in impacted_countries:
                        if imp_country in scenario_ds.country_impacted.values:
                            bc_data = remaining_ds.sel(country_impacted=imp_country).sel(scenario_year = year)[impact_var]
                            bc_conc[i] += bc_data.sum().values
                
                # Calculate remaining CO2 emissions
                co2_emis[i] = remaining_ds['co2_emissions'].sel(scenario_year = year).sum().values / 1e9 if plants_left > 0 else 0
                
                # Calculate cumulative CO2 emissions
                cumulative_co2[i] = cumulative_co2[i-1] + co2_emis[i]
            
            # Store CO2 data
            co2_data[strategy][rate] = {
                'years': timeline,
                'co2_emissions': co2_emis,
                'cumulative_co2': cumulative_co2,
                'bc_concentration': bc_conc,
                'capacity': capacity
            }
            
            # Combine strategy style with rate color
            line_style = strategy_styles[strategy]['linestyle']
            color = colors[r_idx]
            
            # Construct label - only add full label for first strategy to avoid duplicate legends
            if s_idx == 0:
                label = f"{rate:.1f} plants/year, {strategy_names[strategy]}"
            else:
                label = f"{strategy_names[strategy]}"
            
            # Plot capacity (top row)
            axs[0].plot(timeline, capacity, color=color, linestyle=line_style, 
                        label=label, linewidth=2)
            
            # Plot BC concentration (middle row)
            axs[1].plot(timeline, bc_conc, color=color, linestyle=line_style, linewidth=2)
            
            # Plot cumulative CO2 (bottom row)
            axs[2].plot(timeline, cumulative_co2, color=color, linestyle=line_style, linewidth=2)
 
    # Add grid to all plots
    for row in range(3):
        axs[row].grid(True, linestyle='--', alpha=0.5)
    
    axs[0].set_ylabel('Capacity (GW)')
    axs[1].set_ylabel(f'Black Carbon Concentration (ng/m³)')
    axs[2].set_ylabel('Cumulative CO₂ (Gt)')
    axs[2].set_xlabel('Year')

    # Create legend elements for rates (colors)
    rate_legend_elements = [Line2D([0], [0], color=colors[i], lw=2, 
                                  label=f"{rate:.1f} plants/year")
                           for i, rate in enumerate(rates)]
    
    # Create legend elements for strategies (line styles)
    strategy_legend_elements = [Line2D([0], [0], color='black', lw=2, 
                                      linestyle=strategy_styles[s]['linestyle'],
                                      label=strategy_names[s])
                               for s in strategies]
    
    # Add both legends
    fig.legend(handles=rate_legend_elements, loc='upper right', title="Retirement Rate", bbox_to_anchor=(1.1, 0.5),)
    fig.legend(handles=strategy_legend_elements, loc='upper right', title="Strategy", bbox_to_anchor=(1.11, 0.28),)
    
    plt.tight_layout()
    return fig, axs, co2_data


def compare_multi_country_strategies_and_rates(scenario_ds, countries=['MALAYSIA', 'INDONESIA', 'VIETNAM'],
                                              rates=[1.0, 2.0, 4.0], 
                                              force_closure_by={'MALAYSIA': 2045, 'CAMBODIA': 2050, 'INDONESIA': 2040, 'VIETNAM': 2050},
                                              strategies=['Year_of_Commission', 'MW', 'CO2_weighted_capacity_1000tonsperMW'],
                                              impact_var='BC_pop_weight_mean_conc',
                                              impacted_country='China',
                                              figsize=(20, 12)):
    """
    Compare different retirement strategies and rates for multiple countries side by side in a 3x3 grid.
    Each column represents a country, and each row represents a different metric.
    
    Parameters:
    -----------
    scenario_ds : xarray.Dataset
        Dataset containing power plant data
    countries : list
        List of countries to compare (default: ['MALAYSIA', 'INDONESIA', 'VIETNAM'])
    rates : list or dict
        Retirement rates to compare. Can be:
        - List: Same rates for all countries (e.g., [1.0, 2.0, 4.0])
        - Dict: Country-specific rates (e.g., {'MALAYSIA': [1.0, 2.0], 'INDONESIA': [2.0, 4.0], 'VIETNAM': [1.0, 3.0]})
    force_closure_by : dict
        Year by which all plants must be closed, regardless of rate, dictionary
    strategies : list
        List of strategies to compare (e.g., 'Year_of_Commission', 'MW')
    impact_var : str
        Variable to analyze (e.g., 'BC_pop_weight_mean_conc', 'co2_emissions')
    impacted_country : str or list
        Country/countries receiving impacts to analyze. If a list is provided, impacts will be summed across all countries.
    figsize : tuple
        Figure size for the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the comparison plots
    axs : numpy.ndarray
        2D array of axes (3 rows x 3 columns)
    country_data : dict
        Dictionary containing data for each country, strategy, and rate
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Handle rates parameter - convert to dictionary format for consistent processing
    if isinstance(rates, dict):
        country_rates = rates
        # Get all unique rates across countries for color mapping
        all_rates = []
        for country_rate_list in country_rates.values():
            all_rates.extend(country_rate_list)
        unique_rates = all_rates #sorted(list((all_rates)))
    else:
        # If rates is a list, apply same rates to all countries
        country_rates = {country: rates for country in countries}
        unique_rates = sorted(rates)
    print(unique_rates)
    # Handle impacted_country parameter - convert string to list for consistent processing
    if isinstance(impacted_country, str):
        impacted_countries = [impacted_country]
    else:
        impacted_countries = list(impacted_country)
    
    # Create figure with 3 rows (capacity, BC concentration, cumulative CO2) and 3 columns (countries)
    # Don't share x-axis since countries have different timelines
    fig, axs = plt.subplots(3, len(countries), figsize=figsize)#, sharey='row')
    
    # Strategy names and styles
    strategy_names = {
        'Year_of_Commission': 'Oldest First',
        'MW': 'Largest First', 
        'EMISFACTOR.CO2': r'Highest CO$_2$ Intensity First'
    }
    
    strategy_styles = {
        'Year_of_Commission': {'linestyle': '-'},
        'MW': {'linestyle': '--'}, 
        'EMISFACTOR.CO2': {'linestyle': ':'}
    }
    
    # Create colormap for rates based on unique rates across all countries
    #colors = plt.cm.tab10(np.linspace(0, 1, len(unique_rates)))
    colors = [
         "#007336",   # Dark Teal Blue
    '#E66F20',   # Rust Orange
    "#7CC8FF",    # Pale Yellow
    '#0F8B8D',   # Dark Teal
    '#E07C10',   # Burnt Orange
    "#52B0FD",   # Deep Yellow
    "#0B7F73",   # Aqua
    '#D85E0D',   # Dark Orange
    "#179DFC",   # Mustard Yellow
]
    # Create rate-to-color mapping
    # Dictionary to store all data
    country_data = {}
    rate_color_map = {}
    timeline = {}
    # Process each country
    mult_val = 0
    names_dict = {0: 'Ambitious', 1: 'On-time', 2: 'Slow'}
    for col_idx, country in enumerate(countries):
        
        rate_color_map[country] = {rate: colors[i+mult_val*3] for i, rate in enumerate(rates[country])}

        # Define timeline from 2025 to force_closure_by
        timeline[country] = np.arange(2025, force_closure_by[country] + 1)

        # Initialize country data dictionary
        country_data[country] = {}
        
        # Get the country dataset
        country_ds = scenario_ds.where(scenario_ds.country_emitting == country, drop=True)
        
        # Get total number of plants for this country
        total_plants = len(country_ds.unique_ID)
        
        # For each strategy
        for s_idx, strategy in enumerate(strategies):
            # Create dict to store data for this strategy
            country_data[country][strategy] = {}
            
            # Sort plants by strategy
            if strategy == 'Year_of_Commission':
                sorted_ds = country_ds.sortby(strategy)  # oldest first
                if sorted_ds[strategy][0] >= sorted_ds[strategy][1]:
                    print(f"Warning: not in ascending age order for {country} with strategy {strategy}")
            else:
                sorted_ds = country_ds.sortby(strategy, ascending=False)  # Biggest first
                if sorted_ds[strategy][0] < sorted_ds[strategy][1]:
                    print(f"Warning: not in descending order for {country} with strategy {strategy}")
            # Generate trajectories for different retirement rates for this country
            for r_idx, rate in enumerate(country_rates[country]):
                # Calculate how many years it takes to retire all plants at this rate
                years_to_retire = int(np.ceil(total_plants / rate))
                
                # Track capacity, BC concentration, and CO2 emissions over time
                capacity = np.zeros(len(timeline[country]))
                bc_conc = np.zeros(len(timeline[country]))
                co2_emis = np.zeros(len(timeline[country]))
                cumulative_co2 = np.zeros(len(timeline[country]))
                
                # Initialize with all plants operating
                capacity[0] = sorted_ds['MW'].sum().values / 1000  # Convert to GW

                # Get BC concentration for the impacted country/countries
                bc_conc[0] = 0  # Initialize
                for imp_country in impacted_countries:
                    if imp_country in scenario_ds.country_impacted.values:
                        bc_data = sorted_ds.sel(country_impacted=imp_country).sel(scenario_year = timeline[country][0])[impact_var]
                        bc_conc[0] += bc_data.sum().values

                # Get CO2 emissions
                co2_emis[0] = sorted_ds['co2_emissions'].sel(scenario_year = timeline[country][0]).sum().values / 1e9  # Convert to giga tons
                cumulative_co2[0] = co2_emis[0]
                
                # Simulate retirement over time
                plants_left = total_plants
                
                for i, year in enumerate(timeline[country][1:], 1):
                    # Calculate years until forced closure
                    years_until_force_closure = force_closure_by[country] - year

                    # If we're approaching the force_closure_by year, calculate plants that must be retired
                    if years_until_force_closure > 0:
                        plants_to_retire_this_year = min(rate, plants_left)
                    else:
                        # Force all remaining plants to close in the final year
                        plants_to_retire_this_year = plants_left
                    
                    # Retire plants
                    plants_left -= plants_to_retire_this_year
                    
                    # Get remaining plants
                    remaining_ds = sorted_ds.isel(unique_ID=slice(int(total_plants - plants_left), total_plants))
                    
                    # Calculate remaining capacity
                    capacity[i] = remaining_ds['MW'].sum().values / 1000 if plants_left > 0 else 0
                    
                    # Calculate remaining BC concentration
                    bc_conc[i] = 0  # Initialize
                    if plants_left > 0:
                        for imp_country in impacted_countries:
                            if imp_country in scenario_ds.country_impacted.values:
                                bc_data = remaining_ds.sel(country_impacted=imp_country).sel(scenario_year = year)[impact_var]
                                bc_conc[i] += bc_data.sum().values
                    
                    # Calculate remaining CO2 emissions
                    co2_emis[i] = remaining_ds['co2_emissions'].sel(scenario_year = year).sum().values / 1e9 if plants_left > 0 else 0
                    
                    # Calculate cumulative CO2 emissions
                    cumulative_co2[i] = cumulative_co2[i-1] + co2_emis[i]
                
                # Extend timeline and values to 2050 if country timeline ends before 2050
                extended_timeline = timeline[country].copy()
                extended_co2_emis = co2_emis.copy()
                extended_cumulative_co2 = cumulative_co2.copy()
                extended_bc_conc = bc_conc.copy()
                extended_capacity = capacity.copy()
                
                if timeline[country][-1] < 2050:
                    # Create extended timeline to 2050
                    years_to_add = np.arange(timeline[country][-1] + 1, 2051)
                    extended_timeline = np.concatenate([timeline[country], years_to_add])
                    
                    # Extend arrays with final values
                    final_co2_emis = 0  # No new emissions after retirement
                    final_cumulative_co2 = cumulative_co2[-1]  # Maintain final cumulative value
                    final_bc_conc = 0  # No concentration after retirement
                    final_capacity = 0  # No capacity after retirement
                    
                    # Add zeros for annual emissions and concentration, maintain cumulative CO2
                    extended_co2_emis = np.concatenate([co2_emis, np.full(len(years_to_add), final_co2_emis)])
                    extended_cumulative_co2 = np.concatenate([cumulative_co2, np.full(len(years_to_add), final_cumulative_co2)])
                    extended_bc_conc = np.concatenate([bc_conc, np.full(len(years_to_add), final_bc_conc)])
                    extended_capacity = np.concatenate([capacity, np.full(len(years_to_add), final_capacity)])
                
                # Store data
                country_data[country][strategy][rate] = {
                    'years': extended_timeline,
                    'co2_emissions': extended_co2_emis,
                    'cumulative_co2': extended_cumulative_co2,
                    'bc_concentration': extended_bc_conc,
                    'capacity': extended_capacity
                }
                
                # Use extended timeline for plotting
                extended_timeline = country_data[country][strategy][rate]['years']
                extended_co2_emis = country_data[country][strategy][rate]['co2_emissions']
                extended_cumulative_co2 = country_data[country][strategy][rate]['cumulative_co2']
                extended_bc_conc = country_data[country][strategy][rate]['bc_concentration']
                extended_capacity = country_data[country][strategy][rate]['capacity']
                
                # Combine strategy style with rate color
                line_style = strategy_styles[strategy]['linestyle']
                color = rate_color_map[country][rate]  # Use the rate-color mapping
                
                # Construct label - only add labels for first column 
                if col_idx == 0 and s_idx == 0:
                    label = f"{names_dict[r_idx]}: {rate:.1f} plants/year"
                elif col_idx == 0 and r_idx == 0:
                    label = f"{strategy_names[strategy]}"
                else:
                    label = None
                
                # Plot capacity (row 0)
                axs[0, col_idx].plot(extended_timeline, extended_capacity, color=color, linestyle=line_style, 
                                   label=label, linewidth=2)
                
                # Plot BC concentration (row 1)
                axs[1, col_idx].plot(extended_timeline, extended_bc_conc, color=color, linestyle=line_style, linewidth=2)
                
                # Plot cumulative CO2 (row 2)
                axs[2, col_idx].plot(extended_timeline, extended_cumulative_co2, color=color, linestyle=line_style, linewidth=2)
        mult_val +=1
    # Set column titles (country names)
    for col_idx, country in enumerate(countries):
        axs[0, col_idx].set_title(f'{country.title()}', fontsize=16)
    
    # Set row labels (only for first column)
    axs[0, 0].set_ylabel('Capacity (GW)', fontsize=14)
    axs[1, 0].set_ylabel(f'Population weighted mean\nBlack Carbon Concentration\n(ng/m³/person)', fontsize=14)
    axs[2, 0].set_ylabel('Cumulative CO₂ (Gt)', fontsize=14)
    
    # Set x-axis labels and ticks (only for bottom row)
    for col_idx in range(len(countries)):
        axs[2, col_idx].set_xlabel('Year', fontsize=14)
        
        # Set x-ticks every 5 years from 2025 to 2050 for bottom row only
        x_ticks = np.arange(2025, 2051, 5)  # 2025, 2030, 2035, 2040, 2045, 2050
        axs[2, col_idx].set_xticks(x_ticks)
    
    # Remove x-ticks from top and middle rows
    for row in range(2):  # Only rows 0 and 1 (not bottom row)
        for col in range(len(countries)):
            axs[row, col].set_xticks([])
    
    # Add grid to all plots
    for row in range(3):
        for col in range(len(countries)):
            axs[row, col].grid(True, linestyle='--', alpha=0.5)
            
            # Add subplot lettering
            subplot_letter = chr(ord('a') + row * len(countries) + col)
            axs[row, col].text(0.04, 0.93, f'{subplot_letter})', transform=axs[row, col].transAxes, 
                              fontsize=14)
    
    # Set the x limits
    for row in range(3):
        for col, country in enumerate(countries):
            axs[row, col].set_xlim([2024, 2050])  
            axs[row, col].set_ylim(bottom=0, top =axs[row, col].get_ylim()[1]*1.06)  
    
    # Create legends based on whether rates is a dictionary or list
    if isinstance(rates, dict):
        # Create separate legends for each country showing only their specific rates
        for col, country in enumerate(countries):
            country_rates_list = country_rates[country]
            country_rate_legend_elements = [Line2D([0], [0], color=rate_color_map[country][rate], lw=2, 
                                                  label=f"{names_dict[idx]}: {rate:.0f} plants/year" if rate >= 1 else f"{names_dict[idx]}: {rate:.1f} plants/year")
                                           for idx, rate in enumerate(country_rates_list)]
            
            # Add rate legend underneath each country's bottom subplot
            axs[2, col].legend(handles=country_rate_legend_elements, 
                              loc='upper center', 
                              bbox_to_anchor=(0.5, -0.15),
                              title=f"Closure Rate",
                              fontsize=12,
                              frameon=True,
                              ncol=min(len(country_rates_list), 1))  # Max 3 columns
        
        # Create strategy legend on the right side
        strategy_legend_elements = [Line2D([0], [0], color='black', lw=2, 
                                          linestyle=strategy_styles[s]['linestyle'],
                                          label=strategy_names[s])
                                   for s in strategies]
        
        fig.legend(handles=strategy_legend_elements, 
                  loc='center right',
                  bbox_to_anchor=(1.03, 0.5),
                  title="Strategies",
                  fontsize=14,
                  frameon=True)
        
        # Adjust layout to make room for the legends
        plt.tight_layout()
        plt.subplots_adjust(right=0.85, bottom=0.15)  # Make room for legends
        
    else:
        # Create combined legend elements
        # Rates (colors)
        rate_legend_elements = [Line2D([0], [0], color=rate_color_map[country][rate], lw=2, 
                                      label=f"{rate:.1f} plants/year")
                               for rate in unique_rates]
        
        # Strategies (line styles)
        strategy_legend_elements = [Line2D([0], [0], color='black', lw=2, 
                                          linestyle=strategy_styles[s]['linestyle'],
                                          label=strategy_names[s])
                                   for s in strategies]
        
        # Combine all legend elements
        all_legend_elements = rate_legend_elements + strategy_legend_elements
        all_legend_labels = [elem.get_label() for elem in all_legend_elements]
        
        # Create a single legend to the right of the plots
        fig.legend(handles=all_legend_elements, 
                  labels=all_legend_labels,
                  loc='center right',
                  bbox_to_anchor=(0.99, 0.5),
                  fontsize=14,
                  frameon=True)
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for the legend on the right

    return fig, axs, country_data

def prepare_plant_benefits_data(single_year_ds, CGP_df, impact_var='BC_pop_weight_mean_conc', tcr=1.65, current_year=2025):
    """
    Prepare basic plant benefits data for mapping analysis.
    
    Parameters:
    -----------
    single_year_ds : xarray.Dataset
        Single year dataset with plant data
    CGP_df : pandas.DataFrame
        Coal plant dataframe with plant information
    impact_var : str
        Impact variable to analyze (default: 'BC_pop_weight_mean_conc')
    tcr : float
        Transient Climate Response factor (default: 1.65)
    current_year : int
        Current year to calculate remaining lifetime from (default: 2025)
        
    Returns:
    --------
    dict : Dictionary containing prepared data for mapping analysis
    """
    from shapely.geometry import Point
    import geopandas as gpd
    
    # Calculate benefits per MW for each plant
    plant_benefits = []
    
    for unique_id in single_year_ds.unique_ID.values:
        plant_data = single_year_ds.sel(unique_ID=unique_id)
        plant_info = CGP_df.loc[CGP_df.index == unique_id]
        
        if len(plant_info) > 0:
            # Get plant capacity in MW
            capacity_mw = plant_data['MW'].values.item()
            
            # Get year of commission from the dataset
            year_commissioned = plant_data['Year_of_Commission'].values.item()
            
            # Calculate remaining lifetime (assuming 40-year total lifetime)
            plant_total_lifetime = 40
            years_operating = current_year - year_commissioned
            remaining_years = max(0, plant_total_lifetime - years_operating)
            
            # Air quality benefit per MW (ng/m³ per MW)
            bc_benefit_per_mw = plant_data[impact_var].sum('country_impacted').mean('scenario_year').values.item() / capacity_mw
            
            # Temperature benefit per MW
            # CO2 component - direct marginal calculation
            co2_emissions_gt = plant_data['co2_emissions'].mean('scenario_year').values.item() / 1e9  # Gt CO2/yr
            # TCR K per 1000 Gt CO2 
            co2_temp_k = tcr * co2_emissions_gt / 1000  # K per year from CO2
            
            # BC component
            bc_temp_k = plant_data['dt_sum'].mean('scenario_year').values.item()
            
            # Total temperature per MW
            total_temp_per_mw = (co2_temp_k + bc_temp_k) / capacity_mw

            plant_benefits.append({
                'unique_ID': unique_id,
                'latitude': plant_info['latitude'].iloc[0],
                'longitude': plant_info['longitude'].iloc[0],
                'country': plant_info['COUNTRY'].iloc[0],
                'MW': capacity_mw,  # Use 'MW' to match other functions
                'year_commissioned': year_commissioned,
                'remaining_years': remaining_years,
                'bc_benefit_per_mw': bc_benefit_per_mw,
                'temp_benefit_per_mw': total_temp_per_mw
            })
    
    # Convert to GeoDataFrame
    benefits_df = pd.DataFrame(plant_benefits)
    geometry = [Point(xy) for xy in zip(benefits_df['longitude'], benefits_df['latitude'])]
    gdf_benefits = gpd.GeoDataFrame(benefits_df, geometry=geometry)
    
    # Calculate total remaining lifetime impacts using the remaining years
    gdf_benefits['total_bc_benefit_lifetime'] = 0.0
    gdf_benefits['total_temp_benefit_lifetime'] = 0.0

    # Calculate lifetime benefits using remaining years
    for idx, row in gdf_benefits.iterrows():
        plant_id = row['unique_ID']
        remaining_years = row['remaining_years']
        
        try:
            plant_data = single_year_ds.sel(unique_ID=plant_id, scenario_year=current_year)
            
            # BC benefit over remaining lifetime
            bc_data = plant_data[impact_var].sum('country_impacted')
            bc_impact = bc_data.item() * remaining_years
            
            # Temperature benefit - use same methodology as per-MW calculation
            # CO2 component
            co2_emissions_gt = plant_data['co2_emissions'].item() / 1e9  # Gt CO2/yr
            co2_temp_k = tcr * co2_emissions_gt / 1000  # K per year from CO2
            
            # BC component
            bc_temp_k = plant_data['dt_sum'].item()
            
            # Total temperature impact per year
            total_temp_per_year = co2_temp_k + bc_temp_k
            
            # Remaining lifetime temperature impact
            temp_impact = total_temp_per_year * remaining_years
            
            gdf_benefits.loc[idx, 'total_bc_benefit_lifetime'] = bc_impact
            gdf_benefits.loc[idx, 'total_temp_benefit_lifetime'] = temp_impact
            
        except Exception as e:
            print(f"Error processing plant {plant_id}: {e}")
            gdf_benefits.loc[idx, 'total_bc_benefit_lifetime'] = 0.0
            gdf_benefits.loc[idx, 'total_temp_benefit_lifetime'] = 0.0

    # Return comprehensive data dictionary for mapping
    return gdf_benefits



def plot_plant_benefits_maps(gdf_benefits, country_df, figsize=(18, 8)):
    """
    Create maps showing plant benefits per capacity from prepared data.
    
    Parameters:
    -----------
    prepared_data : dict
        Data dictionary from prepare_plant_benefits_data()
    country_df : geopandas.GeoDataFrame
        Country boundaries dataframe
    figsize : tuple
        Figure size as (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the maps
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    
    # Define colors for each category
    category_colors = {
        'Above Median (Temperature and Black Carbon)': '#2E8B57',     
        'Above Median (Black Carbon)': '#CD853F',      
        'Above Median (Temp)': '#4169E1', 
        'Below Median (Temperature and Black Carbon)': '#FF6347'    
    }
    
    # Create the plot with manual positioning to ensure equal sizes
    fig = plt.figure(figsize=figsize)
    
    # Create a GridSpec: 1 row, 3 columns with equal spacing
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], 
                        hspace=0.3, wspace=0.3)

    # All three plots in one row
    ax1 = fig.add_subplot(gs[0])  # Left - BC benefit
    ax2 = fig.add_subplot(gs[1])  # Center - Temperature benefit  
    ax3 = fig.add_subplot(gs[2])  # Right - Categorical benefits


     # Add subplot labels using a loop
    axes = [ax1, ax2, ax3]
    labels = ['a)', 'b)', 'c)']
    
    for ax, label in zip(axes, labels):
        ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=18,
                 verticalalignment='top')


    country_fill_color = 'lightgray'
    country_edge_color = 'darkgray'

    # Plot 1: BC benefit per MW (first plot)
    country_df.plot(ax=ax1, color=country_fill_color, edgecolor=country_edge_color, alpha=0.4, linewidth=0.5)

    scatter1 = gdf_benefits.plot(ax=ax1, column='bc_benefit_per_mw', cmap='Greens', 
                                markersize=30, legend=False, vmin=0, vmax=gdf_benefits['bc_benefit_per_mw'].max()*0.8,
                                edgecolor='black', linewidth=0.5)

    # Create colorbar for first plot
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    norm1 = Normalize(vmin=0, vmax=gdf_benefits['bc_benefit_per_mw'].max()*0.8)
    sm1 = ScalarMappable(norm=norm1, cmap='Greens')
    cbar1 = plt.colorbar(sm1, cax=cax1)
    cbar1.set_label('Black Carbon Benefit\n(ng/m³/MW)', fontsize=14)

    ax1.set_xlim(90, 150)
    ax1.set_ylim(-15, 30)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    # Plot 2: Temperature benefit per MW (second plot)
    country_df.plot(ax=ax2, color=country_fill_color, edgecolor=country_edge_color, alpha=0.4, linewidth=0.5)

    scatter2 = gdf_benefits.plot(ax=ax2, column='temp_benefit_per_mw', cmap='Blues',
                                markersize=30, legend=False, vmin=0, vmax=gdf_benefits['temp_benefit_per_mw'].max()*0.8,
                                edgecolor='black', linewidth=0.5)

    # Create colorbar for second plot
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    norm2 = Normalize(vmin=0, vmax=gdf_benefits['temp_benefit_per_mw'].max()*0.8)
    sm2 = ScalarMappable(norm=norm2, cmap='Blues')
    cbar2 = plt.colorbar(sm2, cax=cax2)
    cbar2.set_label('Temperature Benefit\n(K/MW)', fontsize=14)

    ax2.set_xlim(90, 150)
    ax2.set_ylim(-15, 30)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    # Plot 3: Categorical benefits (third plot)
    country_df.plot(ax=ax3, color=country_fill_color, edgecolor=country_edge_color, alpha=0.4, linewidth=0.5)

    # Calculate median values for categorization
    median_temp = gdf_benefits['temp_benefit_per_mw'].median()
    median_bc = gdf_benefits['bc_benefit_per_mw'].median()
    
    # Create categorical benefit classification
    def classify_benefits(row):
        temp_above_median = row['temp_benefit_per_mw'] > median_temp
        bc_above_median = row['bc_benefit_per_mw'] > median_bc
        
        if temp_above_median and bc_above_median:
            return 'Above Median (Temperature and Black Carbon)'
        elif not temp_above_median and not bc_above_median:
            return 'Below Median (Temperature and Black Carbon)'
        elif temp_above_median and not bc_above_median:
            return 'Above Median (Temp)'
        else:  # bc_above_median and not temp_above_median
            return 'Above Median (Black Carbon)'
    
    gdf_benefits['benefit_category'] = gdf_benefits.apply(classify_benefits, axis=1)
    
    
    # Plot categorical data
    for category, color in category_colors.items():
        category_data = gdf_benefits[gdf_benefits['benefit_category'] == category]
        if len(category_data) > 0:
            category_data.plot(ax=ax3, color=color, markersize=30, 
                            edgecolor='black', linewidth=0.5, label=category)

    # Create invisible colorbar for third plot to maintain alignment
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cax3.set_visible(False)  # Make it invisible

    ax3.set_xlim(90, 150)
    ax3.set_ylim(-15, 30)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    # Create custom legend for categorical plot with line breaks
    legend_labels = []
    for category in category_colors.keys():
        if 'Above Median (Temperature and Black Carbon)' in category:
            legend_labels.append('Above Median\n(Temperature and Black Carbon)')
        elif 'Below Median (Temperature and Black Carbon)' in category:
            legend_labels.append('Below Median\n(Temperature and Black Carbon)')
        elif 'Above Median (Temp)' in category:
            legend_labels.append('Above Median\n(Temp)')
        elif 'Above Median (Black Carbon)' in category:
            legend_labels.append('Above Median\n(Black Carbon)')
        else:
            legend_labels.append(category)

    legend_elements = [Patch(facecolor=color, edgecolor='black', label=label)
                    for (category, color), label in zip(category_colors.items(), legend_labels)]
    ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(2.15, 1.0), 
            fontsize=14, frameon=True, title='Benefit Category')

    plt.tight_layout()
    
    # Print some statistics
    print(f"\nPlant Benefits Summary:")
    print(f"Number of plants: {len(gdf_benefits)}")
    print(f"BC benefit per MW - Mean: {gdf_benefits['bc_benefit_per_mw'].mean():.2e}, Range: {gdf_benefits['bc_benefit_per_mw'].min():.2e} - {gdf_benefits['bc_benefit_per_mw'].max():.2e}")
    print(f"Temp benefit per MW - Mean: {gdf_benefits['temp_benefit_per_mw'].mean():.2e}, Range: {gdf_benefits['temp_benefit_per_mw'].min():.2e} - {gdf_benefits['temp_benefit_per_mw'].max():.2e}")
    print(f"Median BC benefit per MW: {median_bc:.2e}")
    print(f"Median Temp benefit per MW: {median_temp:.2e}")
    
    # Print category counts
    print(f"\nBenefit Category Counts:")
    category_counts = gdf_benefits['benefit_category'].value_counts()
    for category, count in category_counts.items():
        print(f"{category}: {count} plants ({count/len(gdf_benefits)*100:.1f}%)")
    
    return fig

def get_topn(gdf_benefits, n = 30):
     
    # Get top n plants for each category
    top_bc_per_mw = gdf_benefits.nlargest(n, 'bc_benefit_per_mw')
    top_temp_per_mw = gdf_benefits.nlargest(n, 'temp_benefit_per_mw')
    top_bc_lifetime = gdf_benefits.nlargest(n, 'total_bc_benefit_lifetime')
    top_temp_lifetime = gdf_benefits.nlargest(n, 'total_temp_benefit_lifetime')

    # Get all unique plant indices that appear in any top n list
    all_top_indices = set()
    all_top_indices.update(top_bc_per_mw.index)
    all_top_indices.update(top_temp_per_mw.index)
    all_top_indices.update(top_bc_lifetime.index)
    all_top_indices.update(top_temp_lifetime.index)

    # Find plants that appear in multiple lists
    appearances = {}
    for plant_idx in all_top_indices:
        appearances[plant_idx] = []
        if plant_idx in top_bc_per_mw.index:
            appearances[plant_idx].append('BC per-MW')
        if plant_idx in top_temp_per_mw.index:
            appearances[plant_idx].append('Temp per-MW')
        if plant_idx in top_bc_lifetime.index:
            appearances[plant_idx].append('BC lifetime')
        if plant_idx in top_temp_lifetime.index:
            appearances[plant_idx].append('Temp lifetime')
    return top_bc_per_mw, top_temp_per_mw, top_bc_lifetime, top_temp_lifetime, all_top_indices, appearances

def plot_plant_benefits_bars(gdf_benefits, n = 30, save_path=False):
    """
    Create bar plots showing top-n plants by different benefit categories.
    
    Parameters:
    -----------
    prepared_data : dict
        Data dictionary from prepare_plant_benefits_data()
    save_path : str or False
        Path to save the plot, or False to not save
        
    Returns:
    --------
    None (displays plot and prints analysis)
    """

    top_bc_per_mw, top_temp_per_mw, top_bc_lifetime, top_temp_lifetime, all_top_indices, appearances = get_topn(gdf_benefits, n)

    # Only assign hatch patterns to plants that appear in multiple lists
    multi_list_plants = [plant_idx for plant_idx, lists in appearances.items() if len(lists) > 1]

    # Create a consistent hatch pattern mapping only for plants that appear in multiple lists
    hatch_patterns = ['///', '\\\\\\', '|||', '---', '+++', '...', 'xxx', 
                    '^^^', '>>>', '<<<', '..', '--', '++', '**', '||', '//', '\\\\','ooo', '***']

    # Extend pattern list if we have more plants than patterns
    if len(multi_list_plants) > len(hatch_patterns):
        hatch_patterns = hatch_patterns * (len(multi_list_plants) // len(hatch_patterns) + 1)

    # Create a consistent mapping of plant index to hatch pattern (only for multi-list plants)
    plant_hatch_map = {}
    for i, plant_idx in enumerate(sorted(multi_list_plants)):
        plant_hatch_map[plant_idx] = hatch_patterns[i]
    
    # Create the plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 16))
    plt.rcParams['font.size'] = 12
    

    # Add subplot labels using a loop
    axes = [ax1, ax2, ax3, ax4]
    labels = ['a)', 'b)', 'c)', 'd)']
    
    for ax, label in zip(axes, labels):
        ax.text(0.95, 0.98, label, transform=ax.transAxes, fontsize=22,
                 verticalalignment='top')


  # Create country color mapping
    country_color_map = {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }

    countries = ['MALAYSIA', 'CAMBODIA', 'INDONESIA', 'VIETNAM']

    # Function to get hatch pattern for a plant (returns None if plant appears in only one list)
    def get_plant_hatch(plant_idx):
        return plant_hatch_map.get(plant_idx, None)

    # ====================== PLOTTING ======================

    # BC benefits per MW plot (top left)
    colors_bc_per_mw = [country_color_map[country] for country in top_bc_per_mw['country']]
    hatches_bc_per_mw = [get_plant_hatch(idx) for idx in top_bc_per_mw.index]

    bars1 = ax1.bar(range(len(top_bc_per_mw)), top_bc_per_mw['bc_benefit_per_mw'], 
                    color=colors_bc_per_mw, alpha=0.8, hatch=hatches_bc_per_mw)
    #ax1.set_xlabel('Plant Rank (Highest to Lowest Impact)')
    ax1.set_ylabel('Black Carbon Benefit per MW (ng/m³/MW)')
    #ax1.set_title(f'Top {n} Plants by Black Carbon Benefits per MW')
    #ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])

    # Temperature benefits per MW plot (top right)
    colors_temp_per_mw = [country_color_map[country] for country in top_temp_per_mw['country']]
    hatches_temp_per_mw = [get_plant_hatch(idx) for idx in top_temp_per_mw.index]

    bars2 = ax2.bar(range(len(top_temp_per_mw)), top_temp_per_mw['temp_benefit_per_mw'], 
                    color=colors_temp_per_mw, alpha=0.8, hatch=hatches_temp_per_mw)
    #ax2.set_xlabel('Plant Rank (Highest to Lowest Impact)')
    ax2.set_ylabel('Temperature Benefit per MW (K/MW)')
    #ax2.set_title(f'Top {n} Plants by Temperature Benefits per MW')
    #ax2.grid(True, alpha=0.3)
    ax2.set_xticks([])
    # BC lifetime benefits plot (bottom left)
    colors_bc_lifetime = [country_color_map[country] for country in top_bc_lifetime['country']]
    hatches_bc_lifetime = [get_plant_hatch(idx) for idx in top_bc_lifetime.index]

    bars3 = ax3.bar(range(len(top_bc_lifetime)), top_bc_lifetime['total_bc_benefit_lifetime'], 
                    color=colors_bc_lifetime, alpha=0.8, hatch=hatches_bc_lifetime)
    ax3.set_xlabel('Plant Rank (Highest to Lowest Impact)')
    ax3.set_ylabel('Total Black Carbon Benefit over 40-year Lifetime (ng/m³·years)')
    #ax3.set_title(f'Top {n} Plants by Total Black Carbon Benefits over 40-year Lifetime')
    #ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    # Temperature lifetime benefits plot (bottom right)
    colors_temp_lifetime = [country_color_map[country] for country in top_temp_lifetime['country']]
    hatches_temp_lifetime = [get_plant_hatch(idx) for idx in top_temp_lifetime.index]

    bars4 = ax4.bar(range(len(top_temp_lifetime)), top_temp_lifetime['total_temp_benefit_lifetime'], 
                    color=colors_temp_lifetime, alpha=0.8, hatch=hatches_temp_lifetime)
    ax4.set_xlabel('Plant Rank (Highest to Lowest Impact)')
    ax4.set_ylabel('Total Temperature Benefit over 40-year Lifetime (K·years)')
    #ax4.set_title(f'Top {n} Plants by Total Temperature Benefits over 40-year Lifetime')
    #ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

    # ====================== LEGEND ======================
    # Add country legend
    legend_elements = []

    # Add country colors - capitalize only first letter
    for country in countries:
        country_name = country.title()  # Capitalize only first letter
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=country_color_map[country], 
                                        label=country_name))

    # Add separator
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='white', alpha=0, label=''))

    # Add explanation for hatching
    if multi_list_plants:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', hatch='///', 
                                        alpha=0.8, label=f'Plants in multiple lists ({len(multi_list_plants)} plants)'))

    # Place legend on the top right subplot
    ax2.legend(handles=legend_elements,
            bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # ====================== ANALYSIS ======================
    
    print(f"\n{'='*80}")
    print("MEDIAN VALUES")
    print(f"{'='*80}")
    print('Lifetime bc benefit median, all plants:', gdf_benefits['total_bc_benefit_lifetime'].median())
    print('Lifetime temp benefit median, all plants:', gdf_benefits['total_temp_benefit_lifetime'].median())
    print('BC/MW benefit median, all plants:', gdf_benefits['bc_benefit_per_mw'].median())
    print('Temp/MW benefit median, all plants:', gdf_benefits['temp_benefit_per_mw'].median())

    print(f"\n{'='*80}")
    print(f"PLANT TRACKING ANALYSIS")
    print(f"{'='*80}")

    print(f"\nTotal unique plants appearing in any top-{n} list: {len(all_top_indices)}")
    print(f"Plants appearing in multiple lists (hatched): {len(multi_list_plants)}")
    print(f"Plants appearing in only one list (no hatch): {len(all_top_indices) - len(multi_list_plants)}")

    # Count plants by number of appearances
    appearance_counts = {}
    for plant_idx, lists in appearances.items():
        count = len(lists)
        if count not in appearance_counts:
            appearance_counts[count] = []
        appearance_counts[count].append(plant_idx)

    print(f"\nPlants by number of top-{n} list appearances:")
    for count in sorted(appearance_counts.keys(), reverse=True):
        plants = appearance_counts[count]
        if count == 1:
            print(f"\nPlants in {count} list ({len(plants)} plants - no hatching):")
        else:
            print(f"\nPlants in {count} lists ({len(plants)} plants - with hatching):")
        
        for plant_idx in plants[:15]:  # Show first 15 plants
            plant_info = gdf_benefits.loc[plant_idx]
            lists_appeared = ", ".join(appearances[plant_idx])
            if count > 1:
                hatch = plant_hatch_map[plant_idx]
                print(f"  Plant {plant_idx} ({plant_info['country']}, {hatch}): {lists_appeared}")
            else:
                print(f"  Plant {plant_idx} ({plant_info['country']}, no hatch): {lists_appeared}")
        if len(plants) > 15:
            print(f"  ... and {len(plants) - 15} more plants")

    # Find the "super plants" that appear in all 4 lists
    super_plants = [plant_idx for plant_idx, lists in appearances.items() if len(lists) == 4]
    if super_plants:
        print(f"\n{'='*50}")
        print(f"SUPER PLANTS (appear in all 4 top-{n} lists): {len(super_plants)} plants")
        print(f"{'='*50}")
        for plant_idx in super_plants:
            plant_info = gdf_benefits.loc[plant_idx]
            hatch = plant_hatch_map[plant_idx]
            
            # Get rankings in each list
            bc_per_mw_rank = list(top_bc_per_mw.index).index(plant_idx) + 1
            temp_per_mw_rank = list(top_temp_per_mw.index).index(plant_idx) + 1
            bc_lifetime_rank = list(top_bc_lifetime.index).index(plant_idx) + 1
            temp_lifetime_rank = list(top_temp_lifetime.index).index(plant_idx) + 1
            
            print(f"\nPlant {plant_idx} (Pattern: {hatch}):")
            print(f"  Country: {plant_info['country']}")
            print(f"  Capacity: {plant_info['MW']:.1f} MW")
            print(f"  Rankings: BC per-MW #{bc_per_mw_rank}, Temp per-MW #{temp_per_mw_rank}")
            print(f"            BC lifetime #{bc_lifetime_rank}, Temp lifetime #{temp_lifetime_rank}")
            print(f"  BC per-MW: {plant_info['bc_benefit_per_mw']:.2e}")
            print(f"  Temp per-MW: {plant_info['temp_benefit_per_mw']:.2e}")
            print(f"  Total BC lifetime: {plant_info['total_bc_benefit_lifetime']:.2e}")
            print(f"  Total Temp lifetime: {plant_info['total_temp_benefit_lifetime']:.2e}")

def plot_top_plants_maps(gdf_benefits, country_df, n=30, figsize=(20, 12), save_path=None, jitter_amount=0.05):
    """
    Create maps showing the top N plants for each benefit metric.
    Handles overlapping points with jitter and transparency.
    
    Parameters:
    -----------
    gdf_benefits : GeoDataFrame
        GeoDataFrame with plant benefits data including geometry (lat/lon points)
    country_df : GeoDataFrame
        GeoDataFrame with country boundaries
    n : int
        Number of top plants to show (default: 30)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    jitter_amount : float
        Amount of random jitter to add to coordinates (default: 0.05 degrees)
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import numpy as np
    
    # Define the four benefit metrics
    metrics = {
        'bc_benefit_per_mw': 'Black Carbon Concentration\nDecrease per MW',
        'temp_benefit_per_mw': 'Temperature\nDecrease per MW',
        'total_bc_benefit_lifetime': 'Total Black Carbon Concentration\nDecrease over Lifetime',
        'total_temp_benefit_lifetime': 'Total Temperature\nDecrease over Lifetime'
    }
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Color palette for countries
    country_colors = {
        'MALAYSIA': '#0072B2',  
        'CAMBODIA': '#009E73',  
        'INDONESIA': '#D55E00', 
        'VIETNAM': '#CC79A7'    
    }
    
    for idx, (metric, title) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Plot base map - all countries in light gray
        country_df.plot(ax=ax, color='lightgray', edgecolor='black', 
                       linewidth=0.5, alpha=0.3)
        
        # Highlight the four countries of interest
        for country, color in country_colors.items():
            country_geom = country_df[country_df['country'] == country]
            if not country_geom.empty:
                country_geom.plot(ax=ax, color=color, edgecolor='black',
                                linewidth=0.8, alpha=0.4)
        
        # Get top N plants for this metric
        top_plants = gdf_benefits.nlargest(n, metric).copy()
        
        # Add small random jitter to separate overlapping points
        np.random.seed(42 + idx)  # Different seed for each subplot but reproducible
        top_plants['lon_jittered'] = top_plants.geometry.x + np.random.uniform(
            -jitter_amount, jitter_amount, len(top_plants)
        )
        top_plants['lat_jittered'] = top_plants.geometry.y + np.random.uniform(
            -jitter_amount, jitter_amount, len(top_plants)
        )
        
        # Calculate colorbar limits
        vmin = top_plants[metric].min()
        vmax = top_plants[metric].max()
        
        # Plot top plants as scatter points with jittered coordinates
        scatter = ax.scatter(
            top_plants['lon_jittered'],
            top_plants['lat_jittered'],
            s=200,  # Marker size
            c=top_plants[metric],
            cmap='YlOrRd',
            edgecolors='black',
            linewidth=1,
            zorder=5,
            alpha=0.7,  # Semi-transparent to see overlapping points
            vmax=vmax,
            vmin=vmin
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(title, fontsize=10)
        
        # Format colorbar labels based on magnitude
        if vmax < 0.01:
            cbar.ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Set map extent to Southeast Asia
        ax.set_xlim(95, 125)
        ax.set_ylim(-10, 25)
        
        # Add title with count info
        unique_locs = len(top_plants[['latitude', 'longitude']].drop_duplicates())
        ax.set_title(f'Top {n} Plants by {title}', 
                    fontsize=11)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Remove axis labels for cleaner look
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Add subplot label
        subplot_label = chr(ord('a') + idx)
        ax.text(0.02, 0.98, f'{subplot_label})', transform=ax.transAxes,
               fontsize=14, verticalalignment='top')
        
        # Print statistics for this metric
        print(f"\n{title}:")
        print(f"  Total plants: {len(top_plants)}")
        print(f"  Unique locations: {unique_locs}")
        print(f"  Value range: {vmin:.2e} - {vmax:.2e}")
        print(f"  Countries represented:")
        country_counts = top_plants['country'].value_counts()
        for country, count in country_counts.items():
            print(f"    {country}: {count} plants")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    return fig, axes