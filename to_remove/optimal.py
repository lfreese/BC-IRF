
def sorting(scenario_ds, countries=['MALAYSIA', 'INDONESIA', 'VIETNAM'],
                                              rates=[1.0, 2.0, 4.0], 
                                              force_closure_by={'MALAYSIA': 2045, 'CAMBODIA': 2050, 'INDONESIA': 2040, 'VIETNAM': 2050},
                                              strategies=['Year_of_Commission', 'MW', 'CO2_weighted_capacity_1000tonsperMW']):
    """

    
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

        

    """
    # Dictionary to store all data
    country_data = {}
    
    timeline = {}
    # Process each country
    for country in countries:

        # Define timeline from 2025 to force_closure_by
        timeline[country] = np.arange(2025, force_closure_by[country] + 1)

        # Initialize country data dictionary
        country_data[country] = {}
        
        # Get the country dataset
        country_ds = scenario_ds.where(scenario_ds.country_emitting == country, drop=True)

        sorted_ds = {}
        # For each strategy
        for strategy in strategies:
            #print(country_ds[strategy])
            # Sort plants by strategy
            if strategy == 'Year_of_Commission':
                sorted_ds[strategy] = country_ds.sortby(strategy)  # oldest first
               
            else:
                sorted_ds[strategy] = country_ds.sortby(strategy, ascending=False)  # Biggest first
               
    return sorted_ds


sorted_ds = sorting(scenario_ds, countries=[ 'INDONESIA', 'MALAYSIA', 'VIETNAM'],
                                              force_closure_by=closure_yr_dict,
                                              strategies=['Year_of_Commission', 'EMISFACTOR.CO2'],
                                             )