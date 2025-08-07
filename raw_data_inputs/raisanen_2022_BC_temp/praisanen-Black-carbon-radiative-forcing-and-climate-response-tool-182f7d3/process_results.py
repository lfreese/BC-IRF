import pandas as pd
import glob
import re
import os

def parse_fortran_output(filepath):
    """Parse the Fortran output file to extract key results"""
    results = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if file is empty
        if len(content.strip()) == 0:
            results = {key: None for key in ['DRF_TOA', 'SNOWRF_TOA', 'SUM_RF_TOA', 'DT_DRF', 'DT_SNOWRF', 'DT_SUM']}
            results['status'] = 'EMPTY'
            return results
        
        # Updated patterns based on actual output format
        patterns = {
            'DRF_TOA': r'DRF_TOA\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            'SNOWRF_TOA': r'SNOWRF_TOA\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            'SUM_RF_TOA': r'SUM_RF_TOA\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            'DT_DRF': r'DT_DRF\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            'DT_SNOWRF': r'DT_SNOWRF\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
            'DT_SUM': r'DT_SUM\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        }
        
        # Extract main variables
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    results[key] = float(match.group(1))
                except ValueError:
                    results[key] = None
            else:
                results[key] = None
        
        # Determine status
        main_vars = ['DRF_TOA', 'SNOWRF_TOA', 'SUM_RF_TOA', 'DT_DRF', 'DT_SNOWRF', 'DT_SUM']
        if any(results.get(var) is not None for var in main_vars):
            results['status'] = 'SUCCESS'
        else:
            results['status'] = 'NO_VALUES_FOUND'
        
    except Exception as e:
        results = {key: None for key in ['DRF_TOA', 'SNOWRF_TOA', 'SUM_RF_TOA', 'DT_DRF', 'DT_SNOWRF', 'DT_SUM']}
        results['status'] = 'ERROR'
        results['error'] = str(e)
    
    return results

def combine_all_plant_results():
    """Combine results from all plants into a single DataFrame"""
    
    # Load plants summary
    plants_file = "input_files_all_plants/plants_summary.csv"
    if not os.path.exists(plants_file):
        print(f"ERROR: {plants_file} not found.")
        return None
    
    plants_df = pd.read_csv(plants_file)
    print(f"Loading {len(plants_df)} plants...")
    
    # Parse all output files
    all_results = []
    
    for i, (_, plant) in enumerate(plants_df.iterrows()):
        plant_id = plant['plant_id']
        output_file = f"results_all_plants/output_{plant_id}.txt"
        
        if os.path.exists(output_file):
            results = parse_fortran_output(output_file)
        else:
            results = {key: None for key in ['DRF_TOA', 'SNOWRF_TOA', 'SUM_RF_TOA', 'DT_DRF', 'DT_SNOWRF', 'DT_SUM']}
            results['status'] = 'NOT_FOUND'
        
        # Combine plant info with results
        combined = {**plant.to_dict(), **results}
        all_results.append(combined)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(plants_df)} files...")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save combined results
    output_file = "results_all_plants/combined_plant_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total plants: {len(results_df)}")
    
    # Status summary
    status_counts = results_df['status'].value_counts()
    print("\nStatus summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Show sample successful results
    successful = results_df[results_df['status'] == 'SUCCESS']
    if len(successful) > 0:
        print(f"\nSuccessful results: {len(successful)}")
        print("\nSample results:")
        sample_cols = ['UNITID', 'plant_name', 'country', 'latitude', 'longitude', 'bc_emission_kg_m2_year', 'DRF_TOA', 'SNOWRF_TOA']
        print(successful[sample_cols].head())
        
        # Summary statistics for successful plants
        print(f"\nRadiative Forcing Statistics for Successful Plants:")
        numeric_cols = ['DRF_TOA', 'SNOWRF_TOA', 'SUM_RF_TOA', 'DT_DRF', 'DT_SNOWRF', 'DT_SUM']
        print(successful[numeric_cols].describe())
        
    else:
        print("No successful results found!")
    
    return results_df

if __name__ == "__main__":
    df = combine_all_plant_results()