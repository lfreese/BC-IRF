import pandas as pd
import numpy as np
import os

import sys
sys.path.insert(0, '/net/fs11/d0/emfreese/BC-IRF/')
import utils


def create_input_files_from_cgp_df(cgp_csv_path):
    """Extract coordinates and emissions from CGP_df and create input_data.asc files for each plant"""
    
    # Read the CGP dataframe
    CGP_df = pd.read_csv(cgp_csv_path)
    
    print(f"Total plants in CGP_df: {len(CGP_df)}")
    
    # Filter for plants with valid year of start:
    CGP_df.columns = CGP_df.columns.str.replace(' ', '_')
    CGP_df = CGP_df.rename(columns = {'YEAR':'Year_of_Commission'})

    CGP_df = CGP_df.loc[CGP_df['Year_of_Commission'].dropna().index]

  # Filter for plants with valid coordinates and BC emissions
    valid_plants = CGP_df[
        (CGP_df['latitude'].notna()) & 
        (CGP_df['longitude'].notna()) & 
        (CGP_df['BC_(kg/m2/year)'].notna()) &
        (CGP_df['BC_(kg/m2/year)'] > 0)
    ].copy()
    print(f"Limited to valid data")

    # Create output directory
    output_dir = "input_files_all_plants"
    os.makedirs(output_dir, exist_ok=True)
    
    # Template for input_data.asc format
    template = """!******************************************************************************
! INSTRUCTIONS:
! 1. for the longitude limits, give first the western borundary, 
!    second the eastern boundary (allowed range -360...360 degrees east)
! 2. for the latiitude limits, give first the southern borundary, 
!    second the northern boundary (allowed range -90...90 degrees north)
! 3. The BC emissions can be given either as annual-mean (1 value)
!    or monthly-mean values (12 values). Give the emissions on line(s) after 
!    the keyword "EMISSION_RATE". Monthly emission values can be separated 
!    by space, comma (,) or semicolon (;). The units are [kg m-2 s-1]
!
! To avoid problems, try not to change the structure of this file!
!******************************************************************************

LONGITUDE_LIMITS {lon_w:.4f} {lon_e:.4f}
LATITUDE_LIMITS {lat_s:.4f} {lat_n:.4f}
EMISSION_RATE 
 {emissions}
"""
    
    plants_processed = []
    
    # Define grid cell size for each plant (you can adjust this)
    grid_size = 0.5  # 0.5 degrees around each plant
    
    # Process each plant
    for idx, plant in valid_plants.iterrows():
        try:
            lat = float(plant['latitude'])
            lon = float(plant['longitude'])
            
            # Convert from kg/m2/year to kg/m2/s
            bc_emission_year = float(plant['BC_(kg/m2/year)'])
            bc_emission_per_sec = bc_emission_year / (365.25 * 24 * 3600)  # Convert to per second
            
            # Create grid bounds around the plant
            lat_s = lat - grid_size/2
            lat_n = lat + grid_size/2
            lon_w = lon - grid_size/2
            lon_e = lon + grid_size/2
            
            # Get the actual UNITID from the data, or create one if missing
            unit_id = plant.get('UNITID', f"plant_{idx:04d}")
            
            # Create plant identifier for filename (clean for filesystem)
            if pd.notna(unit_id):
                plant_id = f"plant_{unit_id}"
            elif 'PLANT_NAME' in plant and pd.notna(plant['PLANT_NAME']):
                # Clean plant name for filename
                clean_name = str(plant['PLANT_NAME']).replace(' ', '_').replace('/', '_')[:20]
                plant_id = f"plant_{clean_name}_{idx}"
            else:
                plant_id = f"plant_{idx:04d}"
            
            # Use annual emission (single value)
            emissions_formatted = f"{bc_emission_per_sec:.2E}"
            
            # Store plant info - use actual UNITID, not plant_id
            plants_processed.append({
                'plant_id': plant_id,  # For filename identification
                'UNITID': unit_id,     # Actual UNITID from original data
                'original_index': idx,
                'plant_name': plant.get('PLANT_NAME', 'Unknown'),
                'country': plant.get('COUNTRY', 'Unknown'),
                'city': plant.get('CITY', 'Unknown'),
                'state': plant.get('STATE', 'Unknown'),
                'latitude': lat,
                'longitude': lon,
                'lat_s': lat_s,
                'lat_n': lat_n,
                'lon_w': lon_w,
                'lon_e': lon_e,
                'bc_emission_kg_m2_year': bc_emission_year,
                'bc_emission_kg_m2_sec': bc_emission_per_sec,
                'capacity_mw': plant.get('CAPACITY (MW)', np.nan),
                'year_commission': plant.get('Year_of_Commission', np.nan)
            })
            
            # Create input file for this plant
            input_content = template.format(
                lon_w=lon_w,
                lon_e=lon_e,
                lat_s=lat_s,
                lat_n=lat_n,
                emissions=emissions_formatted
            )
            
            # Write to file using plant_id for filename
            filename = f"{output_dir}/input_data_{plant_id}.asc"
            with open(filename, 'w') as f:
                f.write(input_content)
                
        except Exception as e:
            print(f"Error processing plant at index {idx}: {e}")
            continue
    
    # Save plants summary
    df = pd.DataFrame(plants_processed)
    df.to_csv(f"{output_dir}/plants_summary.csv", index=False)
    
    print(f"\nCreated {len(plants_processed)} input files in '{output_dir}/' directory")
    print(f"Plants summary saved to '{output_dir}/plants_summary.csv'")
    
    # Show sample plants
    print("\nSample plants:")
    print(df[['plant_id', 'UNITID', 'plant_name', 'country', 'latitude', 'longitude', 'bc_emission_kg_m2_year']].head(10))
    
    # Summary statistics
    print(f"\nBC Emission Statistics (kg/m2/year):")
    print(f"  Min: {df['bc_emission_kg_m2_year'].min():.2e}")
    print(f"  Max: {df['bc_emission_kg_m2_year'].max():.2e}")
    print(f"  Mean: {df['bc_emission_kg_m2_year'].mean():.2e}")
    print(f"  Median: {df['bc_emission_kg_m2_year'].median():.2e}")
    
    return df

if __name__ == "__main__":
    # Path to your CGP CSV file - adjust this path as needed
    cgp_csv_path = f"{utils.data_output_path}plants/BC_SE_Asia_all_financing_SEA_GAINS_Springer_v2.csv"
    
    # Alternative paths to check
    possible_paths = [
        cgp_csv_path,
    ]
    
    cgp_file_found = None
    for path in possible_paths:
        if os.path.exists(path):
            cgp_file_found = path
            print(f"Found CGP file at: {path}")
            break
    
    if cgp_file_found:
        df = create_input_files_from_cgp_df(cgp_file_found)
    else:
        print("CGP CSV file not found. Please check the path.")
        print("Expected paths:")
        for path in possible_paths:
            print(f"  {path}")