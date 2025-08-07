#!/bin/bash
# filepath: /home/emfreese/fs11/d0/emfreese/BC-IRF/raw_data_inputs/raisanen_2022_BC_temp/praisanen-Black-carbon-radiative-forcing-and-climate-response-tool-182f7d3/run_all_plants.sh

echo "=== Running BC Calculator for All Power Plants ==="

# Check if executable exists
if [ ! -f "bc_calculator" ]; then
    echo "ERROR: bc_calculator executable not found"
    echo "Please compile first using: ./compile_and_run.sh"
    exit 1
fi

# Create output directory
mkdir -p "results_all_plants"

# Get list of input files
input_dir="input_files_all_plants"
if [ ! -d "$input_dir" ]; then
    echo "ERROR: Input files directory '$input_dir' not found"
    echo "Please run extract_lat_lon_plants.py first"
    exit 1
fi

# Count total files
total_files=$(ls ${input_dir}/input_data_*.asc | wc -l)
echo "Found $total_files plants to process"

# Process each input file
counter=0
for input_file in ${input_dir}/input_data_*.asc; do
    counter=$((counter + 1))
    
    # Extract plant ID from filename
    basename=$(basename "$input_file")
    plant_id=${basename#input_data_}
    plant_id=${plant_id%.asc}
    
    echo "[$counter/$total_files] Processing plant: $plant_id"
    
    # Copy input file to expected location
    cp "$input_file" "input_data.asc"
    
    # Run the calculator
    output_file="results_all_plants/output_${plant_id}.txt"
    if ./bc_calculator > "$output_file" 2>&1; then
        echo "  ✓ Success: $output_file"
    else
        echo "  ✗ Failed: Check $output_file for errors"
    fi
    
    # Optional: Add a small delay to avoid overwhelming the system
    # sleep 0.1
done

echo ""
echo "=== Processing Complete ==="
echo "Results saved in 'results_all_plants/' directory"
echo "Total plants processed: $counter"

# Create summary of results
echo ""
echo "Creating results summary..."
python3 -c "
import pandas as pd
import glob
import re

results = []
for file in glob.glob('results_all_plants/output_*.txt'):
    plant_match = re.search(r'output_(.+)\.txt', file)
    if plant_match:
        plant_id = plant_match.group(1)
        try:
            with open(file, 'r') as f:
                content = f.read()
                # Extract key results (adjust based on actual output format)
                if 'DRF_TOA' in content and 'error' not in content.lower():
                    status = 'SUCCESS'
                else:
                    status = 'FAILED'
        except:
            status = 'ERROR'
        
        results.append({'plant_id': plant_id, 'status': status, 'output_file': file})

df = pd.DataFrame(results)
df.to_csv('results_all_plants/processing_summary.csv', index=False)
print(f'Summary saved to results_all_plants/processing_summary.csv')
print(f'Success: {sum(df.status == \"SUCCESS\")}')
print(f'Failed: {sum(df.status == \"FAILED\")}')
print(f'Error: {sum(df.status == \"ERROR\")}')
"