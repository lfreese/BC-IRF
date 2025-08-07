#!/bin/bash
# filepath: /home/emfreese/fs11/d0/emfreese/BC-IRF/raw_data_inputs/raisanen_2022_BC_temp/praisanen-Black-carbon-radiative-forcing-and-climate-response-tool-182f7d3/test_compilation.sh

echo "=== Testing Compilation ==="

# Load modules (adjust based on your system)
echo "Loading modules..."
module load netcdf-fortran 2>/dev/null || module load netcdf/fortran 2>/dev/null || echo "Could not load NetCDF module"

# Test compilation with different methods
echo "Testing compilation methods..."

if [ -f "calculate_forcing_and_climate_response.f90" ]; then
    # Method 1: Intel compiler
    if command -v ifort &> /dev/null; then
        echo "Trying Intel compiler..."
        ifort calculate_forcing_and_climate_response.f90 -lnetcdff -o test_intel 2>&1 && echo "Intel compilation: SUCCESS" || echo "Intel compilation: FAILED"
    fi
    
    # Method 2: gfortran
    if command -v gfortran &> /dev/null; then
        echo "Trying gfortran..."
        gfortran calculate_forcing_and_climate_response.f90 -lnetcdff -o test_gfortran 2>&1 && echo "gfortran compilation: SUCCESS" || echo "gfortran compilation: FAILED"
    fi
    
    # Method 3: Using nf-config
    if command -v nf-config &> /dev/null; then
        echo "Trying compilation with nf-config..."
        gfortran calculate_forcing_and_climate_response.f90 $(nf-config --fflags) $(nf-config --flibs) -o test_nfconfig 2>&1 && echo "nf-config compilation: SUCCESS" || echo "nf-config compilation: FAILED"
    fi
else
    echo "ERROR: calculate_forcing_and_climate_response.f90 not found"
fi

# Clean up test executables
rm -f test_intel test_gfortran test_nfconfig