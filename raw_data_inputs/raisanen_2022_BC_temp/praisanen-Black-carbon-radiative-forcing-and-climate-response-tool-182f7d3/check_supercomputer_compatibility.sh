#!/bin/bash
# filepath: /home/emfreese/fs11/d0/emfreese/BC-IRF/raw_data_inputs/raisanen_2022_BC_temp/praisanen-Black-carbon-radiative-forcing-and-climate-response-tool-182f7d3/check_supercomputer_compatibility.sh

echo "=== Checking Supercomputer Compatibility ==="
echo ""

echo "1. Checking available modules..."
echo "NetCDF modules:"
module avail 2>&1 | grep -i netcdf || echo "No NetCDF modules found"
echo ""
echo "Fortran/Intel modules:"
module avail 2>&1 | grep -i -E "(fortran|intel|gcc)" || echo "No Fortran modules found"
echo ""

echo "2. Checking available compilers..."
for compiler in ifort gfortran f90 f95; do
    if command -v $compiler &> /dev/null; then
        echo "$compiler: $(which $compiler)"
        $compiler --version 2>/dev/null | head -1
    else
        echo "$compiler: not found"
    fi
done
echo ""

echo "3. Attempting to load NetCDF module..."
# Try common NetCDF module names
for module_name in "netcdf-fortran" "netcdf/fortran" "NetCDF-Fortran" "netcdf" "NetCDF" "netcdf-c" "netcdf-cxx"; do
    echo "Trying: module load $module_name"
    if module load $module_name 2>/dev/null; then
        echo "Success! Loaded $module_name"
        echo "nf-config location: $(which nf-config 2>/dev/null || echo 'not found')"
        if command -v nf-config &> /dev/null; then
            echo "NetCDF configuration:"
            nf-config --all
        fi
        module unload $module_name 2>/dev/null
        break
    else
        echo "Failed to load $module_name"
    fi
done
echo ""

echo "4. Checking for required files..."
echo "Current directory: $(pwd)"
for file in "calculate_forcing_and_climate_response.f90" "input_data.csv" "BC_forcing_and_climate_response_normalized_by_emissions.nc"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
    fi
done
echo ""

echo "5. System information..."
echo "OS: $(uname -a)"
echo "Available cores: $(nproc 2>/dev/null || echo 'unknown')"
echo "Memory: $(free -h 2>/dev/null | grep Mem || echo 'unknown')"
echo ""

echo "6. Environment check..."
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "NETCDF_ROOT: ${NETCDF_ROOT:-not set}"
echo "NETCDF_FORTRAN_ROOT: ${NETCDF_FORTRAN_ROOT:-not set}"