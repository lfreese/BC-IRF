#!/bin/bash
# filepath: /home/emfreese/fs11/d0/emfreese/BC-IRF/raw_data_inputs/raisanen_2022_BC_temp/praisanen-Black-carbon-radiative-forcing-and-climate-response-tool-182f7d3/compile_and_run.sh

#!/bin/bash

echo "=== Compiling and Running BC Radiative Forcing Calculator ==="
echo ""

# Check for required files
missing_files=0
for file in "calculate_forcing_and_climate_response.f90" "input_data.csv" "BC_forcing_and_climate_response_normalized_by_emissions.nc"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file '$file' not found"
        missing_files=$((missing_files + 1))
    fi
done

if [ $missing_files -gt 0 ]; then
    echo "Please ensure all required files are in the current directory"
    exit 1
fi

# Try to load NetCDF module
echo "Loading NetCDF module..."
module_loaded=false
for module_name in "netcdf" ; do
    if module load $module_name 2>/dev/null; then
        echo "Loaded module: $module_name"
        module_loaded=true
        break
    fi
done

# Compilation attempts
compilation_success=false
executable_name="bc_calculator"

echo ""
echo "Attempting compilation..."

# # Method 1: Intel compiler (preferred for supercomputers)
# if command -v ifort &> /dev/null && [ "$compilation_success" = false ]; then
#     echo "Trying Intel compiler..."
#     if ifort calculate_forcing_and_climate_response.f90 -lnetcdff -o $executable_name 2>/dev/null; then
#         echo "✓ Compilation successful with Intel compiler"
#         compilation_success=true
#     fi
# fi

# Method 2: nf-config
if command -v nf-config &> /dev/null && [ "$compilation_success" = false ]; then
    echo "Trying compilation with nf-config..."
    compiler="gfortran"
    if command -v ifort &> /dev/null; then
        compiler="ifort"
    fi
    
    if $compiler calculate_forcing_and_climate_response.f90 $(nf-config --fflags) $(nf-config --flibs) -o $executable_name 2>/dev/null; then
        echo "✓ Compilation successful with nf-config"
        compilation_success=true
    fi
fi

# # Method 3: gfortran basic
# if command -v gfortran &> /dev/null && [ "$compilation_success" = false ]; then
#     echo "Trying gfortran..."
#     if gfortran calculate_forcing_and_climate_response.f90 -lnetcdff -o $executable_name 2>/dev/null; then
#         echo "✓ Compilation successful with gfortran"
#         compilation_success=true
#     fi
# fi

# # # Method 4: Try with common paths
# # if [ "$compilation_success" = false ]; then
# #     echo "Trying with manual include/library paths..."
# #     for include_path in "/usr/include" "/opt/local/include" "/usr/local/include" "$NETCDF_ROOT/include" "$NETCDF_FORTRAN_ROOT/include"; do
# #         for lib_path in "/usr/lib" "/opt/local/lib" "/usr/local/lib" "$NETCDF_ROOT/lib" "$NETCDF_FORTRAN_ROOT/lib"; do
# #             if [ -d "$include_path" ] && [ -d "$lib_path" ] && [ "$compilation_success" = false ]; then
# #                 if command -v gfortran &> /dev/null; then
# #                     if gfortran calculate_forcing_and_climate_response.f90 -I"$include_path" -L"$lib_path" -lnetcdff -o $executable_name 2>/dev/null; then
# #                         echo "✓ Compilation successful with manual paths"
# #                         compilation_success=true
# #                         break 2
# #                     fi
# #                 fi
# #             fi
# #         done
# #     done
# # fi

if [ "$compilation_success" = false ]; then
    echo "✗ All compilation attempts failed"
    echo "Please check:"
    echo "  1. NetCDF Fortran library is installed"
    echo "  2. Appropriate modules are loaded"
    echo "  3. Compiler has access to NetCDF headers and libraries"
    exit 1
fi

# Run the program
echo ""
echo "Running the program..."
if [ -x "$executable_name" ]; then
    ./$executable_name > output.txt 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Program executed successfully"
        echo "Output written to: output.txt"
        echo ""
        echo "Output preview:"
        head -20 output.txt
    else
        echo "✗ Program execution failed"
        echo "Error output:"
        cat output.txt
    fi
else
    echo "✗ Executable not found or not executable"
    exit 1
fi

# Clean up
echo ""
echo "Cleaning up..."
# Uncomment the next line if you want to remove the executable after running
# rm -f $executable_name

echo "Done!"