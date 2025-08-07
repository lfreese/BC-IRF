# Black-carbon-radiative-forcing-and-climate-response-tool
This repository provides a fortran-based tool which computes an estimate for the radiative forcings and associated global-mean temperature response,
based on the region and annual or monthly BC emission rates given by the user.

This tool is based on the research reported in the manuscript

Räisänen, P., Merikanto, J., Makkonen, R., Savolahti, M., Kirkevåg, A., Sand, M., Seland, Ø., and Partanen, A.-I.: Mapping the dependence of BC 
radiative forcing on emission region and season, submitted to Atmos. Chem. Phys., 2022.

and data available at https://fmi.b2share.csc.fi/records/6808480a473e437fa56f6bf05e8d6a8b
(DOI: 10.23728/fmi-b2share.6808480a473e437fa56f6bf05e8d6a8b)

This tool is provided in hope it will be useful. At the same time, potential users of the tool should be aware of its limitations. The tool is based on 
results  from a single climate model, and therefore, the estimates are subject to significant quantitative uncertainties. This is especially true for the 
temperature  response, which was not (and indeed would not be feasible to be) simulated explicitly but was estimated using so-called Regional Climate
Sensitivity coefficients. For details of the method, please consult the manuscript Räisänen et al. (2022) mentioned above.

The following files are included in the repository:

-------------------------
1. README.md

= this file

-------------------------
2. LICENSE

The MIT license is applied.

-------------------------------------------------------------
3. BC_forcing_and_climate_response_normalized_by_emissions.nc

This file contains four primary fields that can be used for estimating the radiative forcing and climate response (i.e., global annual-mean equilibrium
temperature change) associated with BC emissions, as discussed in Räisänen et al. (2022):

DIRSF_TOA = global annual-mean top-of-the-atmosphere direct specific radiative forcing due to BC absorption in air, normalized by emissions [TJ kg^-1]

SNOWSF_TOA = Global annual-mean top-of-the-atmosphere specific radiative forcing due to BC absorption in snow, normalized by emissions [TJ kg^-1]

DT_NORM_DRF = global-annual mean temperature response due to BC absorption in air, normalized by annual-mean emission rate [K (kg s^-1)^-1]

DT_NORM_SNOWRF = global-annual mean temperature response due to BC absorption in snow, normalized by annual-mean emission rate [K (kg s^-1)^-1]

These quantities are provided for 192 latitude-longitude regions (ca. 11.4 deg x 30 deg) for BC emissions and for five seasonal distributions 
of emissions:
- ANN: constant emissions throughout the year; 
- DJF, MAM, JJA and SON: emissions confined to a single meteorological season

The specific radiative forcing fields DIRSF_TOA and SNOWSF_TOA are derived from the NorESM1-Happi experiments described in Räisänen et al. (2022). The 
estimated normalized climate responses (DT_NORM_DRF and DT_NORM_SNOWRF) combine annual-mean radiative forcings from the NorESM1-Happi experiments with
Regional Climate Sensitivity coefficients (RCSs) derived from experiments with other climate models, as detailed in Section 6.2 and Appendix A of
Räisänen et al. (2022).

The RCSs are only available for experiments in which BC emissions occur throughout the annual cycle. However, in the NetCDF file, climate response
estimates are also provided for the cases with single-season BC emissions. To accomplish this, the annual-mean BC radiative forcings were computed first,
and these were then combined with the RCSs. This involves the assumption that the RCSs do not depend on the seasonal distribution of BC emissions and
radiative forcing. This assumption remains unvalidated and it may not be entirely realistic.  Therefore, one should be especially cautious about
the temperature responses estimated for single-season emissions.

----------------------------------------------
4. calculate_forcing_and_climate_response.f90

This fortran program is provided for demonstrating how radiative forcings and climate responses can be calculated using the above NetCDF file.
The program computes the following quantities, based on the geographical region (latitude and longitude limits) and the (annual-mean or monthly) 
BC emission rate(s) given by the user:

i) global-annual mean radiative forcings at the top-of-the atmosphere [W m-2]

  DRF_TOA    = direct radiative forcing due to BC absorption in air
  
  SNOWRF_TOA = radiative forcing due to BC in snow
  
  SUM_RF_TOA = DRF_TOA + SNOWRF_TOA

ii) global-annual mean temperature responses [K]

  DT_DRF    = temperature response to BC absorption in air
  
  DT_SNOWRF = temperature response to BC absorption in snow
  
  DT_SUM    = DT_DRF + DT_SNOWRF

Note that, even in the case of a uniform emission rate over the year, the results depend slightly on whether the emission rate is input as annual-mean or
monthly-mean values. The reason for this is that in the case of annual-mean input, the values are based on the ANN experiments in Räisänen et al. (2022),
while for monthly input, the results from the seasonal emission experiments (DJF, MAM, JJA and SON) are used. This has only a marginal impact on DRF, but
a slightly larger impact on SNOWRF.

The program makes use of the Fortran NetCDF library, which needs to be linked to the program at the time of compiling. The way this is done may depend
on the computer system and compiler in use, and it is not feasible to give here all-encompassing instructions. Two examples are given here:

1) For the "puhti" supercomputer at Center for Scientific Computing, Finland, using an Intel fortran compiler:

> module load netcdf-fortran
> ifort calculate_forcing_and_climate_response.f90 -lnetcdff

2) For a Dell laptop, linux Ubuntu 20.04, gfortran compiler

> f95 calculate_forcing_and_climate_response.f90 -I/usr/include -lnetcdff

where the path for the include directory (I) was found with the command

>nf-config --all

The executable can then be run with the standard ommand

> ./a.out > output.txt

------------------
5. input_data.asc

This file provides the input data for the fortran program. The input data include the following:

- LONGITUDE_LIMITS: the western and eastern boundary of the BC emission region defined by the user [allowed values between -360 and 360 deg]
- LATITUDE_LIMITS: the southern and northern boundary of the BC emission region defined by the user [allowed values between -90 and 90 deg]
- EMISSION_RATE: BC emission rate, in units of [kg m^-2 s^-1]. The emissions can be given either as a single annual-mean value, or
  as a series of monthly-mean values (12 values)

----------------------
6. output_example.txt

An example of the output produced by the program "calculate_forcing_and_climate_response.f90". This specific example represents the radiative forcing and
temperature response to BC emissions in the region 22-30E, 60-70N, for the year 2015 emissions employed in Räisänen et al. (2022).

##### Lyssa's additions

1. create all data input files (takes a netcdf file of our lats and lons and turns them into the acs format; requires a netcdf file named BC_forcing_and_climate_response_normalized_by_emissions.nc) 

  python extract_lat_lon.py '

2. compile the code 
  ./compile_and_run.sh (if you haven't prepped it for compilation, first do chmod +x copmile_and_run.sh)

3. run for all regions
  ./run_all_regions.sh (again if you haven't yet, do chmod +x run_all_regions.sh first)

4. combine all results into one file
