#!/bin/bash
#SBATCH --partition=fdr

########## Jan submissions #######
sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_16x_pulse --location SEA --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location SEA  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Indo  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Indo  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Viet  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Viet  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Malay  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Malay  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location all_countries  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location all_countries  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Cambod  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Cambod  --month_step Jan

########## base 2x ###########
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step NW_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step NW_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step SE_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step SE_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step Center_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step Center_pulse

########### base 2x ###########
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_base  --location template  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_base  --location template  --month_step Apr
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_base  --location template  --month_step Apr
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 7   --model_run stretch_base  --location template  --month_step July
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 8  --model_run stretch_base  --location template  --month_step July
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 10   --model_run stretch_base  --location template  --month_step Oct
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 11  --model_run stretch_base  --location template  --month_step Oct



########## time dif ###########
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan6_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan6_pulse


sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_base  --location template  --month_step Jan6_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan6_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan6_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3  --model_run stretch_base  --location template  --month_step Jan6_pulse

###### LEFT OFF ######

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan11_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan11_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_base  --location template  --month_step Jan11_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan11_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan11_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3  --model_run stretch_base  --location template  --month_step Jan11_pulse

####
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan16_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan16_pulse


sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_base  --location template  --month_step Jan16_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan16_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan16_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3  --model_run stretch_base  --location template  --month_step Jan16_pulse

####
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan21_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan21_pulse


sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_base  --location template  --month_step Jan21_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan21_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan21_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3  --model_run stretch_base  --location template  --month_step Jan21_pulse

####
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan26_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan26_pulse


sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_base  --location template  --month_step Jan26_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan26_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3   --model_run stretch_2x_pulse  --location Jan_uncertainty_Indo  --month_step Jan26_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 3  --model_run stretch_base  --location template  --month_step Jan26_pulse


# sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location all_countries_cos --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 2 --model_run stretch_step --location all_countries_cos --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location all_countries_add --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 2 --model_run stretch_step --location all_countries_add --month_step Jan


sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location all_countries_add --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 2 --model_run stretch_step --location all_countries_add --month_step Jan


sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location Indo_cos --month_step Jan
sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 2 --model_run stretch_step --location Indo_cos --month_step Jan
