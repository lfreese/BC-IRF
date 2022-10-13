#!/bin/bash
#SBATCH --partition=edr

########## Jan submissions #######
#sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location SEA --month_step Jan
#sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location SEA  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_step  --location Indo  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location Indo  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_step  --location Viet  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location Viet  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_step  --location Malay  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location Malay  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_step  --location all_countries  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location all_countries  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_step  --location Cambod  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location Cambod  --month_step Jan

########## base 2x ###########
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step NW_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step NW_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step SE_pulse
sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step SE_pulse

sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1   --model_run stretch_2x_pulse  --location loc_uncertainty_Indo  --month_step Center_pulse

# ########## base 2x ###########
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_base  --location template  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_base  --location template  --month_step Apr


# ########## submissions 2x #######
# sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 4  --model_run stretch_16x_pulse --location SEA --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_16x_pulse  --location SEA  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_16x_pulse  --location Indo  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_16x_pulse  --location Indo  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_16x_pulse  --location Viet  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_16x_pulse  --location Viet  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_16x_pulse  --location Malay  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_16x_pulse  --location Malay  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_16x_pulse  --location all_countries  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_16x_pulse  --location all_countries  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 4   --model_run stretch_16x_pulse  --location Cambod  --month_step Apr
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 5  --model_run stretch_16x_pulse  --location Cambod  --month_step Apr


# ########## Jan submissions pulse ########
# sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location SEA_pulse --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location SEA_pulse  --month_step Jan


# ########## Jan submissions add ########
# sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_step --location SEA_addition --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_step  --location SEA_addition  --month_step Jan

# ########## Jan submissions base ########
# sbatch regrid_gchp_stretched.py --cube_res 90 --lat 180 --lon 288 --month 1 --model_run stretch_base --location template --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_base  --location template  --month_step Jan


########## Jan submissions 16x #######
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location all_countries  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location all_countries  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location SEA  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location SEA  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Indo  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Indo  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Viet  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Viet  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Malay  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Malay  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 1  --model_run stretch_16x_pulse  --location Cambod  --month_step Jan
# sbatch regrid_gchp_stretched.py --cube_res 90  --lat 180  --lon 288  --month 2  --model_run stretch_16x_pulse  --location Cambod  --month_step Jan
