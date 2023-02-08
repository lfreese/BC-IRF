#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=edr



sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 2 --end_year 4 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 4 --end_year 6 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 6 --end_year 8 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 8 --end_year 10 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 10 --end_year 12 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 12 --end_year 14 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 14 --end_year 16 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 16 --end_year 18 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 18 --end_year 20 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 20 --end_year 22 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 22 --end_year 24 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 24 --end_year 26 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 26 --end_year 28 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 28 --end_year 30 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 30 --end_year 32 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 32 --end_year 34 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 34 --end_year 36 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 36 --end_year 38 --type 'weighted_co2' --country 'CAMBODIA'
sbatch shutdowns_GAINS.py --start_year 38 --end_year 40 --type 'weighted_co2' --country 'CAMBODIA'




# for f in C_out*_0.nc; do mv "$f" "${f/0_2_/}"; done
# for f in C_out*_2.nc; do mv "$f" "${f/2_4_/}"; done
# for f in C_out*_4.nc; do mv "$f" "${f/4_6_/}"; done
# for f in C_out*_6.nc; do mv "$f" "${f/6_8_/}"; done
# for f in C_out*_8.nc; do mv "$f" "${f/8_10_/}"; done
# for f in C_out*_10.nc; do mv "$f" "${f/10_12_/}"; done
# for f in C_out*_12.nc; do mv "$f" "${f/12_14_/}"; done
# for f in C_out*_14.nc; do mv "$f" "${f/14_16_/}"; done
# for f in C_out*_16.nc; do mv "$f" "${f/16_18_/}"; done
# for f in C_out*_18.nc; do mv "$f" "${f/18_20_/}"; done
# for f in C_out*_20.nc; do mv "$f" "${f/20_22_/}"; done
# for f in C_out*_22.nc; do mv "$f" "${f/22_24_/}"; done
# for f in C_out*_24.nc; do mv "$f" "${f/24_26_/}"; done
# for f in C_out*_26.nc; do mv "$f" "${f/26_28_/}"; done
# for f in C_out*_28.nc; do mv "$f" "${f/28_30_/}"; done
# for f in C_out*_30.nc; do mv "$f" "${f/30_32_/}"; done
# for f in C_out*_32.nc; do mv "$f" "${f/32_34_/}"; done
# for f in C_out*_34.nc; do mv "$f" "${f/34_36_/}"; done
# for f in C_out*_36.nc; do mv "$f" "${f/36_38_/}"; done
# for f in C_out*_38.nc; do mv "$f" "${f/38_40_/}"; done
