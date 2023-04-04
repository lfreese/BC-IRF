#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=edr


['MALAYSIA', 'INDONESIA', 'VIETNAM', 'SUM']

sbatch shutdowns_GAINS.py --start_year 10 --end_year 40 --type 'weighted_co2' --country 'MALAYSIA' 
sbatch shutdowns_GAINS.py --start_year 10 --end_year 40 --type 'weighted_co2' --country 'INDONESIA'
sbatch shutdowns_GAINS.py --start_year 10 --end_year 40 --type 'weighted_co2' --country 'VIETNAM' ##submitted as a test
sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'annual_co2' --country 'MALAYSIA'
sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'annual_co2' --country 'INDONESIA'
sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'annual_co2' --country 'VIETNAM'
sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'age_retire' --country 'MALAYSIA'
sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'age_retire' --country 'INDONESIA'
sbatch shutdowns_GAINS.py --start_year 0 --end_year 40 --type 'age_retire' --country 'VIETNAM'

