#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=edr

sbatch shutdowns_individual_plants_single_year.py --country 'MALAYSIA' 
sbatch shutdowns_individual_plants_single_year.py --country 'INDONESIA' 
sbatch shutdowns_individual_plants_single_year.py --country 'VIETNAM' 
sbatch shutdowns_individual_plants_single_year.py --country 'CAMBODIA' 
