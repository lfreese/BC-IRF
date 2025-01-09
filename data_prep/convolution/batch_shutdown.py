#!/home/emfreese/anaconda3/envs/gchp/bin/python
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=edr

sbatch shutdowns_individual_plants.py --start_year 10 --end_year 45 --country 'MALAYSIA' 
sbatch shutdowns_individual_plants.py --start_year 10 --end_year 45 --country 'INDONESIA' 
sbatch shutdowns_individual_plants.py --start_year 10 --end_year 45 --country 'VIETNAM' 
sbatch shutdowns_individual_plants.py --start_year 10 --end_year 45 --country 'CAMBODIA' 


sbatch GF_mean_times_pt1.py --region 'Indo'
sbatch GF_mean_times_pt1.py --region 'Malay'
sbatch GF_mean_times_pt1.py --region 'all_countries'
sbatch GF_mean_times_pt1.py --region 'Viet'
sbatch GF_mean_times_pt1.py --region 'Cambod'


sbatch shutdowns_individual_plants.py --start_year 15 --end_year 20 --country 'INDONESIA' 
sbatch shutdowns_individual_plants.py --start_year 30 --end_year 45 --country 'INDONESIA' 


sbatch shutdowns_individual_plants_single_year.py --country 'MALAYSIA' 
sbatch shutdowns_individual_plants_single_year.py --country 'INDONESIA' 
sbatch shutdowns_individual_plants_single_year.py --country 'VIETNAM' 
sbatch shutdowns_individual_plants_single_year.py --country 'CAMBODIA' 
