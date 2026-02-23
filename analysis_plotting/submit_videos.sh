#!/bin/bash
#SBATCH --job-name=bc_videos
#SBATCH --output=logs/videos_%j.out
#SBATCH --error=logs/videos_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
module load anaconda/2023a

# Activate your conda environment - UPDATE THIS WITH YOUR ENV NAME
source activate bc-irf  # or whatever environment has xarray, geopandas, cartopy, etc.

# If you need to find your environment name, run: conda env list
# Common names: base, myenv, geo_env, etc.

# Run the video generation script
#python videos_shutdowns.py
python videos.py

echo "Video generation complete!"
