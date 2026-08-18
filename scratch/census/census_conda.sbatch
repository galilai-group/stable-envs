#!/bin/bash
#SBATCH --job-name=se_census_conda
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/conda_%j.log

R=/mnt/vast/home/randall.balestriero/code/stable-envs
C=/mnt/vast/home/randall.balestriero/miniconda3/envs
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=$R
cd $R

for e in test robocasa xminigrid habitat; do
  P=$C/$e/bin/python
  [ -x "$P" ] || { echo "MISSING $e"; continue; }
  echo "### conda:$e"
  timeout 900 $P $R/scratch/census/census.py 2>&1 | tail -3
done
