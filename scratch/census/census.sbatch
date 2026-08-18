#!/bin/bash
#SBATCH --job-name=se_census
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/census_%j.log

R=/mnt/vast/home/randall.balestriero/code/stable-envs
V=/mnt/vast/home/randall.balestriero/venvs
export MUJOCO_GL=egl
export STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=$R

for venv in se-envs se-tiles se-robocasa se-hb dmc se-fetch se-jumanji se-pgx se-oc se-ocjax se3d senvdiff pixnat mjwarp mjgpu se-docs se-brand; do
  P=$V/$venv/bin/python
  [ -x "$P" ] || { echo "MISSING $venv"; continue; }
  echo "### $venv"
  timeout 600 $P $R/scratch/census/census.py 2>&1 | tail -3
done
