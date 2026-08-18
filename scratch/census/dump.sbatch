#!/bin/bash
#SBATCH --job-name=se_dump
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/census/dump_%x_%j.log
VENV=$1; shift
R=/mnt/vast/home/randall.balestriero/code/stable-envs
P=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1 PYTHONPATH=$R
cd $R/examples
$P dump_frames.py "$@"
