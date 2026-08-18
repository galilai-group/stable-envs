#!/bin/bash
#SBATCH --job-name=altrender
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/%x_%j.out

set -u
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1 PYTHONPATH=.
cd /mnt/vast/home/randall.balestriero/code/stable-envs
PY=${PY:-/mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python}
echo "node $(hostname) py $PY"
export PYTHONPATH=.:scratch/altrender
export PYTHONUNBUFFERED=1
stdbuf -o0 -e0 $PY -X faulthandler "$@" 2>&1
