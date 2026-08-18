#!/bin/bash
#SBATCH --job-name=rnetgen
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-058,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/%x_%A_%a.out
set -u
LIST=${TASKLIST:?set TASKLIST}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=.:scratch/altrender:scratch/altrender/rendernet PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
cd /mnt/vast/home/randall.balestriero/code/stable-envs
echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  cmd: $LINE"
stdbuf -o0 -e0 /mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python -X faulthandler $LINE 2>&1
