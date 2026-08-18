#!/bin/bash
#SBATCH --job-name=msaaarr
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/msaanet/%x_%A_%a.out

set -euo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=.
export MUJOCO_GL=egl
export SDL_VIDEODRIVER=dummy
export PYTHONUNBUFFERED=1
readarray -t ARGS < "$MSAA_ARGS"
LINE="${ARGS[$SLURM_ARRAY_TASK_ID]}"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID loadavg=$(cat /proc/loadavg)"
echo "cmd: $LINE"
eval "~/venvs/dmc/bin/python scratch/msaa/$LINE"
echo "done loadavg=$(cat /proc/loadavg)"
