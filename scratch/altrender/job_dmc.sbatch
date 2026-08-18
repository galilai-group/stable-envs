#!/bin/bash
#SBATCH --job-name=dmcbudget
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/%x_%A_%a.out
set -u
LIST=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/dmc_cells.txt
CELL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
OUT=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export PYTHONPATH=.:scratch/altrender PYTHONUNBUFFERED=1
cd /mnt/vast/home/randall.balestriero/code/stable-envs
echo "node $(hostname) cell $CELL loadavg $(cat /proc/loadavg)"
stdbuf -o0 -e0 /mnt/vast/home/randall.balestriero/venvs/dmc/bin/python -X faulthandler \
    scratch/altrender/probe_budget_frontier.py "$CELL" --no-jpeg \
    --dump-official "$OUT/official_$CELL.npy" "$@" 2>&1
