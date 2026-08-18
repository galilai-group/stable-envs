#!/bin/bash
#SBATCH --job-name=se3d_ceil2
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/ceil2_%A_%a.log
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
echo "node $(hostname) loadavg $(cat /proc/loadavg)"
/mnt/vast/home/randall.balestriero/venvs/dmc/bin/python -u scratch/perf3d/ceiling.py \
  --batch 1024 --size 256 \
  --json /mnt/vast/home/randall.balestriero/logs/stable_envs_logs/perf/ceiling_${SLURM_ARRAY_TASK_ID}.json
