#!/bin/bash
#SBATCH --job-name=mwfast
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/mwfast/%x_%j.out

set -euo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=.:scratch/mwfast:scratch/altrender/rendernet:scratch/altrender
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONUNBUFFERED=1
echo "node $SLURMD_NODENAME  job $SLURM_JOB_ID  $*"
srun /mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python "$@"
