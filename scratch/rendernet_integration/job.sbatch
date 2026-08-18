#!/bin/bash
#SBATCH --job-name=rnint
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/%x_%j.out
set -u
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=. PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/inductor_cache
export TRITON_CACHE_DIR=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/triton_cache
cd /mnt/vast/home/randall.balestriero/code/stable-envs
echo "node $(hostname)  args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
stdbuf -o0 -e0 /mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python -X faulthandler "$@" 2>&1
