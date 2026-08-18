#!/bin/bash
#SBATCH --job-name=madrona
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/%x_%j.out

set -u
# madrona_mjx is built against CUDA 12.5.1 and JITs its GPU engine with that toolkit's nvrtc at RUN
# time, so the 12.5 libraries have to be ahead of the driver's 13.x on the loader path.
export CUDA125=/mnt/vast/home/randall.balestriero/envs/cuda125
export LD_LIBRARY_PATH=$CUDA125/lib:${LD_LIBRARY_PATH:-}
# MADRONA_MWGPU_KERNEL_CACHE names a FILE, not a directory: pointed at a directory the
# executor aborts in checkAndLoadMegakernelCache before it ever builds anything.
mkdir -p /mnt/vast/home/randall.balestriero/.cache/madrona
export MADRONA_MWGPU_KERNEL_CACHE=/mnt/vast/home/randall.balestriero/.cache/madrona/mjx.cache
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONPATH=.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
cd /mnt/vast/home/randall.balestriero/code/stable-envs
PY=/mnt/vast/home/randall.balestriero/venvs/madrona/bin/python
echo "node $(hostname)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export PYTHONUNBUFFERED=1
stdbuf -o0 -e0 $PY -X faulthandler "$@" 2>&1
