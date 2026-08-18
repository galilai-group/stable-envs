#!/bin/bash
#SBATCH --job-name=rnintcold
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/%x_%j.out
# A COLD Inductor and Triton cache, on purpose. max-autotune picks its convolution kernels by
# benchmarking them, and a benchmark is a measurement of the machine -- so the question this job asks
# is whether the choice, and therefore the pixels, are the same when nothing is remembered. Anything
# else measures the cache.
set -u
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=. PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
COLD=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/cold_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=$COLD/inductor TRITON_CACHE_DIR=$COLD/triton
mkdir -p "$COLD"
cd /mnt/vast/home/randall.balestriero/code/stable-envs
echo "node $(hostname)  cold cache $COLD  args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
stdbuf -o0 -e0 /mnt/vast/home/randall.balestriero/venvs/se-hb/bin/python -X faulthandler "$@" 2>&1
rm -rf "$COLD"
