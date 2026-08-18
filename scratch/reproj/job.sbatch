#!/bin/bash
#SBATCH --job-name=reproj
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-068,gb-west-a1-gpu-070,gb-west-a1-gpu-082,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj/%x_%j.log

# usage: sbatch --job-name=<tag> job.sbatch <venv> <script.py> [args...]
VENV=$1; shift
SCRIPT=$1; shift
R=/mnt/vast/home/randall.balestriero/code/stable-envs
P=/mnt/vast/home/randall.balestriero/venvs/$VENV/bin/python
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy
export STABLE_ENVS_HUMANOIDBENCH=1
export PYTHONPATH=$R
export PYTHONUNBUFFERED=1
export WARP_CACHE_PATH=/mnt/vast/home/randall.balestriero/.cache/warp
cd $R
echo "node=$(hostname) venv=$VENV script=$SCRIPT args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
stdbuf -o0 -e0 $P -X faulthandler "$SCRIPT" "$@" 2>&1
