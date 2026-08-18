#!/bin/bash
#SBATCH --job-name=se_pusht_probe
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/code/stable-envs/examples/lewm_pusht/logs/probe-%j.out
set -x
EX=/mnt/vast/home/randall.balestriero/code/stable-envs/examples/lewm_pusht
cd "$EX"
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
PY=/mnt/vast/home/randall.balestriero/miniconda3/envs/test/bin/python
$PY parity_probe.py --n "${N:-48}" --size 224 \
    --out "$EX/results/parity_224.json" --png "$EX/results/parity_224.png"
