#!/bin/bash
#SBATCH --job-name=se-hwbench
#SBATCH --partition=open_research
#SBATCH --account=open_research
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gb-west-a1-gpu-047,gb-west-a1-gpu-070,gb-west-a1-gpu-120,gb-west-a1-gpu-184
#SBATCH --output=/mnt/vast/home/randall.balestriero/se_logs/hwbench_%j.out
set -uo pipefail
cd /mnt/vast/home/randall.balestriero/code/stable-envs
export PYTHONPATH=. SDL_VIDEODRIVER=dummy
/mnt/vast/home/randall.balestriero/venvs/se-envs/bin/python scratch/highway/bench.py \
  Highway,Merge,Roundabout,Intersection 4096,16384 2>&1 | grep -vE "Deprecation|logger"
echo BENCH_DONE
