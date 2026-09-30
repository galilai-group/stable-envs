#!/bin/bash
#SBATCH --job-name=se-scene-gpu-scaling
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/scene-gpu-scaling-20260929/gpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/scene-gpu-scaling-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/scene-gpu-scaling-20260929 30
