#!/bin/bash
#SBATCH --job-name=se-structural-gpu
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --array=30
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/structural-render-20260930/gpu-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/structural-render-20260930/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/structural-render-20260930 "$SLURM_ARRAY_TASK_ID"
