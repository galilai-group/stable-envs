#!/bin/bash
#SBATCH --job-name=se-3090-saturation
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-41%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/renderer-saturation-df80c9c5-20260929/gpu-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/renderer-saturation-df80c9c5-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/renderer-saturation-df80c9c5-20260929 "$SLURM_ARRAY_TASK_ID"
