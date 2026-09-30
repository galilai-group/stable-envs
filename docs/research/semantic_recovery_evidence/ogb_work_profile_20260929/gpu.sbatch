#!/bin/bash
#SBATCH --job-name=se-ogb-work-profile
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=28,30%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogb-work-profile-20260929/gpu-%A_%a.log
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/ogb-work-profile-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/ogb-work-profile-20260929 "$SLURM_ARRAY_TASK_ID"
