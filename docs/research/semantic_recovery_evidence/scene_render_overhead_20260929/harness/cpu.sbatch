#!/bin/bash
#SBATCH --job-name=se-ogb-render-overhead-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=70,72%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/cpu-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/cpu-candidate/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/scene-render-overhead-20260929/cpu-candidate "$SLURM_ARRAY_TASK_ID"
