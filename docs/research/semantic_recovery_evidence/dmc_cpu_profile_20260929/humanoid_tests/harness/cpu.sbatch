#!/bin/bash
#SBATCH --job-name=se-dmc-humanoid-tests
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --array=66
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-tests-20260929/cpu-%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-tests-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/dmc-humanoid-chunk-tests-20260929 "$SLURM_ARRAY_TASK_ID"
