#!/bin/bash
#SBATCH --job-name=se-humanoid-residual-profile
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/humanoid-chunk4-residual-profile-20260929/cpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/humanoid-chunk4-residual-profile-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/humanoid-chunk4-residual-profile-20260929 66
