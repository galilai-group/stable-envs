#!/bin/bash
#SBATCH --job-name=se-scene-kinematics-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/cpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/candidate/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/candidate 72
