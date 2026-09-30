#!/bin/bash
#SBATCH --job-name=se-scene-kinematics
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/supplemental-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
for variant in supplemental-baseline supplemental-candidate; do
 /usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/$variant/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/scene-render-kinematics-20260929/$variant 30
done
