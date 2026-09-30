#!/bin/bash
#SBATCH --job-name=se-source-cuda-proof
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=a5000|a5500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/source-gpu-audit-diagnostic-20260929/gpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/source-gpu-audit-diagnostic-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/source-gpu-audit-diagnostic-20260929 34
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/source-gpu-audit-diagnostic-20260929/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/source-gpu-audit-diagnostic-20260929 40
