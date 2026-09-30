#!/bin/bash
#SBATCH --job-name=se-cube-context-budget
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/cube-context-budget-20260929/gpu-%j.log
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
status=0
for variant in baseline candidate; do
 /usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/cube-context-budget-20260929/$variant/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/cube-context-budget-20260929/$variant 28 || status=1
done
exit "$status"
