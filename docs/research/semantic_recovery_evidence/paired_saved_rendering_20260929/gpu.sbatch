#!/bin/bash
#SBATCH --job-name=se-paired-full
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-41%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/gpu-%A_%a.log
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-full "$SLURM_ARRAY_TASK_ID"
exit $?
