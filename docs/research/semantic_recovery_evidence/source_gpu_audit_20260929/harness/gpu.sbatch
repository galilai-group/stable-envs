#!/bin/bash
#SBATCH --job-name=se-source-gpu-audit
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --constraint=geforce3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=34,40%2
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-navgpu/gpu-%A_%a.log
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PAIRED_PILOT=1
unset PAIRED_PILOT
/usr/bin/python3 /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-navgpu/profile_oscar.py /oscar/scratch/sboughan/se-semantic-recovery/renderer-paired-df80c9c5-20260929-navgpu "$SLURM_ARRAY_TASK_ID"
exit $?
