#!/bin/bash
#SBATCH --job-name=se-dmc-components-v2-cuda
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-components-v2-cuda-20260929/gpu-%j.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
unset PAIRED_PILOT
root=/oscar/scratch/sboughan/se-semantic-recovery/dmc-components-v2-cuda-20260929
for index in 2 22 25; do
  for leg in baseline candidate; do
    /usr/bin/python3 "$root/$leg/profile_oscar.py" "$root/$leg" "$index"
  done
done
