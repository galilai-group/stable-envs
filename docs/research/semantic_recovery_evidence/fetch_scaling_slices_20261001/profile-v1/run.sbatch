#!/bin/bash
#SBATCH --job-name=se-fetch-scaling
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
set -euo pipefail
stage=/oscar/scratch/sboughan/se-semantic-recovery/fetch-scaling-20261001-v1
cd "$stage"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
lscpu
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python fetch_scaling.py --output result.json
