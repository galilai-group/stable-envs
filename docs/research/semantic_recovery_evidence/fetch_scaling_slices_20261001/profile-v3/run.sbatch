#!/bin/bash
#SBATCH --job-name=se-fetch-scaling
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
set -euo pipefail
stage=/oscar/scratch/sboughan/se-semantic-recovery/fetch-scaling-20261001-v3
cd "$stage"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
lscpu
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python fetch_scaling.py --output result.json --batches 64 128 256 512 1024 2048 --workers 4 8 --kinds threads async_shared --repeats 3
