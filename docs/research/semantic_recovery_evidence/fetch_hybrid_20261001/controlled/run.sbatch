#!/bin/bash
#SBATCH --job-name=se-fetch-hybrid
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cascade
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=00:30:00
set -euo pipefail
stage=/oscar/scratch/sboughan/se-semantic-recovery/fetch-hybrid-20261001-v2
cd "$stage"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
lscpu
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python fetch_hybrid.py --output result.json --batches 64 256 1024 --repeats 5 --pin-workers
