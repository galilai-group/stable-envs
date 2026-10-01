#!/bin/bash
#SBATCH --job-name=se-hybrid-profile
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cascade
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/fetch-hybrid-attribution-20261001-v1/terminal.log
set -euo pipefail
cd /oscar/scratch/sboughan/se-semantic-recovery/fetch-hybrid-attribution-20261001-v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
lscpu
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python fetch_hybrid_profile.py --output result.json
