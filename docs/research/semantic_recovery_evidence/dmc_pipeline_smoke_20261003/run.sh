#!/bin/bash
#SBATCH --job-name=dmc-pipeline-smoke
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-smoke-v1-20261003/gate-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-smoke-v1-20261003
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
nvidia-smi
lscpu
/oscar/scratch/sboughan/venvs/se-step6/bin/python scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --domain humanoid --benchmark dmc_pipeline --batches 64 --output "$BASE/results/Humanoid"
