#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/navigation-online-profile-v1-20261002
cd "$BASE/code"
export PYTHONPATH="$BASE/matplotlib3109:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/${SLURM_ARRAY_TASK_ID:-bounded}"
export MPLCONFIGDIR="$BASE/matplotlib/${SLURM_ARRAY_TASK_ID:-bounded}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
nvidia-smi
lscpu
/oscar/scratch/sboughan/venvs/se-step6/bin/python scripts/benchmarks/online_stepping/navigation_profile.py --env SimplePointMaze --batches 1 16 64 --repeats 3 --output "$BASE/bounded-v2/SimplePointMaze.json"
