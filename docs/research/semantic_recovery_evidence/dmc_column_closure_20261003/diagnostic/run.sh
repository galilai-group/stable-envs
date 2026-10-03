#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/quadruped-export-diagnostic-v1-20261003
cd /oscar/scratch/sboughan/se-semantic-recovery/dmc-online-column-v2-20261002/code
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
nvidia-smi
lscpu
"$PY" "$BASE/diagnose_dmc_export_frames.py" --domain quadruped --reference-adapters /oscar/scratch/sboughan/se-semantic-recovery/dmc-online-baseline-20261002/code/stable_envs/_adapters --devices cuda --verify-gl --batches 1 --output "$BASE/results/cuda.json"
