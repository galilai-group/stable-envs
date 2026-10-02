#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/fetch-family-3090-followup-20261002
SOURCE=/oscar/scratch/sboughan/se-semantic-recovery/fetch-family-3090-20261002
cd "$SOURCE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
TASKS=(Push Slide PickAndPlace PickAndPlace)
SUFFIXES=(Dict Dict Dense Dict)
nvidia-smi
lscpu
"$PY" scripts/benchmarks/online_stepping/fetch_family.py --task "${TASKS[$SLURM_ARRAY_TASK_ID]}" --suffixes "${SUFFIXES[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results/${TASKS[$SLURM_ARRAY_TASK_ID]}${SUFFIXES[$SLURM_ARRAY_TASK_ID]}.json"
