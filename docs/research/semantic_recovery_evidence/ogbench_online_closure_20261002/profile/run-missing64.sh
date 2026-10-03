#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-online-profile-v1-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/missing64-$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
nvidia-smi
lscpu
FAMILIES=(Cube Cube Scene Scene)
DEVICES=(cpu cuda cpu cuda)
/oscar/scratch/sboughan/venvs/se7-ogb/bin/python scripts/benchmarks/online_stepping/ogbench_profile.py --family "${FAMILIES[$SLURM_ARRAY_TASK_ID]}" --devices "${DEVICES[$SLURM_ARRAY_TASK_ID]}" --batches 64 --output "$BASE/missing64/${FAMILIES[$SLURM_ARRAY_TASK_ID]}-${DEVICES[$SLURM_ARRAY_TASK_ID]}.json"
