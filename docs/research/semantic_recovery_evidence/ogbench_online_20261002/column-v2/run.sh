#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-online-column-v2-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
nvidia-smi
lscpu
FAMILIES=(Maze Cube Scene)
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
"$PY" -m pytest -q tests/unit/test_ogbench_state_export.py -k "${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark ogbench_state_export --domain "${FAMILIES[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results/${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
