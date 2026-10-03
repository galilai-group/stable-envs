#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-manipulation-process-v1-20261003
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
nvidia-smi
lscpu
FAMILIES=(Cube Scene)
FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
"$PY" -m pytest -q tests/unit/test_ogbench_manipulation_process.py -k "$FAMILY" --junitxml="$BASE/$FAMILY-tests.xml"
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark ogbench_manipulation_process --domain "$FAMILY" --output "$BASE/results/$FAMILY"
