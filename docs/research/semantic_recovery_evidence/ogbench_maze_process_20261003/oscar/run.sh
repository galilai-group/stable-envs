#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-maze-process-v1-20261003
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
nvidia-smi
lscpu
"$PY" -m pytest -q tests/unit/test_ogbench_maze_process.py tests/unit/test_ogbench_semantics_oracle.py tests/unit/test_ogbench_state_export.py -k 'not Cube and not Scene' --junitxml="$BASE/tests.xml"
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark ogbench_maze_process --domain Maze --output "$BASE/results"
