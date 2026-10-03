#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/classic-online-column-v2-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_CACHE_HOME"
nvidia-smi
lscpu
DOMAINS=(cartpole pendulum acrobot mountain_car mountain_car_continuous)
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
"$PY" -m pytest -q tests/unit/test_classic_state_export.py -k "${DOMAINS[$SLURM_ARRAY_TASK_ID]}"
"$PY" scripts/benchmarks/online_stepping/classic_state_export.py --domain "${DOMAINS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results/${DOMAINS[$SLURM_ARRAY_TASK_ID]}.json"
