#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/step-masking-v2-20261003
cd "$BASE/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$MPLCONFIGDIR"
nvidia-smi
lscpu
export PYTHONPATH="$PWD"
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
TASKS=(Reach Push Slide PickAndPlace)
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
 "$PY" -m pytest -q tests/unit/test_fetch_process_adapter.py tests/unit/test_fetch_execution.py --junitxml="$BASE/fetch-tests.xml"
fi
"$PY" scripts/benchmarks/online_stepping/fetch_pipeline.py --task "${TASKS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/fetch/${TASKS[$SLURM_ARRAY_TASK_ID]}.json" --production --gates-only
