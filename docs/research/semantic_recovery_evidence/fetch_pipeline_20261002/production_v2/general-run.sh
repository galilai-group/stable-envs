#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/fetch-pipeline-general-v2-20261002
cd /oscar/scratch/sboughan/se-semantic-recovery/fetch-pipeline-production-v2-20261002/code
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
PY=/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python
TASKS=(Reach Push Slide PickAndPlace)
nvidia-smi
lscpu
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    "$PY" -m pytest -q -rs tests/unit/test_fetch_process_adapter.py tests/unit/test_fetch_execution.py tests/unit/test_step_capture.py
fi
"$PY" scripts/benchmarks/online_stepping/fetch_pipeline.py --task "${TASKS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results/${TASKS[$SLURM_ARRAY_TASK_ID]}.json" --production
