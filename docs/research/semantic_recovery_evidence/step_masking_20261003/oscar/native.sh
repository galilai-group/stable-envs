#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/step-masking-v1-20261003
cd "$BASE/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/${SLURM_JOB_ID}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$MPLCONFIGDIR"
nvidia-smi
lscpu
export PYTHONPATH="/oscar/scratch/sboughan/se-semantic-recovery/navigation-online-profile-v1-20261002/matplotlib3109:$PWD"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
NAMES=(TwoRoom Piecewise SimplePointMaze)
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
 "$PY" -m pytest -q tests/unit/test_step_masking.py tests/unit/test_step_capture.py tests/unit/test_native_navigation_semantics.py --junitxml="$BASE/native-tests.xml"
fi
"$PY" scripts/benchmarks/online_stepping/step_masking.py --env "${NAMES[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/native/${NAMES[$SLURM_ARRAY_TASK_ID]}.json"
