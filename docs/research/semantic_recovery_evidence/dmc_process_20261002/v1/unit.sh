#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-process-v1-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/${SLURM_ARRAY_TASK_ID:-unit}"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
nvidia-smi
lscpu
"$PY" -m pytest -q -rs tests/unit/test_dmc_process_adapter.py tests/unit/test_dmc_state_export.py tests/unit/test_dmc_first_family_adapter.py tests/unit/test_dmc_aligned_adapter.py tests/unit/test_dmc_continuation_adapter.py tests/unit/test_dmc_humanoid_adapter.py
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python -m pytest -q -rs tests/unit/test_fetch_process_adapter.py tests/unit/test_fetch_execution.py tests/unit/test_step_capture.py
