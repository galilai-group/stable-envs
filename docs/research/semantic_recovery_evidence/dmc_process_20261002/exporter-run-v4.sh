#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-online-column-v2-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
DOMAINS=(acrobot pendulum cartpole reacher cheetah hopper walker ball_in_cup finger manipulator quadruped humanoid)
nvidia-smi
lscpu
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
 "$PY" -m pytest -q tests/unit/test_dmc_state_export.py tests/unit/test_dmc_first_family_adapter.py tests/unit/test_dmc_aligned_adapter.py tests/unit/test_dmc_continuation_adapter.py tests/unit/test_dmc_humanoid_adapter.py
fi
"$PY" /oscar/scratch/sboughan/se-semantic-recovery/dmc-process-v3-20261002/code/scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --reference-adapters /oscar/scratch/sboughan/se-semantic-recovery/dmc-online-baseline-20261002/code/stable_envs/_adapters --domain "${DOMAINS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results-v4/${DOMAINS[$SLURM_ARRAY_TASK_ID]}"
