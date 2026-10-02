#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-online-baseline-v2-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
DOMAINS=(acrobot pendulum cartpole reacher cheetah hopper walker ball_in_cup finger manipulator quadruped humanoid)
nvidia-smi
lscpu
"$PY" scripts/benchmarks/online_stepping/dmc_profile.py --domain "${DOMAINS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results/${DOMAINS[$SLURM_ARRAY_TASK_ID]}.json"
