#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-online-baseline-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/$SLURM_ARRAY_TASK_ID"
DOMAINS=(acrobot pendulum cartpole reacher cheetah hopper walker ball_in_cup finger manipulator quadruped humanoid)
nvidia-smi
lscpu
/oscar/scratch/sboughan/venvs/se6-verify-fetch311/bin/python scripts/benchmarks/online_stepping/dmc_profile.py --domain "${DOMAINS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/results/${DOMAINS[$SLURM_ARRAY_TASK_ID]}.json"
