#!/bin/bash
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-process-v2-20261002
cd "$BASE/code"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache/${SLURM_ARRAY_TASK_ID:-unit}"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
nvidia-smi
lscpu
DOMAINS=(acrobot pendulum cartpole reacher cheetah hopper walker ball_in_cup finger manipulator quadruped humanoid)
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark dmc_process_probe --domain "${DOMAINS[$SLURM_ARRAY_TASK_ID]}" --output "$BASE/gates/${DOMAINS[$SLURM_ARRAY_TASK_ID]}" --gates-only
