#!/bin/bash
#SBATCH --job-name=ogb-scene-default
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-scene-default-v2-20261003/gate-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/ogbench-scene-default-v2-20261003
cd /oscar/scratch/sboughan/se-semantic-recovery/ogbench-manipulation-default-v1-20261003/code
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
PY=/oscar/scratch/sboughan/venvs/se7-ogb/bin/python
nvidia-smi
lscpu
FAMILY=Scene
"$PY" -m pytest -q tests/unit/test_ogbench_execution.py --junitxml="$BASE/tests-$FAMILY.xml"
"$PY" scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --benchmark ogbench_manipulation_process --domain "$FAMILY" --gates-only --production --output "$BASE/results/$FAMILY"
