#!/bin/bash
#SBATCH --job-name=dmc-pipeline-smoke
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-smoke-v2-20261003/gate-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/dmc-pipeline-smoke-v2-20261003
cd "$BASE/code"
PY=/oscar/scratch/sboughan/venvs/se-step6/bin/python
export PIP_CACHE_DIR="$BASE/pip-cache"
"$PY" -m pip install --no-deps --target "$BASE/extras" --report "$BASE/dependency-install.json" mujoco-warp==3.12.0 warp-lang==1.17.0
export PYTHONPATH="$BASE/extras:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
nvidia-smi
lscpu
/oscar/scratch/sboughan/venvs/se-step6/bin/python scripts/benchmarks/online_stepping/dmc_state_export_oscar.py --domain humanoid --benchmark dmc_pipeline --batches 64 --output "$BASE/results/Humanoid"
