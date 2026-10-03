#!/bin/bash
#SBATCH --job-name=quadruped-render-trace
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --nodelist=gpu2109
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/oscar/scratch/sboughan/se-semantic-recovery/quadruped-render-history-v1-20261003/gate-%j.log
set -euo pipefail
BASE=/oscar/scratch/sboughan/se-semantic-recovery/quadruped-render-history-v1-20261003
cd /oscar/scratch/sboughan/se-semantic-recovery/dmc-online-column-v2-20261002/code
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export XDG_CACHE_HOME="$BASE/cache"
mkdir -p "$XDG_CACHE_HOME"
nvidia-smi
/oscar/scratch/sboughan/venvs/se-step6/bin/python "$BASE/trace_dmc_render_history.py" --domain quadruped --reference-adapters /oscar/scratch/sboughan/se-semantic-recovery/dmc-online-baseline-20261002/code/stable_envs/_adapters --devices cuda --verify-gl --batches 1 --output "$BASE/results/cuda.json"
