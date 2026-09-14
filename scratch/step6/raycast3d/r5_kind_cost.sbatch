#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5cost
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-5 step 1: the per-kind cost table the predicate is derived from. One thread, because the
# certification rows this has to fix are one-thread rows, and because a thread pool would time the
# scheduler rather than the arithmetic.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r5_kind_cost.py --threads 1 \
    --out "$WS/examples/results/step6/raycast3d/experiments/r5_kind_cost.json"
echo "R5_COST done"
