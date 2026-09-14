#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5kinds
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
set -x
cd "$WS"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r5_mover_kinds.py \
    --out "$WS/examples/results/step6/raycast3d/experiments/r5_mover_kinds.json"
echo "R5_KINDS done"
