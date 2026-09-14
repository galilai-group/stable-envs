#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6gt
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The fast-path suite after the cover-compaction case was re-pointed at _composite_window.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$WS $PY -m pytest tests/gpu/test_fast_paths.py -q -p no:randomly 2>&1 | tail -8
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -6
echo "R6_GT done"
