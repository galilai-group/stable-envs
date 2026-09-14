#!/bin/bash
#SBATCH --job-name=se6-raycast3d-bakelt
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2c' item 3: which plate did Step 4 freeze, and is the renderer's lifetime the mechanism.
# Report only. No renderer-lifetime change is committed from this job.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY scratch/step6/raycast3d/bake_lifetime_probe.py --device cuda --bakes 8 --max-fresh 6 \
    --configs AntmazeTeleportExplore AntmazeGiantNavigate HumanoidmazeMediumNavigate \
    --scenes ant/teleport ant/giant humanoid/medium \
    --out "$OUT/bake_lifetime_probe.json" || echo "LT rc=$?"
echo "BAKELT done"
