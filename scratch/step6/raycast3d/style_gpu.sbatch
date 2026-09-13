#!/bin/bash
#SBATCH --job-name=se6-raycast3d-style
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6C ticket 1, deliverable 4: the correctness diagnosis for the two OGBench cells that fail
# Gate A on check_style_is_state. Measured on cuda because Step 4 measured it on cuda.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv

# The two failing cells, two passing siblings that share their SCENE, and a pointmaze control.
$PY scratch/step6/raycast3d/style_bisect.py --device cuda --batch 64 \
    --configs AntmazeTeleportExplore HumanoidmazeMediumNavigate \
              AntmazeTeleportNavigate HumanoidmazeLargeNavigate PointmazeLargeNavigate \
    --out "$OUT/style_is_state_bisect.json" || echo "BISECT rc=$?"
echo "STYLE done"
