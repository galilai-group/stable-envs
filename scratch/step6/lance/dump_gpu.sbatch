#!/bin/bash
#SBATCH --job-name=se6-lance-dump
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Phase 1 of the Step-6 storage baseline: render the 36 registered_default configs with OUR GPU
# renderer at each env's own certified size and encode them at JPEG q95, exactly as
# stable_worldmodel's lance format writes a corpus. Fetch is dumped by dump_fetch.sbatch in the
# 3.11 venv; everything else runs here in se-step5 (mujoco 3.12 + mujoco_warp + stable_worldmodel).
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/lance
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step5/bin/python
FRAMES=$SCR/se6/lance_frames
mkdir -p "$FRAMES"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scratch/step6/lance/dump_frames_step6.py --out "$FRAMES" --device cuda \
    --skip FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict \
    2>&1 || echo "DUMP rc=$?"
ls -la "$FRAMES"
echo "DUMP-GPU done"
