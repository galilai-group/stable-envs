#!/bin/bash
#SBATCH --job-name=se6-lance-dumpfetch
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The 12 Fetch configs need the Step-3B 3.11 pin (se-s3b-fetch311). That venv has no lancedb, but it
# does have PIL, and encoding is all Phase 1 needs -- the Lance table is built later by the bench.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/lance
export PYTHONPATH=$WS
PY=$SCR/venvs/se-s3b-fetch311/bin/python
FRAMES=$SCR/se6/lance_frames
mkdir -p "$FRAMES"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scratch/step6/lance/dump_frames_step6.py --out "$FRAMES" --device cuda \
    --envs FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict \
    2>&1 || echo "DUMPFETCH rc=$?"
ls -la "$FRAMES"
echo "DUMP-FETCH done"
