#!/bin/bash
#SBATCH --job-name=se6-lance-dumprest
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The remainder of Phase 1, in one GPU job because norm-gpu allows one at a time per agent:
#   (a) the four configs se-step5 could not reach -- MountainCar/MountainCarContinuous/Pendulum need
#       pygame, PointmazeLargeNavigate needs ogbench; se6-verify carries both (and mujoco_warp,
#       dm_control, lancedb), so it is the one interpreter that covers the whole 3.12 arm.
#   (b) the 12 Fetch configs, which are pinned to the Step-3B mujoco 3.11 stack.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se6/lance
export PYTHONPATH=$WS
FRAMES=$SCR/se6/lance_frames
mkdir -p "$FRAMES"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$SCR/venvs/se6-verify/bin/python scratch/step6/lance/dump_frames_step6.py --out "$FRAMES" --device cuda \
    --envs MountainCar,MountainCarContinuous,Pendulum,PointmazeLargeNavigate 2>&1 || echo "DUMP4 rc=$?"

$SCR/venvs/se-s3b-fetch311/bin/python scratch/step6/lance/dump_frames_step6.py --out "$FRAMES" --device cuda \
    --envs FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict \
    2>&1 || echo "DUMPFETCH rc=$?"

ls -la "$FRAMES" | tail -45
ls "$FRAMES"/*.npz | wc -l
echo "DUMP-REST done"
