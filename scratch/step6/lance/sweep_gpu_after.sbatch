#!/bin/bash
#SBATCH --job-name=se6-lance-gpusweep
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# 23 of the 36 registered_default GPU cells in the committed corpus are STEP-3 numbers: no Step-6
# shard re-measured them on a GPU (raycast3d's GPU sweep covers 2 envs, cpu-3d changed only the CPU
# backend, Fetch was never re-swept). Comparing a pre-Step-6 GPU row against a freshly measured
# storage baseline would quietly understate the renderer, so this takes the post-Step-6 GPU sweep
# for all 36 on one 3090, in the mandatory batch list, with the integration branch on PYTHONPATH.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se6/lance
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=$(cat "$WS/.se6_commit" 2>/dev/null || echo unknown)
OUT=$WS/examples/results/step6/lance
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NONFETCH=Acrobot,CartPole,CraftaxClassic,DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCCheetahRun,DMCFingerTurnHard,DMCHopperHop,DMCHumanoidStand,DMCPendulumSwingup,DMCReacherHard,DMCWalkerWalk,MountainCar,MountainCarContinuous,Pendulum,Piecewise,PiecewiseZones,PointmazeLargeNavigate,PushT,SimplePointMaze,TwoRoom,TwoRoomTarget
FETCH=FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict

$SCR/venvs/se6-verify/bin/python examples/bench_step3.py --device cuda --threads 4 \
  --envs "$NONFETCH" --reps 15 --max-cell-s 30 \
  --out "$OUT/renderer_gpu_after_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "GPU24 rc=$?"

$SCR/venvs/se-s3b-fetch311/bin/python examples/bench_step3.py --device cuda --threads 4 \
  --envs "$FETCH" --reps 15 --max-cell-s 30 \
  --out "$OUT/renderer_gpu_after_fetch311_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "GPUFETCH rc=$?"
set +x
echo "GPUSWEEP done"
