#!/bin/bash
#SBATCH --job-name=se6-lance-r4gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The two halves of gpu_pipeline_ratio must come off the SAME card in the SAME job, or the ratio is
# two machines divided by each other. So this job holds both:
#   numerator   examples/bench_step3_pipeline.py -- stored states -> sampler -> H2D -> render ->
#               frames on the device;
#   denominator scratch/step6/lance/bench_jpeg_to_gpu.py -- stored JPEG -> 8-worker DataLoader with
#               pin_memory -> H2D -> frames on the device, plus the nvJPEG variant and the H2D term
#               alone.
# The DataLoader here is given bench_lance's own batch schedule (8 x 8 chunks of 256 over the same
# table), because a 512-row table holds two 256-row batches and a plain epoch would leave six of the
# eight workers idle.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se6/lance
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=$(cat "$WS/.se6_commit" 2>/dev/null || echo unknown)
PY=$SCR/venvs/se6-verify/bin/python
PYF=$SCR/venvs/se-s3b-fetch311/bin/python
FRAMES=$SCR/se6/lance_frames
OUT=$WS/examples/results/step6/lance
mkdir -p "$OUT"
LOCAL=/tmp/se_lance_$SLURM_JOB_ID
mkdir -p "$LOCAL"
echo "=== node facts ==="; hostname; cat /proc/loadavg
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | egrep 'Model name|Core|Thread|CPU\(s\)|NUMA'
stat -f -c 'fstype=%T' "$LOCAL"

NONFETCH=Acrobot,CartPole,CraftaxClassic,DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCCheetahRun,DMCFingerTurnHard,DMCHopperHop,DMCHumanoidStand,DMCPendulumSwingup,DMCReacherHard,DMCWalkerWalk,MountainCar,MountainCarContinuous,Pendulum,Piecewise,PiecewiseZones,PointmazeLargeNavigate,PushT,SimplePointMaze,TwoRoom,TwoRoomTarget
FETCH=FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict

set -x
cd "$WS"
$PY scratch/step6/lance/bench_jpeg_to_gpu.py --frames-dir "$FRAMES" --local "$LOCAL" \
  --chunk 256 --reps 4 --workers 8 \
  --out "$OUT/jpeg_to_gpu_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "J2G rc=$?"

$PY examples/bench_step3_pipeline.py --device cuda --threads 4 --envs "$NONFETCH" \
  --batches 256,1024,4096 --n-states 8192 --reps 3 \
  --out "$OUT/pipeline_gpu_r4_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "PIPE rc=$?"

$PYF examples/bench_step3_pipeline.py --device cuda --threads 4 --envs "$FETCH" \
  --batches 256,1024,4096 --n-states 8192 --reps 3 \
  --out "$OUT/pipeline_gpu_r4_fetch311_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "PIPEF rc=$?"
set +x
rm -rf "$LOCAL"
cat /proc/loadavg
echo "R4GPU done"
