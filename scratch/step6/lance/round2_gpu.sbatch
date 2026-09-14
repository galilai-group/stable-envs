#!/bin/bash
#SBATCH --job-name=se6-lance-r2gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Round 2, GPU side, ONE job so the Lance GPU arms, the end-to-end JPEG->device path and the
# end-to-end state->device path are all on the same card under the same load:
#   (1) the three storage arms + the nvJPEG arm, with CPU accounting;
#   (2) stored JPEG -> frames ON THE DEVICE: a real 8-worker DataLoader with pin_memory plus the H2D
#       copy, timed together, and the nvJPEG variant, and the H2D term alone;
#   (3) our renderer end-to-end through examples/bench_step3_pipeline.py (stored states -> sampler ->
#       H2D -> render -> frames on device).
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
SHARED=$SCR/se6/lance_tmp_shared_$SLURM_JOB_ID
JOBTMP=/sss/jobtmp/$USER/$SLURM_JOB_ID/se_lance
mkdir -p "$LOCAL" "$SHARED" "$JOBTMP" 2>/dev/null

echo "=== node facts ==="; hostname; cat /proc/loadavg
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz|L3|NUMA'
for P in "$LOCAL" "$SHARED" "$JOBTMP"; do echo "--- $P"; stat -f -c 'fstype=%T' "$P"; df -T "$P"|tail -1; done

NONFETCH=Acrobot,CartPole,CraftaxClassic,DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCCheetahRun,DMCFingerTurnHard,DMCHopperHop,DMCHumanoidStand,DMCPendulumSwingup,DMCReacherHard,DMCWalkerWalk,MountainCar,MountainCarContinuous,Pendulum,Piecewise,PiecewiseZones,PointmazeLargeNavigate,PushT,SimplePointMaze,TwoRoom,TwoRoomTarget
FETCH=FetchPickAndPlace,FetchPickAndPlaceDense,FetchPickAndPlaceDict,FetchPush,FetchPushDense,FetchPushDict,FetchReach,FetchReachDense,FetchReachDict,FetchSlide,FetchSlideDense,FetchSlideDict

set -x
cd "$WS"
$PY scratch/step6/lance/bench_jpeg_to_gpu.py --frames-dir "$FRAMES" --local "$LOCAL" \
  --chunk 256 --reps 5 --workers 8 \
  --out "$OUT/jpeg_to_gpu_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "J2G rc=$?"

$PY scratch/step6/lance/lance_arms2.py --frames-dir "$FRAMES" --chunk 256 --reps 8 --workers 1,8 --gpu \
  --arm "local=$LOCAL" --arm "shared=$SHARED" --arm "jobtmp=$JOBTMP" \
  --out "$OUT/lance_arms2_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "ARMS2 rc=$?"

$PY examples/bench_step3_pipeline.py --device cuda --threads 4 --envs "$NONFETCH" \
  --batches 256,1024,4096 --n-states 8192 --reps 3 \
  --out "$OUT/pipeline_gpu_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "PIPE rc=$?"

$PYF examples/bench_step3_pipeline.py --device cuda --threads 4 --envs "$FETCH" \
  --batches 256,1024,4096 --n-states 8192 --reps 3 \
  --out "$OUT/pipeline_gpu_fetch311_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "PIPEF rc=$?"
set +x
rm -rf "$LOCAL" "$SHARED" "$JOBTMP"
cat /proc/loadavg
echo "R2GPU done"
