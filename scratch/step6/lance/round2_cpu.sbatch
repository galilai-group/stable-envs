#!/bin/bash
#SBATCH --job-name=se6-lance-r2cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --constraint=cascade
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Round 2, CPU side, ONE job so the 8-worker Lance arm and the 8-way renderer arm share a node, a
# core allocation and a load average:
#   (1) the three storage arms with CPU accounting (cores actually used) and read bandwidth, at
#       1 worker and 8 workers;
#   (2) our CPU renderer at 1 thread (the certification budget), at 8 threads, and as 8 single-thread
#       processes -- the last aggregated exactly the way bench_lance.time_parallel aggregates.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
WS=$SCR/se6/lance
export PYTHONPATH=$WS
export STABLE_ENVS_GIT_COMMIT=$(cat "$WS/.se6_commit" 2>/dev/null || echo unknown)
PY=$SCR/venvs/se6-verify/bin/python
FRAMES=$SCR/se6/lance_frames
OUT=$WS/examples/results/step6/lance
mkdir -p "$OUT"
LOCAL=/tmp/se_lance_$SLURM_JOB_ID
SHARED=$SCR/se6/lance_tmp_shared_$SLURM_JOB_ID
JOBTMP=/sss/jobtmp/$USER/$SLURM_JOB_ID/se_lance
mkdir -p "$LOCAL" "$SHARED" "$JOBTMP" 2>/dev/null

echo "=== node facts ==="; hostname; cat /proc/loadavg
lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz|L3|NUMA'
taskset -cp $$ || true
for P in "$LOCAL" "$SHARED" "$JOBTMP"; do echo "--- $P"; stat -f -c 'fstype=%T' "$P"; df -T "$P"|tail -1; done
scontrol show node $SLURMD_NODENAME | egrep 'NodeName|CPUTot|RealMemory|TmpDisk|AvailableFeatures' || true

CPUENVS=Acrobot,CartPole,CraftaxClassic,DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCCheetahRun,DMCFingerTurnHard,DMCHopperHop,DMCHumanoidStand,DMCPendulumSwingup,DMCReacherHard,DMCWalkerWalk,MountainCar,MountainCarContinuous,Pendulum,Piecewise,PiecewiseZones,PointmazeLargeNavigate,PushT,SimplePointMaze,TwoRoom,TwoRoomTarget

set -x
cd "$WS"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY scratch/step6/lance/lance_arms2.py \
  --frames-dir "$FRAMES" --chunk 256 --reps 8 --workers 1,8 \
  --arm "local=$LOCAL" --arm "shared=$SHARED" --arm "jobtmp=$JOBTMP" \
  --out "$OUT/lance_arms2_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "ARMS2 rc=$?"

$PY scratch/step6/lance/bench_cpu_parallel.py --envs "$CPUENVS" --batch 256 --reps 5 --workers 8 \
  --tree "$WS" --max-cell-s 90 \
  --out "$OUT/renderer_cpu_8way_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "CPU8WAY rc=$?"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY examples/bench_step3.py --device cpu --threads 1 \
  --envs "$CPUENVS" --reps 10 --max-cell-s 20 \
  --out "$OUT/renderer_cpu_1thread_r2_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "CPU1 rc=$?"
set +x
rm -rf "$LOCAL" "$SHARED" "$JOBTMP"
cat /proc/loadavg
echo "R2CPU done"
