#!/bin/bash
#SBATCH --job-name=se6-lance-benchcpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --constraint=cascade
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Phase 2a: the 8-worker JPEG-in-Lance baseline on an Oscar BATCH node. 8 cores for 8 workers, so
# "8 workers" really means 8 cores and the aggregate is not thread-thrashing.
# Three storage arms, all named by the filesystem they actually land on, checked from inside the job:
#   local   -> node-local /tmp        (SLURM reports TmpDisk=0 on Oscar; the probe records what it is)
#   shared  -> /oscar/scratch         (NFS v3 over RDMA, hpcnfs:/oscar, nconnect=8)
#   jobtmp  -> /sss/jobtmp/$USER/$SLURM_JOB_ID  (job-local GPFS)
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/lance
export PYTHONPATH=$WS
PY=$SCR/venvs/se6-verify/bin/python
FRAMES=$SCR/se6/lance_frames
OUT=$WS/examples/results/step6/lance
mkdir -p "$OUT"

LOCAL=/tmp/se_lance_$SLURM_JOB_ID
SHARED=$SCR/se6/lance_tmp_shared_$SLURM_JOB_ID
JOBTMP=/sss/jobtmp/$USER/$SLURM_JOB_ID/se_lance
mkdir -p "$LOCAL" "$SHARED" "$JOBTMP" 2>/dev/null

echo "=== node facts ==="
hostname; lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz|L3|NUMA'
cat /proc/loadavg
echo "--- affinity: $(taskset -cp $$ 2>/dev/null)"
for P in "$LOCAL" "$SHARED" "$JOBTMP"; do
  echo "--- $P"; stat -f -c 'fstype=%T blocks=%b avail=%a' "$P" 2>&1
  df -T "$P" 2>&1 | tail -1
  findmnt -n -o SOURCE,TARGET,FSTYPE,OPTIONS --target "$P" 2>&1 | head -1
done
echo "=== /proc/mounts (relevant) ==="
egrep ' /tmp | /oscar | /sss' /proc/mounts
echo "=== SLURM TmpDisk ==="
scontrol show node $SLURMD_NODENAME | egrep 'NodeName|CPUTot|RealMemory|TmpDisk|AvailableFeatures' || true

set -x
cd "$WS"
$PY scratch/step6/lance/run_lance_oscar.py \
  --frames-dir "$FRAMES" --chunk 256 --reps 8 --workers 8 --worker-sweep 1,4 \
  --arm "local=$LOCAL" --arm "shared=$SHARED" --arm "jobtmp=$JOBTMP" \
  --parallel-arms local,shared,jobtmp \
  --out "$OUT/lance_cpu_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "BENCH rc=$?"

# --- supplemental: OUR CPU renderer on the SAME node, same job, same load ----------------------
# Step 2 s5.3 fixes the 1-thread budget as the certification number and the all-core figure as a
# scaling result reported beside it, never substituted for it. Both are taken here so that
# "our CPU fps" and "8-worker Lance fps" are numbers off one node under one load, which is the only
# way the per-core comparison means anything. Fetch is absent: mjwarp-only, no CPU backend.
CPUENVS=Acrobot,CartPole,CraftaxClassic,DMCAcrobotSwingup,DMCBallInCupCatch,DMCCartpoleBalance,DMCCartpoleSwingup,DMCCheetahRun,DMCFingerTurnHard,DMCHopperHop,DMCHumanoidStand,DMCPendulumSwingup,DMCReacherHard,DMCWalkerWalk,MountainCar,MountainCarContinuous,Pendulum,Piecewise,PiecewiseZones,PointmazeLargeNavigate,PushT,SimplePointMaze,TwoRoom,TwoRoomTarget
export STABLE_ENVS_GIT_COMMIT=$(cat "$WS/.se6_commit" 2>/dev/null || echo unknown)

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY examples/bench_step3.py --device cpu --threads 1 \
  --envs "$CPUENVS" --reps 10 --max-cell-s 20 \
  --out "$OUT/renderer_cpu_1thread_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "CPU1 rc=$?"

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY examples/bench_step3.py --device cpu --threads 8 \
  --envs "$CPUENVS" --reps 10 --max-cell-s 20 \
  --out "$OUT/renderer_cpu_8thread_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "CPU8 rc=$?"

set +x
rm -rf "$LOCAL" "$SHARED" "$JOBTMP"
cat /proc/loadavg
echo "BENCH-CPU done"
