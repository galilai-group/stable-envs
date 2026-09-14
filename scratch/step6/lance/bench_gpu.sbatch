#!/bin/bash
#SBATCH --job-name=se6-lance-benchgpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Phase 2b: the same benchmark on a 3090 node, adding the local_gpu (nvJPEG) arm -- the strongest
# thing the stored baseline could become -- on the SAME card class the Step-3/Step-6 GPU renderer
# rows were taken on. 8 CPUs (norm-gpu QOS allows 12) so the 8-worker arm again means 8 cores.
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
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | egrep 'Model name|Socket|Core|Thread|CPU\(s\)|MHz|L3|NUMA'
cat /proc/loadavg
for P in "$LOCAL" "$SHARED" "$JOBTMP"; do
  echo "--- $P"; stat -f -c 'fstype=%T blocks=%b avail=%a' "$P" 2>&1
  df -T "$P" 2>&1 | tail -1
  findmnt -n -o SOURCE,TARGET,FSTYPE,OPTIONS --target "$P" 2>&1 | head -1
done
egrep ' /tmp | /oscar | /sss' /proc/mounts
scontrol show node $SLURMD_NODENAME | egrep 'NodeName|CPUTot|RealMemory|TmpDisk|AvailableFeatures' || true

set -x
cd "$WS"
$PY scratch/step6/lance/run_lance_oscar.py \
  --frames-dir "$FRAMES" --chunk 256 --reps 8 --workers 8 --worker-sweep 1,4 --gpu \
  --arm "local=$LOCAL" --arm "shared=$SHARED" --arm "jobtmp=$JOBTMP" \
  --parallel-arms local,shared,jobtmp \
  --out "$OUT/lance_gpu_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "BENCH rc=$?"

# --- provenance check: are the dumped pictures the ones the Step-6 baseline draws? --------------
# Eight of the fixed-size configs reproduce Randall's committed jpeg_bytes_total to the byte, which
# says the frames are identical across two clusters. MountainCar/MountainCarContinuous do not
# (13,068 vs 12,882 B/frame). This re-dumps three of them from the PRISTINE Step-6 baseline copy
# ($SCR/se6/baseline, never modified) so the difference can be dated: same bytes as the branch means
# the drift is older than 89fa5ca, different bytes means Step 6 moved those pixels.
BASEFR=$SCR/se6/lance_frames_baseline
mkdir -p "$BASEFR"
( cd $SCR/se6/baseline && PYTHONPATH=$SCR/se6/baseline $SCR/venvs/se6-verify/bin/python \
    $WS/scratch/step6/lance/dump_frames_step6.py --out "$BASEFR" --device cuda \
    --envs Acrobot,MountainCar,MountainCarContinuous,Pendulum,CartPole 2>&1 ) || echo "BASEDUMP rc=$?"

set +x
rm -rf "$LOCAL" "$SHARED" "$JOBTMP"
cat /proc/loadavg
echo "BENCH-GPU done"
