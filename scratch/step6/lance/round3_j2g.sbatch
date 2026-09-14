#!/bin/bash
#SBATCH --job-name=se6-lance-j2g
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Re-run of the stored-JPEG -> frames-on-device arm after the DataLoader fetcher fix: with
# batch_size=None torch disables auto-collation and hands the whole index list to __getitem__,
# not to __getitems__, so the first attempt (job 6367697) raised on the Lance take.
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
mkdir -p "$LOCAL"
echo "=== node facts ==="; hostname; cat /proc/loadavg
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | egrep 'Model name|Core|Thread|CPU\(s\)'
stat -f -c 'fstype=%T' "$LOCAL"
set -x
cd "$WS"
$PY scratch/step6/lance/bench_jpeg_to_gpu.py --frames-dir "$FRAMES" --local "$LOCAL" \
  --chunk 256 --reps 5 --workers 8 \
  --out "$OUT/jpeg_to_gpu_${SLURMD_NODENAME}_${SLURM_JOB_ID}.json" 2>&1 || echo "J2G rc=$?"
set +x
rm -rf "$LOCAL"
cat /proc/loadavg
echo "J2G done"
