#!/bin/bash
#SBATCH --job-name=se9-wave1-gpu-mjwarp
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# MJWarp/nvdiff section only. 2D pixels/timings already passed on 6507919; 3D pixels/timings
# already passed there too. This job is the remaining I gate.
set -uo pipefail

SCR=/oscar/scratch/sboughan
WS=${SE9_WS:?submit with SE9_WS pointing at the staged merged Wave-1 clone}
OUT=$WS/examples/results/step9/consolidated-gpu
mkdir -p "$OUT" "$SCR/logs"
cd "$WS" || exit 1

{
  echo "host $(hostname)"
  echo "job ${SLURM_JOB_ID:-none}"
  echo "date $(date -Is)"
  echo "head $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$OUT/mjwarp_only_provenance_${SLURM_JOB_ID:-local}.txt"

log="$OUT/mjwarp_nvdiff_${SLURM_JOB_ID:-local}.log"
echo "=== mjwarp_nvdiff: scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch ==="
if SE9_WS="$WS" bash scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch >"$log" 2>&1; then
  echo "mjwarp_nvdiff: PASS"
  echo "ALLDONE $(date -Is)"
  echo "CONSOLIDATED_GPU_MJWARP: PASS"
  exit 0
fi
rc=$?
echo "mjwarp_nvdiff: FAIL rc=$rc (full log: $log)"
echo "ALLDONE $(date -Is)"
echo "CONSOLIDATED_GPU_MJWARP: FAIL"
exit 1
