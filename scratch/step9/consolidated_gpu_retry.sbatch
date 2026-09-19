#!/bin/bash
#SBATCH --job-name=se9-wave1-gpu-retry
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# Retry only the three sections whose first consolidated run exposed instrumentation/dependency
# problems. Classic/navigation and DMControl already passed on the merged Wave-1 tip.
set -uo pipefail

SCR=/oscar/scratch/sboughan
WS=${SE9_WS:?submit with SE9_WS pointing at the staged merged Wave-1 clone}
OUT=$WS/examples/results/step9/consolidated-gpu
mkdir -p "$OUT" "$SCR/logs"
cd "$WS" || exit 1
FAIL_COUNT=0

run_section () {
  local label="$1" script="$2"
  local log="$OUT/${label}_${SLURM_JOB_ID:-local}.log"
  echo "=== $label: $script ==="
  if SE9_WS="$WS" bash -e "$script" >"$log" 2>&1; then
    echo "$label: PASS"
  else
    local rc=$?
    echo "$label: FAIL rc=$rc (full log: $log)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

{
  echo "host $(hostname)"
  echo "job ${SLURM_JOB_ID:-none}"
  echo "date $(date -Is)"
  echo "head $(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$OUT/retry_provenance_${SLURM_JOB_ID:-local}.txt"

run_section render_2d scratch/step9/render-2d/gpu_oracle.sbatch
run_section render_3d scratch/step9/render-3d/gpu_oracle.sbatch
run_section mjwarp_nvdiff scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch

echo "ALLDONE $(date -Is)"
if [ "$FAIL_COUNT" -eq 0 ]; then
  echo "CONSOLIDATED_GPU_RETRY: PASS"
  exit 0
fi
echo "CONSOLIDATED_GPU_RETRY: FAIL ($FAIL_COUNT sections)"
exit 1
