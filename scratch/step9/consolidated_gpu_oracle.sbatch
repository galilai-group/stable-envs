#!/bin/bash
#SBATCH --job-name=se9-wave1-consolidated
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# One post-Wave-1 GPU allocation, per the frozen Step-9 cadence ruling.  Each section runs against
# the SAME merged tree and the SAME card.  Their original #SBATCH lines are comments when invoked
# through bash; ``-e`` turns every non-zero oracle/test exit into a section failure.
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
  echo "baseline $(git rev-parse 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} | tee "$OUT/provenance_${SLURM_JOB_ID:-local}.txt"

run_section classic_nav scratch/step9/envs-classic-nav/gpu.sbatch
run_section dmcontrol scratch/step9/envs-dmcontrol/gpu.sbatch
run_section render_2d scratch/step9/render-2d/gpu_oracle.sbatch
run_section render_3d scratch/step9/render-3d/gpu_oracle.sbatch
run_section mjwarp_nvdiff scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch

echo "ALLDONE $(date -Is)"
if [ "$FAIL_COUNT" -eq 0 ]; then
  echo "CONSOLIDATED_GPU_ORACLE: PASS"
  exit 0
fi
echo "CONSOLIDATED_GPU_ORACLE: FAIL ($FAIL_COUNT sections)"
exit 1
