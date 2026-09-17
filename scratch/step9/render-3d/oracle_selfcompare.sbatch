#!/bin/bash
#SBATCH --job-name=se9-render-3d-selfcompare
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu1503
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# THE DECISIVE RUN for the one open item, and it deliberately changes only one thing at a time.
#
# Job 6464319 (gpu1503, RTX A6000) ran the oracle's 38-cell process once per tree and found
# HumanoidmazeGiantNavigate/cuda differing: baseline 929f7c09..., head 6c5385e4....
# Job 6464500 (gpu1504, NVIDIA A40) ran the follow-up probe, which renders that ONE cell in a fresh
# process, three times per tree: every one of the six runs gave 6c5385e4..., 0 differing pixels --
# i.e. the BASELINE tree has now produced BOTH digests, and the head tree only ever one.
#
# Two things differed between those jobs at once (the card, and whether the process had rendered 37
# other cells first), so neither is yet isolated. This job pins the card to gpu1503 -- the one that
# showed the difference -- and runs the oracle in its ORIGINAL 38-cell process shape with both arms
# on the SAME tree:
#
#   leg 1: --arms baseline,baseline   leg 2: --arms head,head   leg 3: --arms baseline,head
#
# If leg 1 reports HumanoidmazeGiantNavigate/cuda as a difference, the cell is not reproducible from
# one process to the next in that shape and job 6464319's finding is a property of the cell, not of
# the patch. If leg 1 and leg 2 are clean and leg 3 differs, the patch changed the cell.
#
# Submitted by the COORDINATOR. Read the three "identical" lines.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se9/render-3d
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step9/render-3d
TAG=gpu_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
cd "$WS" || exit 1
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "head $(git -C "$WS" rev-parse HEAD)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } | tee "$OUT/provenance_self_${TAG}.txt"
for ARMS in baseline,baseline head,head baseline,head; do
  echo "=== arms $ARMS (cuda only, 64 frozen states, the full 38-cell process shape) ==="
  PYTHONPATH="$WS" $PY scratch/step9/render-3d/oracle.py --python "$PY" --devices cuda \
      --n-states 64 --arms "$ARMS" --tag "${TAG}_${ARMS/,/_}" 2>&1 | grep -vE "^ \"(DMC|Mujoco|Point|Ant|Humanoid|OGB)" | tail -30
done
echo "ALLDONE $(date -Is)"
