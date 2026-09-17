#!/bin/bash
#SBATCH --job-name=se9-render-3d-followup
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# Follow-up to job 6464319, which found ONE difference in 59 cell/device pairs:
# HumanoidmazeGiantNavigate on cuda (its cpu arm and every other maze matched). The patch contains
# no arithmetic change, so this asks the prior question first: does that cell give the same frames
# twice from the SAME tree on this card? Three fresh processes per tree, both trees, plus the
# torch-backend and unbounded expressions and a pixel-level diff.
#
# It also RE-RUNS the GPU test selection, because job 6464319 produced no test counts: that section
# was piped into `tail -40` and MuJoCo's EGL teardown noise (dozens of "Exception ignored in
# __del__" lines printed after pytest's verdict) pushed the summary out of the window. Its TEST_RC=1
# is therefore uninterpretable -- it is equally consistent with real failures and with a nonzero
# exit from teardown. Here the whole output goes to a file and only the summary is echoed.
#
# Submitted by the COORDINATOR. Read the VERDICT line and the pytest summary at the end of the log.
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
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } | tee "$OUT/provenance_hmg_${TAG}.txt"
echo "=== 1/2 HumanoidmazeGiantNavigate reproducibility, both trees, three processes each ==="
PYTHONPATH="$WS" $PY scratch/step9/render-3d/hmg_determinism.py \
    --python "$PY" --devices cuda,cpu --repeats 3 --tag "$TAG" 2>&1 | tail -60

echo "=== 2/2 the GPU test selection, with its summary preserved this time ==="
PYTHONPATH="$WS" $PY -m pytest -q -p no:cacheprovider tests/gpu \
    -k "raycast or cpu3d or DMC or Pointmaze or Fetch" > "$OUT/gpu_tests_${TAG}.log" 2>&1
TEST_RC=$?
grep -E "passed|failed|error" "$OUT/gpu_tests_${TAG}.log" | grep -v "Exception ignored" | tail -5
echo "TEST_RC=$TEST_RC (full output: $OUT/gpu_tests_${TAG}.log)"
echo "ALLDONE $(date -Is)"
