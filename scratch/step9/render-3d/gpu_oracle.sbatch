#!/bin/bash
#SBATCH --job-name=se9-render-3d-oracle
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# GPU arm of the Step-9 render-3d evidence. Three things, in one job on one card:
#
#   1. the SAME oracle the CPU arm ran (scratch/step9/render-3d/oracle.py), now with
#      --devices cpu,cuda, so the fused Triton cast and the tiled composite are compared
#      byte for byte against the Step-8 baseline tree as well as the torch fallback -- and so
#      the cells the Mac cannot build (the eleven OGBench mazes, the four maze2d cells, the
#      two OGB manipulation cells, which have no CPU backend at all) are finally covered;
#   2. paired ALTERNATING before/after timings on the GPU hot path (bench.py --set gpu),
#      both legs on this card in this job, which is the only way a before/after means anything;
#   3. the GPU test selection this workstream's code can affect.
#
# THE BASELINE COMES FROM THE TREE, NOT FROM A COPY. Both scripts do `git worktree add --detach`
# of 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc inside $WS, so $WS MUST be a real git clone whose
# history contains that commit (the coordinator makes it from a bundle) and $TMPDIR must be
# writable. If the clone is missing the SHA the scripts fail loudly rather than silently
# comparing the branch against itself.
#
# Submitted by the COORDINATOR, not by the workstream agent.
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
mkdir -p "$OUT" "$SCR/logs"
cd "$WS" || exit 1

{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "date $(date -Is)"
  echo "head $(git -C "$WS" rev-parse HEAD 2>/dev/null)"
  echo "baseline_present $(git -C "$WS" cat-file -e 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc^{commit} && echo yes || echo NO)"
  echo "python $PY"; $PY -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} 2>&1 | tee "$OUT/provenance_${TAG}.txt"

echo "=== 1/3 oracle (cpu + cuda, 64 frozen states per cell) ==="
PYTHONPATH="$WS" $PY scratch/step9/render-3d/oracle.py \
    --python "$PY" --devices cpu,cuda --n-states 64 --tag "$TAG" 2>&1 | tail -80
ORACLE_RC=${PIPESTATUS[0]}

echo "=== 2/3 paired alternating GPU timings (same card, this job) ==="
PYTHONPATH="$WS" $PY scratch/step9/render-3d/bench.py \
    --python "$PY" --device cuda --set gpu --rounds 5 --reps 5 --tag "$TAG" 2>&1 | tail -40

echo "=== 3/3 GPU tests that can see this workstream's code ==="
# The WHOLE output goes to a file, and only the summary is echoed. Job 6464319 piped this straight
# into `tail -40` and MuJoCo's EGL teardown -- dozens of "Exception ignored in __del__" lines that
# arrive AFTER pytest has printed its verdict -- pushed the summary out of the window, so the run
# produced no test counts at all. Interpreter teardown noise must never be able to eat the result.
PYTHONPATH="$WS" $PY -m pytest -q -p no:cacheprovider tests/gpu \
    -k "raycast or cpu3d or DMC or Pointmaze or Fetch" > "$OUT/gpu_tests_${TAG}.log" 2>&1
TEST_RC=$?
grep -E "passed|failed|error" "$OUT/gpu_tests_${TAG}.log" | grep -v "Exception ignored" | tail -5
echo "(full pytest output: $OUT/gpu_tests_${TAG}.log)"

echo "ORACLE_RC=$ORACLE_RC TEST_RC=$TEST_RC"
echo "artifacts in $OUT (oracle_${TAG}.json, timings_${TAG}.json, provenance_${TAG}.txt)"
echo "ALLDONE $(date -Is)"
