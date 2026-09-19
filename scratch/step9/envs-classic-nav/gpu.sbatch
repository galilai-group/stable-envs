#!/bin/bash
#SBATCH --job-name=se9-envs-classic-nav-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Workstream A's GPU arm, at the POST-MERGE head, closing the two gaps job 6464309 left:
#   1. its oracle compared the two trees on device="cpu", so the CUDA-only fused render paths
#      (render/gfxdraw_classic, render/navdot) were never executed, let alone compared. This runs the
#      comparison arm a second time with --device cuda, which is the only way those frames are
#      diffed baseline-vs-branch.
#   2. its tests/gpu section produced no pytest summary (an EGL teardown traceback is all that
#      reached the log), so the subset result was indeterminate. Every pytest run here writes its own
#      results file under $OUT and the summary line is echoed, so "COMPLETED" is not the evidence.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
WS=${SE9_WS:-$SCR/se9/classic-nav}
OUT=$WS/examples/results/step9/envs-classic-nav/gpu
PY=$SCR/venvs/se-step6/bin/python
SHA=$(cat $WS/.git_sha 2>/dev/null || echo unknown)
mkdir -p "$OUT"
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "sha $SHA";
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } | tee "$OUT/provenance_head.txt"

cd "$WS" || exit 1
export PYTHONPATH=$WS

echo "=== oracle, comparison arm on CPU (frames from the per-primitive/torch path)"
$PY scratch/step9/envs-classic-nav/oracle.py --python "$PY" --device cpu --head-sha "$SHA" \
    --name oracle_head_cpu.json --out "$OUT" 2>&1 | tail -20

echo "=== oracle, comparison arm on CUDA (frames from the FUSED paths)"
$PY scratch/step9/envs-classic-nav/oracle.py --python "$PY" --device cuda --head-sha "$SHA" \
    --name oracle_head_cuda.json --out "$OUT" 2>&1 | tail -20

# These two files ARE this workstream's families end to end, so they run whole.
for T in gfxdraw_classic navdot_fused; do
  F=$OUT/pytest_${T}.txt
  echo "=== tests/gpu/test_${T}.py (whole file)"
  $PY -m pytest -q -p no:cacheprovider "tests/gpu/test_${T}.py" > "$F" 2>&1
  echo "exit $? :: $(grep -E '[0-9]+ (passed|failed|error)' "$F" | tail -1)"
done

# These three cover every cell in the repo, so they are filtered to this workstream's. The filter
# spells both the CLASS names and the dataset ``name``s, because the three suites do not agree on
# which they parametrise with. Exit 5 (nothing selected) is reported, not swallowed: an empty
# selection looks exactly like a pass in a summary line, which is how a filter silently stops testing.
K="CartPole or Pendulum or Acrobot or MountainCar or MountainCarContinuous or TwoRoom or TwoRoomTarget"
K="$K or Piecewise or PiecewiseZones or SimplePointMaze"
K="$K or cartpole or pendulum or acrobot or mountain-car or mountain-car-continuous"
K="$K or swm-two-room or swm-two-room-target or swm-piecewise or swm-piecewise-zones or swm-simple-point-maze"
for T in conformance fast_paths reproducibility; do
  F=$OUT/pytest_${T}.txt
  echo "=== tests/gpu/test_${T}.py -k <this workstream's cells>"
  $PY -m pytest -q -p no:cacheprovider "tests/gpu/test_${T}.py" -k "$K" > "$F" 2>&1
  RC=$?
  echo "exit $RC :: $(grep -E '[0-9]+ (passed|failed|error|deselected)|no tests ran' "$F" | tail -1)"
  [ "$RC" = "5" ] && echo "WARNING: -k selected NOTHING in test_${T}.py"
done

echo "=== the new unit tests on the cuda host"
$PY -m pytest -q -p no:cacheprovider tests/unit/test_envs_classic_nav.py \
    > "$OUT/pytest_unit_classic_nav.txt" 2>&1
echo "exit $? :: $(tail -1 "$OUT/pytest_unit_classic_nav.txt")"

echo "ALLDONE9A-HEAD $(date -Is)"
