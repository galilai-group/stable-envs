#!/bin/bash
#SBATCH --job-name=se9-render-2d-oracle
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# GPU arm of the Step-9 render-2d evidence. Four things, in one job on one card:
#
#   1. the SAME oracle the CPU arm ran (scratch/step9/render-2d/oracle.py), now with
#      --devices cpu,cuda, so the fused kernels this workstream touched -- raster2d's binned
#      scene kernel, gfxdraw_classic's classic-control kernel, gfxdraw_fused's CarRacing
#      surface kernel, tilemap's blitter and navdot's two passes -- are compared byte for byte
#      against the Step-8 baseline TREE, together with the torch expression behind each of them
#      (the oracle's `variants`). It also covers the cells the Mac cannot build at all:
#      CarRacing, LunarLander, the six MiniGrid configs, BabyAI, Sokoban, Overcooked, Griddly,
#      PGX and MiniHack, whichever of them this venv can import;
#   2. the same oracle for the two Craftax cells in the craftax venv, which is the only
#      interpreter where that family imports;
#   3. paired ALTERNATING before/after timings on the GPU hot paths (bench.py --set gpu), both
#      legs on this card in this job, which is the only way a before/after means anything;
#   4. the GPU test selection this workstream's code can affect.
#
# THE BASELINE COMES FROM THE TREE, NOT FROM A COPY. Both scripts do `git worktree add --detach`
# of 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc inside $WS, so $WS MUST be a real git clone whose
# history contains that commit and $TMPDIR must be writable. If the clone is missing the SHA the
# scripts fail loudly rather than silently comparing the branch against itself.
#
# Submitted by the COORDINATOR, not by the workstream agent.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WS=$SCR/se9/render-2d
PY=$SCR/venvs/se-step6/bin/python
PY_CRAFTAX=$SCR/venvs/se7-craftax-gpu/bin/python
OUT=$WS/examples/results/step9/render-2d
TAG=gpu_${SLURM_JOB_ID:-local}
mkdir -p "$OUT" "$SCR/logs"
cd "$WS" || exit 1

{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "date $(date -Is)"
  echo "head $(git -C "$WS" rev-parse HEAD 2>/dev/null)"
  echo "baseline_present $(git -C "$WS" cat-file -e 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc^{commit} && echo yes || echo NO)"
  echo "python $PY"; $PY -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"
  echo "python_craftax $PY_CRAFTAX"
  $PY_CRAFTAX -c "import torch; print('torch', torch.__version__)" 2>&1 | tail -1
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} 2>&1 | tee "$OUT/provenance_${TAG}.txt"

echo "=== 1/4 oracle (cpu + cuda, 64 frozen states per cell, plus the non-fused variants) ==="
PYTHONPATH="$WS" $PY scratch/step9/render-2d/oracle.py \
    --python "$PY" --devices cpu,cuda --n-states 64 --tag "$TAG" 2>&1 | tail -120
ORACLE_RC=${PIPESTATUS[0]}

echo "=== 2/4 oracle, Craftax cells, craftax venv ==="
if [ -x "$PY_CRAFTAX" ]; then
    PYTHONPATH="$WS" $PY_CRAFTAX scratch/step9/render-2d/oracle.py \
        --python "$PY_CRAFTAX" --devices cpu,cuda --n-states 64 \
        --only CraftaxClassic,CraftaxFull --tag "${TAG}_craftax" 2>&1 | tail -40
    CRAFTAX_RC=${PIPESTATUS[0]}
else
    echo "SKIPPED: $PY_CRAFTAX is not executable"; CRAFTAX_RC=99
fi

echo "=== 3/4 paired alternating GPU timings (same card, this job) ==="
PYTHONPATH="$WS" $PY scratch/step9/render-2d/bench.py \
    --python "$PY" --device cuda --set gpu --rounds 5 --reps 5 --tag "$TAG" 2>&1 | tail -40
# The same-tree floor, so a ratio near the bar can be read. Cheap: two cells, three rounds.
PYTHONPATH="$WS" $PY scratch/step9/render-2d/bench.py \
    --python "$PY" --device cuda --set gpu --rounds 3 --reps 5 --arms head,head \
    --tag "${TAG}_selftime" 2>&1 | tail -20

echo "=== 4/4 GPU tests that can see this workstream's code ==="
# The WHOLE output goes to a file and only the summary is echoed. Workstream H's job 6464319 piped
# pytest into `tail -40` and MuJoCo's EGL teardown -- dozens of "Exception ignored in __del__" lines
# printed AFTER pytest's verdict -- pushed the summary out of the window, so the run produced no
# test counts at all. Interpreter teardown noise must never be able to eat the result.
SEL="PushT or SimplePointMaze or TwoRoom or Piecewise or CartPole or Acrobot or MountainCar or Pendulum or CarRacing or LunarLander or MiniGrid or Crafter or Craftax or Sokoban or Overcooked or Highway or Roundabout or Racetrack or Intersection or Merge or Parking or gfxdraw or raster or tilemap or navdot or prng"
PYTHONPATH="$WS" $PY -m pytest -q -p no:cacheprovider \
    tests/gpu/test_gfxdraw_classic.py tests/gpu/test_gfxdraw_fused.py \
    tests/gpu/test_navdot_fused.py tests/gpu/test_fast_paths.py tests/gpu/test_conformance.py \
    -k "$SEL" > "$OUT/gpu_tests_${TAG}.log" 2>&1
TEST_RC=$?
grep -E "passed|failed|error" "$OUT/gpu_tests_${TAG}.log" | grep -v "Exception ignored" | tail -5
echo "(full pytest output: $OUT/gpu_tests_${TAG}.log)"

echo "ORACLE_RC=$ORACLE_RC CRAFTAX_RC=$CRAFTAX_RC TEST_RC=$TEST_RC"
echo "artifacts in $OUT (oracle_${TAG}.json, oracle_${TAG}_craftax.json, timings_${TAG}.json,"
echo "                   timings_${TAG}_selftime.json, provenance_${TAG}.txt, gpu_tests_${TAG}.log)"
echo "ALLDONE $(date -Is)"
