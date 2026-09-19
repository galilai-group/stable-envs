#!/bin/bash
#SBATCH --job-name=se9-envs-dmcontrol-gpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
# Workstream B's GPU arm. Three things only a CUDA host can settle:
#
#   1. the oracle with its comparison arm on --device cuda. On CPU the twelve parity cells run
#      raycast3d's CPU path and the trackcom cells run cpu3d; on CUDA the parity cells run the fused
#      mover pass and the trackcom cells run MJWARP, which is the backend every visual_fidelity
#      number in this family was measured with and which a cpu run never executes a line of. It also
#      gets DMCSwimmerSwimmer6 rendered at all -- it refuses device="cpu" by declaration, so the
#      local arm has 17 of 18 cells' frames and this one has 18.
#   2. tests/gpu -k "DMC", which is CUDA-gated in its entirety.
#   3. paired GPU timings for two parity cells and one trackcom cell.
#
# Every pytest run writes its own results file under $OUT and the summary line is echoed, so
# "COMPLETED" is not the evidence -- the file is. Do NOT submit this from an agent; the coordinator
# submits and returns the job id.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export MUJOCO_GL=egl
WS=${SE9_WS:-$SCR/se9/envs-dmcontrol}
OUT=$WS/examples/results/step9/envs-dmcontrol/gpu
# The mjwarp arm at the DECLARED 3.11 pair. This family's certification semantics are mujoco 3.11.0 +
# mujoco-warp 3.11.0 (reference_versions), and se-vis311 is the only venv built on that pair -- the
# default se-step6 is mujoco 3.12.0, under which Step 4 recorded three renderer FAILs for a version
# drift (examples/results/step6/texture-fidelity/findings.md). Measuring this family anywhere else
# would be measuring a different reference.
PY=$SCR/venvs/se-vis311/bin/python
SHA=$(cat $WS/.git_sha 2>/dev/null || git -C "$WS" rev-parse HEAD 2>/dev/null || echo unknown)
mkdir -p "$OUT"
{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "sha $SHA"; echo "python $PY";
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader;
  $PY -c "import mujoco, torch; print('mujoco', mujoco.__version__, 'torch', torch.__version__)";
} | tee "$OUT/provenance_head.txt"

cd "$WS" || exit 1
export PYTHONPATH=$WS

# The trackcom tier IS mujoco_warp on CUDA, so a missing import would silently turn six cells into
# skips. Checked loudly, and the two pure-python deps the Step-9 foundation added are installed into
# that venv if they are absent (uv, under $SCR caches -- HOME is at quota).
$PY -c "import mujoco_warp" 2>/dev/null \
  || { echo "FATAL: mujoco_warp missing from $PY; this job cannot measure the visual tier"; exit 2; }
for M in tensordict loguru; do
  $PY -c "import $M" 2>/dev/null || ~/.local/bin/uv pip install --python "$PY" "$M" \
    || { echo "FATAL: could not install $M into $PY"; exit 2; }
done
$PY -c "import tensordict, loguru, mujoco_warp; print('deps ok')" | tee -a "$OUT/provenance_head.txt"

echo "=== oracle, comparison arm on CPU (raycast3d CPU + cpu3d; 17 of 18 cells render)"
$PY scratch/step9/envs-dmcontrol/oracle.py --python "$PY" --device cpu --head-sha "$SHA" \
    --name oracle_head_cpu.json --out "$OUT" 2>&1 | tail -30

echo "=== oracle, comparison arm on CUDA (the fused mover pass + MJWARP; all 18 cells render)"
$PY scratch/step9/envs-dmcontrol/oracle.py --python "$PY" --device cuda --head-sha "$SHA" \
    --name oracle_head_cuda.json --out "$OUT" 2>&1 | tail -30

# The whole DM Control selection of every GPU suite. "DMC" matches the class names; the dataset
# ``name``s are spelled too because the suites do not agree on which they parametrise with. Exit 5
# (nothing selected) is REPORTED, not swallowed: an empty selection looks exactly like a pass in a
# summary line, which is how a filter silently stops testing.
K="DMC or dmc-"
for T in conformance fast_paths reproducibility; do
  F=$OUT/pytest_${T}.txt
  echo "=== tests/gpu/test_${T}.py -k \"$K\""
  $PY -m pytest -q -p no:cacheprovider "tests/gpu/test_${T}.py" -k "$K" > "$F" 2>&1
  RC=$?
  echo "exit $RC :: $(grep -E '[0-9]+ (passed|failed|error|deselected)|no tests ran' "$F" | tail -1)"
  [ "$RC" = "5" ] && echo "WARNING: -k selected NOTHING in test_${T}.py"
done

echo "=== the new unit tests on the cuda host"
$PY -m pytest -q -p no:cacheprovider tests/unit/test_envs_dmcontrol.py \
    tests/unit/test_dmcontrol_tendon.py tests/unit/test_dmcontrol_episode_quat.py \
    tests/unit/test_cpu3d.py tests/certification/test_declarations.py \
    > "$OUT/pytest_unit_dmcontrol.txt" 2>&1
echo "exit $? :: $(tail -1 "$OUT/pytest_unit_dmcontrol.txt")"

# Paired GPU timings, ALTERNATING between the two trees in one job on one card (the same-card rule).
# Two parity cells and one trackcom cell: the parity pair is the fused mover pass, the trackcom cell
# is MJWarp, and they are different programs.
echo "=== paired GPU render timings (2 parity + 1 trackcom)"
git -C "$WS" worktree add --detach /tmp/se9b-baseline 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc \
  2>/dev/null || git --git-dir="$WS/.git" worktree add --detach /tmp/se9b-baseline \
  21cb30678b7a7d0f3344b59f176f33d3f3b09ffc 2>/dev/null
if [ -d /tmp/se9b-baseline/stable_envs ]; then
  $PY scratch/step9/envs-dmcontrol/perf.py --python "$PY" --device cuda \
      --baseline /tmp/se9b-baseline --repeats 5 --iters 9 \
      --cells DMCCartpoleBalance,DMCReacherHard,DMCCheetahRun \
      --out "$OUT/performance_cuda.json" 2>&1 | tail -40
  git -C "$WS" worktree remove --force /tmp/se9b-baseline 2>/dev/null
else
  echo "WARNING: no baseline worktree at /tmp/se9b-baseline; the GPU timings are UNPAIRED and were skipped"
fi

echo "ALLDONE9B-HEAD $(date -Is)"
