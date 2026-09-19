#!/bin/bash
#SBATCH --job-name=se9-render-mjwarp-nvdiff-oracle
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/sboughan/logs/%x_%j.out
#
# GPU arm of Step-9 Workstream I (render/mjwarp.py, mjwarp_lightspec.py, nvdiff.py, rendernet.py,
# _resources.py). NOT SUBMITTED BY THE WORKSTREAM AGENT: per Sami's cadence ruling the coordinator
# runs one GPU oracle after the Wave-1 merges, and this file is the section of that job covering
# these four backends. Nothing in it has been executed; every number it produces is owed.
#
# WHY IT NEEDS FOUR VENVS AND ONE CARD. These backends span four incompatible dependency pins --
# mujoco_warp 3.11 (DM Control trackcom, RoboCasa), the ogbench stack, the nvdiffrast/PyFlyt stack,
# and the Fetch 3.11 pin -- and no interpreter holds all of them. But a paired before/after is only
# a before/after ON THE SAME CARD IN THE SAME JOB, so all four arms run here, sequentially, on the
# one GPU this job holds. Each arm writes its own record tagged with its venv.
#
# WHAT IT MEASURES, per arm:
#   1. frame identity, baseline tree vs this branch, over the frozen corpora  (oracle.py, default arms)
#   2. fresh-process reproducibility: the SAME tree twice                     (oracle.py --arms head,head)
#   3. repeated create/close cycles, with BOTH counters asserted              (inside oracle.py)
#        - stable_envs.render._resources: created == closed, nothing left live
#        - stable_envs.testing.resource_guard: EGL contexts released == 0, renderers closed == 0
#          (closing an env must NEVER release a GL context; see render/_resources.py for why)
#   4. batch-size invariance at 1/2/8/64                                      (inside oracle.py)
#   5. same-card paired throughput at B=1 and B=4096 (stepping down on OOM)   (bench.py)
#   6. the GPU tests that can see this code
#
# THE BASELINE COMES FROM THE TREE. oracle.py and bench.py both `git worktree add --detach` the
# Step-8 SHA inside $WS, so $WS must be a REAL CLONE whose history contains
# 21cb30678b7a7d0f3344b59f176f33d3f3b09ffc, and $TMPDIR must be writable. A clone missing the SHA
# fails loudly rather than silently comparing the branch with itself.
set -uo pipefail
SCR=/oscar/scratch/sboughan
export UV_CACHE_DIR=$SCR/cache/uv PIP_CACHE_DIR=$SCR/cache/pip XDG_CACHE_HOME=$SCR/cache/xdg
export TORCH_HOME=$SCR/cache/torch WARP_CACHE_PATH=$SCR/cache/warp HF_HOME=$SCR/cache/hf
export STABLE_ENVS_RENDERNET_CACHE=$SCR/cache/rendernet
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# HOME is at 100% of its quota: nothing may be written there. Every cache above is under $SCR.

WS=${SE9_WS:-$SCR/se9/render-mjwarp-nvdiff}
OUT=$WS/examples/results/step9/render-mjwarp-nvdiff
TAG=gpu_${SLURM_JOB_ID:-local}
BASE_SHA=21cb30678b7a7d0f3344b59f176f33d3f3b09ffc
mkdir -p "$OUT" "$SCR/logs"
cd "$WS" || exit 1
FAIL_COUNT=0

fail () {
  echo "  FAIL: $*"
  FAIL_COUNT=$((FAIL_COUNT + 1))
}

{ echo "host $(hostname)"; echo "job ${SLURM_JOB_ID:-none}"; echo "date $(date -Is)"
  echo "head $(git -C "$WS" rev-parse HEAD 2>/dev/null)"
  echo "baseline_present $(git -C "$WS" cat-file -e ${BASE_SHA}^{commit} 2>/dev/null && echo yes || echo NO)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} 2>&1 | tee "$OUT/provenance_${TAG}.txt"
git -C "$WS" cat-file -e "${BASE_SHA}^{commit}" 2>/dev/null \
  || fail "baseline commit $BASE_SHA is absent from $WS"

# Dependencies are prepared on the login node. Compute nodes may have no network, so this job checks
# rather than trying an install whose failure used to be reported as a warning and then ignored.
ensure_deps () {
  local py="$1"
  "$py" -c "import tensordict, loguru" >"$OUT/deps_${TAG}.log" 2>&1
}

# arm <label> <venv-python> <comma-separated cells>
arm () {
  local label="$1" py="$2" cells="$3"
  echo "=== arm $label ($py) ==="
  if [ ! -x "$py" ]; then fail "missing interpreter $py"; return 1; fi
  if ! ensure_deps "$py"; then fail "$label lacks tensordict/loguru"; return 1; fi
  if ! "$py" -c "import torch; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    fail "$label has no CUDA-capable torch"
    return 1
  fi

  echo "  -- 1/3 frames: baseline tree vs this branch"
  PYTHONPATH="$WS" "$py" scratch/step9/render-mjwarp-nvdiff/oracle.py \
      --python "$py" --baseline-sha "$BASE_SHA" --n-states 64 --cycles 5 \
      --cells "$cells" --tag "${label}_${TAG}" > "$OUT/oracle_${label}_${TAG}.log" 2>&1
  local rc=$?
  echo "  oracle rc=$rc :: $(grep -E '^ORACLE:' "$OUT/oracle_${label}_${TAG}.log" || true)"
  [ "$rc" -eq 0 ] || fail "$label frame oracle failed"
  grep -E '"identical"|"lifetime_ok"|"cells_ok"' "$OUT/oracle_${label}_${TAG}.log" || true

  echo "  -- 2/3 fresh-process reproducibility: the SAME tree, twice"
  PYTHONPATH="$WS" "$py" scratch/step9/render-mjwarp-nvdiff/oracle.py \
      --python "$py" --arms head,head --n-states 64 --cycles 2 \
      --cells "$cells" --tag "${label}_fresh_${TAG}" > "$OUT/fresh_${label}_${TAG}.log" 2>&1
  rc=$?
  echo "  fresh rc=$rc :: $(grep -E '^ORACLE:' "$OUT/fresh_${label}_${TAG}.log" || true)"
  [ "$rc" -eq 0 ] || fail "$label fresh-process oracle failed"

  echo "  -- 3/3 paired throughput, B=1 and B=4096 (stepping down on OOM), this card"
  PYTHONPATH="$WS" "$py" scratch/step9/render-mjwarp-nvdiff/bench.py \
      --python "$py" --baseline-sha "$BASE_SHA" --rounds 5 --reps 5 \
      --cells "$cells" --tag "${label}_${TAG}" > "$OUT/bench_${label}_${TAG}.log" 2>&1
  rc=$?
  echo "  bench rc=$rc :: $(grep -E 'worst ratio' "$OUT/bench_${label}_${TAG}.log" || true)"
  [ "$rc" -eq 0 ] || fail "$label benchmark failed or exceeded the 1.10 bar"
  grep -E 'ratio [0-9]' "$OUT/bench_${label}_${TAG}.log" || true
}

# The four arms. The cell lists are disjoint because no venv can build another's cells; a cell whose
# dependency is absent is recorded as a SKIP with its reason on both legs, which is itself compared.
arm vis311 "$SCR/venvs/se-vis311/bin/python" \
    "DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCWalkerWalk,DMCQuadrupedWalk"
arm ogb     "$SCR/venvs/se7-ogb/bin/python"       "OGBCube,OGBScene"
arm rocket  "$SCR/venvs/se7-rocket/bin/python"    "PFRocketLanding,MujocoHalfCheetah,MujocoAnt"
arm fetch   "$SCR/venvs/se-s3b-fetch311/bin/python" \
    "FetchReach,FetchPush,FetchSlide,FetchPickAndPlace"

echo "=== direct create/close and cache-overflow probes ==="
for spec in "vis311:se-vis311:mjwarp" "vis311:se-vis311:rendernet" "rocket:se7-rocket:nvdiff"; do
  label=${spec%%:*}; rest=${spec#*:}; venv=${rest%%:*}; backend=${rest##*:}
  py="$SCR/venvs/$venv/bin/python"
  log="$OUT/resource_${label}_${backend}_${TAG}.log"
  if [ ! -x "$py" ]; then
    fail "missing interpreter $py for direct $backend resource probe"
    continue
  fi
  PYTHONPATH="$WS" "$py" scratch/step9/render-mjwarp-nvdiff/resource_probe.py \
      --backend "$backend" >"$log" 2>&1
  rc=$?
  echo "  $label/$backend rc=$rc :: $(grep -E '^RESOURCE_PROBE:' "$log" || true)"
  [ "$rc" -eq 0 ] || fail "$label direct $backend resource probe failed"
done

echo "=== GPU tests that can see this workstream's code ==="
# The WHOLE pytest output goes to a file and only the summary is echoed. Workstream H lost an entire
# run's test counts to this: MuJoCo's EGL teardown prints dozens of "Exception ignored in __del__"
# lines AFTER pytest's verdict, and a `| tail -40` showed only the noise.
for v in vis311:se-vis311 ogb:se7-ogb rocket:se7-rocket fetch:se-s3b-fetch311; do
  label=${v%%:*}; py="$SCR/venvs/${v##*:}/bin/python"
  if [ ! -x "$py" ]; then fail "missing interpreter $py for GPU tests"; continue; fi
  PYTHONPATH="$WS" "$py" -m pytest -q -p no:cacheprovider \
      tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py \
      tests/gpu/test_fast_paths.py tests/gpu/test_rendernet.py \
      > "$OUT/gpu_tests_${label}_${TAG}.log" 2>&1
  rc=$?
  echo "  $label rc=$rc :: $(grep -E '[0-9]+ (passed|failed|error)' "$OUT/gpu_tests_${label}_${TAG}.log" | grep -v 'Exception ignored' || true)"
  [ "$rc" -eq 0 ] || fail "$label GPU tests failed (rc=$rc; rc=5 means no coverage)"
  PYTHONPATH="$WS" "$py" -m pytest -q -p no:cacheprovider tests/unit/test_render_resources.py \
      > "$OUT/unit_resources_${label}_${TAG}.log" 2>&1
  rc=$?
  echo "  $label unit rc=$rc :: $(grep -E '[0-9]+ (passed|failed|error)' "$OUT/unit_resources_${label}_${TAG}.log" || true)"
  [ "$rc" -eq 0 ] || fail "$label resource unit tests failed (rc=$rc)"
done

echo "artifacts in $OUT (oracle_*_${TAG}.json/.log, fresh_*, timings_*, bench_*, provenance_${TAG}.txt)"
echo "ALLDONE $(date -Is)"
if [ "$FAIL_COUNT" -eq 0 ]; then
  echo "GPU_ORACLE: PASS"
  exit 0
fi
echo "GPU_ORACLE: FAIL ($FAIL_COUNT failed sections)"
exit 1
