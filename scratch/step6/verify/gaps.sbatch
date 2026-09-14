#!/bin/bash
#SBATCH --job-name=se6-verify-gaps
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The three things the main GPU pass could not settle, in one allocation on one card.
#
# 1. FETCH. The full tests/gpu suite ran in $SCR/venvs/se6-verify, which has no gymnasium_robotics,
#    so all 19 conformance tests for each of the 12 Fetch cells ERRORED at fixture setup with
#    ModuleNotFoundError rather than running. Fetch is 12 of the 49 frozen configurations, so
#    "errored on a missing import" is not an acceptable place to leave them. $SCR/venvs/se6-verify-
#    fetch311 is built the same way se6-verify is -- a --without-pip venv, no installs -- chaining
#    se-s3b-fetch311 (mujoco 3.11, mujoco_warp 3.11, gymnasium_robotics 1.4.2: the interpreter the
#    fetch shard was FROZEN in) first, then se-step6 for pytest only.
#
# 2. THE FOUR test_state_is_sufficient FAILURES on the giant/teleport OGBench mazes. That test drives
#    the OFFICIAL renderer through two histories and never calls ours, so this branch cannot have
#    caused it -- but "cannot" is an argument. Running the same four tests against the BASELINE tree
#    and against the integration tree in one job on one card turns it into a measurement. The test's
#    own docstring records that this exact symptom was once EGL-context exhaustion rather than a real
#    violation, and this pass constructs more env families in one process than any previous run, so
#    each leg also runs with only these four tests selected -- no other env holds a context.
#
# 3. THE THREE test_visual_tier FAILURES (Cheetah 6.541 vs 6.5, Hopper, Walker). texture-fidelity
#    established these are the MuJoCo 3.11->3.12 floor-texture drift, and scratch/step4/certify.py
#    now BLOCKS the corresponding Gate A check on the reference-version mismatch -- but
#    tests/gpu/test_conformance.py still FAILS on it. Baseline vs integration on one card says
#    whether the number belongs to this branch or to the interpreter.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify2}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
TAG=${TAG:-}
OUT=$OUTROOT/gpu
mkdir -p "$OUT"
cd "$TREE" || exit 1
banner
PYF=$SCR/venvs/se6-verify-fetch311/bin/python
PYV=$(venv_for verify)/bin/python

echo "######################## 1. Fetch, in the interpreter the fetch shard was frozen in"
$PYF -c "import mujoco,mujoco_warp,gymnasium_robotics,pytest;print('mujoco',mujoco.__version__,'mjwarp',mujoco_warp.__version__,'gymrob',gymnasium_robotics.__version__)"
for F in tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py tests/gpu/test_rollout_on_device.py; do
  B=$(basename "$F" .py)
  PYTHONPATH=$TREE timeout 3000 $PYF -m pytest "$F" -v -p no:randomly -k "Fetch" \
      2>&1 | keep "$OUT/fetch_${B}${TAG}.results.txt"
  echo "FETCH FILE $F exit=${PIPESTATUS[0]}"
done

echo "######################## 2+3. the seven failures, BASELINE vs INTEGRATION, one card"
for LEG in baseline integration; do
  [ "$LEG" = baseline ] && T=$BASE || T=$TREE
  echo "---- leg $LEG  ($T)"
  PYTHONPATH=$T $PYV -m pytest "$TREE/tests/gpu/test_conformance.py" -v -p no:randomly \
      -k "state_is_sufficient and (PointmazeGiant or PointmazeTeleport)" \
      2>&1 | keep "$OUT/state_sufficient_${LEG}${TAG}.results.txt"
  echo "STATE_SUFFICIENT $LEG exit=${PIPESTATUS[0]}"
  PYTHONPATH=$T $PYV -m pytest "$TREE/tests/gpu/test_conformance.py" -v -p no:randomly \
      -k "visual_tier and (CheetahRun or HopperHop or WalkerWalk)" \
      2>&1 | keep "$OUT/visual_tier_${LEG}${TAG}.results.txt"
  echo "VISUAL_TIER $LEG exit=${PIPESTATUS[0]}"
  # 4. the fresh-process reproducibility failures, and the node's own reference determinism. The
  #    reference-determinism test says in its own message that a non-zero result is a NODE fault and
  #    that every parity number taken on that host is meaningless, so it is re-run here on whichever
  #    card this job lands on: if it is clean here and was not on gpu2108, the host is the answer.
  PYTHONPATH=$T $PYV -m pytest "$TREE/tests/gpu/test_reproducibility.py" -v -p no:randomly \
      -k "fresh_process and (AntmazeTeleport or HumanoidmazeLarge)" \
      2>&1 | keep "$OUT/fresh_process_${LEG}${TAG}.results.txt"
  echo "FRESH_PROCESS $LEG exit=${PIPESTATUS[0]}"
  PYTHONPATH=$T $PYV -m pytest "$TREE/tests/gpu/test_reproducibility.py" -v -p no:randomly \
      -k "reference_renderer_on_this_machine_is_deterministic and (Pointmaze or Antmaze or Humanoidmaze)" \
      2>&1 | keep "$OUT/node_reference_${LEG}${TAG}.results.txt"
  echo "NODE_REFERENCE $LEG exit=${PIPESTATUS[0]}"
done

# 5. the scene-cache visibility regression test, over EVERY DM Control cell rather than the four the
#    reacher workstream ran. It fails on the seven cells whose episode draws no task object, and the
#    assertion that fires is its own selector's ("advertises VISIBILITY_KNOB but declares no
#    visibility.* leaf"), not a pixel comparison -- so this run is what the report cites, and the
#    per-cell list is the evidence for which cells are affected.
PYTHONPATH=$TREE $PYV -m pytest "$TREE/tests/gpu/test_reproducibility.py" -v -p no:randomly \
    -k "a_baked_scene_does_not_inherit" 2>&1 | keep "$OUT/baked_visibility${TAG}.results.txt"
echo "BAKED_VISIBILITY exit=${PIPESTATUS[0]}"

# 6. tests/gpu/test_gfxdraw_fused.py went RED on an ALL-SKIPPED run: its reference rasterizer is
#    carracing's, which needs Box2D, which se6-verify does not have, and the repo's own EMPTY RUN
#    guard correctly refuses to call four skips a pass. $SCR/venvs/se6-gfxdraw-box2d is the venv the
#    gfxdraw workstream built for exactly this and it has Box2D 2.3.8 and pytest.
PYB=$SCR/venvs/se6-gfxdraw-box2d/bin/python
PYTHONPATH=$TREE $PYB -m pytest "$TREE/tests/gpu/test_gfxdraw_fused.py" -v -rs -p no:randomly \
    2>&1 | keep "$OUT/gfxdraw_fused_box2d${TAG}.results.txt"
echo "GFXDRAW_FUSED_BOX2D exit=${PIPESTATUS[0]}"

# 7. tests/gpu/test_rendernet.py::test_no_checkpoint_means_no_class fails rather than skips because
#    pytest 9's importorskip does not swallow an ImportError raised INSIDE a module that exists, and
#    stable_envs/envs/robocasa.py exists while robocasa does not. Nothing in Step 6 touches it, so
#    the baseline leg is the whole argument.
for LEG in baseline integration; do
  [ "$LEG" = baseline ] && T=$BASE || T=$TREE
  PYTHONPATH=$T $PYV -m pytest "$TREE/tests/gpu/test_rendernet.py" -v -rs -p no:randomly \
      2>&1 | keep "$OUT/rendernet_${LEG}${TAG}.results.txt"
  echo "RENDERNET $LEG exit=${PIPESTATUS[0]}"
done

# 8. the skip REASONS for test_distributed, so "skipped" is a recorded reason and not a shrug.
PYTHONPATH=$TREE $PYV -m pytest "$TREE/tests/gpu/test_distributed.py" -v -rs -p no:randomly \
    2>&1 | tee "$SCR/logs/${SLURM_JOB_ID}_distributed_reasons.txt" | grep -aE "SKIPPED|skipped|cuda devices" \
    > "$OUT/distributed_reasons${TAG}.results.txt" || true
echo "DISTRIBUTED_REASONS written"
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l

echo "GAPS_DONE $(date -u +%FT%TZ)"
