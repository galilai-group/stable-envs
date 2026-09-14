#!/bin/bash
#SBATCH --job-name=se6-verify-postmerge
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
# NODE-1: the OFFICIAL OGBench renderer is nondeterministic at 1/255 on these two, on the
# baseline tree as much as on any branch, and the suite aborts its own session over it.
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# One GPU allocation that re-runs the affected SUBSET after a merge lands on step6/integration, and
# then the full tests/gpu suite, in that order: the replay is what a merge can break silently, the
# suite is what it breaks loudly, and a job that is cut short should have produced the quiet evidence
# first. `gpu`/`norm-gpu` allows one GPU job per agent, so the two cannot be separate jobs without
# paying the queue twice.
#
#     sbatch --export=ALL,TREE=$SCR/se6/verify2,SHARDS='raster',TAG='_gfx' postmerge.sbatch
#     sbatch --export=ALL,TREE=$SCR/se6/verify3,SHARDS='swm raster',TAG='_swmnav' postmerge.sbatch
# SHARDS  the frozen shards to replay (default: raster -- the shard the gfxdraw merge touches)
# FILES   the tests/gpu files to run (default: all nine on the integration branch)
# TAG     suffix on every output file
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify2}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
SHARDS=${SHARDS:-"raster"}
FILES=${FILES:-"tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py tests/gpu/test_gfxdraw_classic.py tests/gpu/test_gfxdraw_fused.py tests/gpu/test_rendernet.py tests/gpu/test_crash_isolation.py tests/gpu/test_rollout_on_device.py tests/gpu/test_distributed.py"}
TAG=${TAG:-}
cd "$TREE" || exit 1
banner

# ---- 1. the frozen replay of the affected shards -------------------------------------------------
OUT=$OUTROOT/replay
mkdir -p "$OUT"
for S in $SHARDS; do
  PY=$(venv_for "$S")/bin/python
  echo "######################## shard $S via $PY"
  PYTHONPATH=$TREE $PY examples/fidelity_step4.py --shard "$S" --device cuda \
      --states-dir "$TREE/examples/results/step4/states" --out "$OUT/${S}${TAG}.json" 2>&1 | tail -30
  echo "SHARD $S rc=$?"
done

# ---- 1b. optional: the four trackcom cells under se-step5 ----------------------------------------
# TRACKCOM5=1 adds the arm the coordinator names for these cells. It is a minute of work and it keeps
# the pair from section 5 of the report alive, so a change under them could never be confused again
# with the venv difference that closed MJWARP-1.
if [ "${TRACKCOM5:-0}" = 1 ]; then
  for LEG in baseline integration; do
    [ "$LEG" = baseline ] && T=$BASE || T=$TREE
    PYTHONPATH=$T $SCR/venvs/se-step5/bin/python "$TREE/scratch/step6/raycast3d/replay_step4.py" \
        --device cuda --configs DMCCheetahRun DMCHopperHop DMCHumanoidStand DMCWalkerWalk \
        --out "$OUT/trackcom_se-step5_${LEG}${TAG}.json" 2>&1 | tail -8
    echo "TRACKCOM5 $LEG rc=$?"
  done
fi

# ---- 2. the full tests/gpu suite, one file at a time ---------------------------------------------
# Collection is asserted separately from execution: tests/gpu/test_fast_paths.py was hand-merged
# between two workstreams' blocks, and a file that fails to COLLECT reports zero failures.
GOUT=$OUTROOT/gpu
mkdir -p "$GOUT"
PY=$(venv_for verify)/bin/python
$PY -c "import torch,mujoco,pytest,mujoco_warp;print('mjwarp',mujoco_warp.__version__,'cuda devices',torch.cuda.device_count(),torch.cuda.get_device_name(0))"
export PYTHONPATH=$TREE
for F in $FILES; do
  echo "### collect $F"
  $PY -m pytest "$F" --collect-only -q -p no:randomly 2>&1 | tail -3
done
for F in $FILES; do
  B=$(basename "$F" .py)
  echo "######################## $F"
  timeout 3600 $PY -m pytest "$F" -v -p no:randomly 2>&1 | keep "$GOUT/${B}${TAG}.results.txt"
  echo "FILE $F exit=${PIPESTATUS[0]}"
done
echo "POSTMERGE_DONE $(date -u +%FT%TZ)"
