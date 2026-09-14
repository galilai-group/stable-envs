#!/bin/bash
#SBATCH --job-name=se6-verify-final
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
# The final incremental pass on cuda, in one allocation: the shards that carry the 21 raycast3d
# configs, the 4 DM Control trackcom cells and the 4 swm_nav configs, then the whole of
# tests/gpu/test_reproducibility.py.
#
# The trackcom cells are the reason the dmc shard runs in $SCR/venvs/se-step5 here rather than in
# se-step3: cpu-3d gives them a CPU backend, and the cuda leg still needs mujoco_warp. se-step3 has
# mujoco_warp too and reproduces the frozen shard's package set exactly, so BOTH are run and the
# pair is the evidence -- if they disagree, the interpreter is the variable, which is exactly the
# question MJWARP-1 turned out to be.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify4}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
TAG=${TAG:-_final}
OUT=$OUTROOT/replay
GOUT=$OUTROOT/gpu
mkdir -p "$OUT" "$GOUT"
cd "$TREE" || exit 1
banner

# ---- 1. the three shards that carry every config in the coordinator's list --------------------
#   dmc    -> the 10 live DMControlSuite raycast3d cells + the 4 trackcom cells
#   ogb    -> the 11 OGBench mazes (the other 11 raycast3d configs)
#   swm    -> the 4 swm_nav configs + SimplePointMaze + CraftaxClassic
for S in ${SHARDS:-"dmc ogb swm"}; do
  PY=$(venv_for "$S")/bin/python
  echo "######################## shard $S via $PY"
  PYTHONPATH=$TREE $PY examples/fidelity_step4.py --shard "$S" --device cuda \
      --states-dir "$TREE/examples/results/step4/states" --out "$OUT/${S}${TAG}.json" 2>&1 | tail -30
  echo "SHARD $S rc=$?"
done

# ---- 2. the four trackcom cells again under se-step5, the venv the coordinator named ----------
PYTHONPATH=$TREE $SCR/venvs/se-step5/bin/python examples/fidelity_step4.py \
    --configs DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCWalkerWalk --device cuda \
    --states-dir "$TREE/examples/results/step4/states" \
    --out "$OUT/trackcom_step5venv${TAG}.json" 2>&1 | tail -20
echo "TRACKCOM_STEP5 rc=$?"

# ---- 3. Gate A over the three shards ----------------------------------------------------------
CDIR=$OUT/certify${TAG}
mkdir -p "$CDIR/conformance"
for S in ${SHARDS:-"dmc ogb swm"}; do cp -f "$OUT/${S}${TAG}.json" "$CDIR/conformance/${S}.json"; done
PYTHONPATH=$TREE $(venv_for dmc)/bin/python scratch/step4/certify.py --shards "$CDIR" --out "$CDIR" \
    --doc "$SCR/logs/verify_final_doc_${SLURM_JOB_ID}.md" 2>&1 | tail -4
echo "GATE_A rc=$?"

# ---- 4. tests/gpu/test_reproducibility.py IN FULL, plus the VISIBILITY-1 regression -----------
PY=$(venv_for verify)/bin/python
PYTHONPATH=$TREE $PY -m pytest tests/gpu/test_reproducibility.py --collect-only -q -p no:randomly 2>&1 | tail -3
PYTHONPATH=$TREE timeout 5400 $PY -m pytest tests/gpu/test_reproducibility.py -v -p no:randomly \
    2>&1 | keep "$GOUT/test_reproducibility${TAG}.results.txt"
echo "REPRO exit=${PIPESTATUS[0]}"
# Called out separately so the VISIBILITY-1 fix has a line of its own rather than a count to subtract.
PYTHONPATH=$TREE $PY -m pytest tests/gpu/test_reproducibility.py -v -p no:randomly \
    -k "a_baked_scene_does_not_inherit" 2>&1 | keep "$GOUT/baked_visibility${TAG}.results.txt"
echo "BAKED_VISIBILITY exit=${PIPESTATUS[0]}"
echo "FINAL_DONE $(date -u +%FT%TZ)"
