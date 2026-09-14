#!/bin/bash
#SBATCH --job-name=se6-verify-replay
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The frozen Step-4 replay of the render-distinct corpus on the INTEGRATION branch, on cuda, one
# card, one allocation.
#
# This is not a new oracle. It is examples/fidelity_step4.py, the script that produced
# examples/results/step4/conformance/<shard>.json, run again over the same read-only frozen states
# with the same seeds, in the same interpreter each shard was frozen in (see venv_for in _env.sh).
# The comparison lives in compare49.py and is a field-by-field diff against the frozen file.
#
# SUBSETTABLE, because the coordinator will ask for this again after the gfxdraw and swm_nav merges:
#     sbatch --export=ALL,SHARDS='raster swm',TAG='_post_gfxdraw' replay.sbatch
#     sbatch --export=ALL,CONFIGS='Acrobot,CartPole',VENV_KEY=dmc,TAG='_spot' replay.sbatch
#     sbatch --export=ALL,TREE=$SCR/se6/verify2,TAG='_v2' replay.sbatch
# SHARDS  space-separated subset of "dmc ogb raster swm fetch"   (default: all five)
# CONFIGS comma-separated config ids; overrides SHARDS, needs VENV_KEY to pick the interpreter
# TAG     suffix on every output file, so a re-run never overwrites the run it is compared against
# TREE    the tree to measure (default $SCR/se6/verify)
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
SHARDS=${SHARDS:-"dmc ogb raster swm fetch"}
CONFIGS=${CONFIGS:-}
VENV_KEY=${VENV_KEY:-}
TAG=${TAG:-}
DEV=${DEV:-cuda}
OUT=$OUTROOT/replay
mkdir -p "$OUT"
cd "$TREE" || exit 1
banner
echo "SHARDS='$SHARDS' CONFIGS='$CONFIGS' TAG='$TAG' DEV=$DEV"

rc_all=0
if [ -n "$CONFIGS" ]; then
  PY=$(venv_for "${VENV_KEY:-verify}")/bin/python
  echo "### configs $CONFIGS via $PY"
  PYTHONPATH=$TREE $PY examples/fidelity_step4.py --configs "$CONFIGS" --device "$DEV" \
      --states-dir "$TREE/examples/results/step4/states" \
      --out "$OUT/adhoc${TAG}.json" 2>&1 | tail -40
  rc_all=$?
else
  for S in $SHARDS; do
    PY=$(venv_for "$S")/bin/python
    echo "######################## shard $S  via $PY"
    $PY -c "import torch,sys;print('  py',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())"
    PYTHONPATH=$TREE $PY examples/fidelity_step4.py --shard "$S" --device "$DEV" \
        --states-dir "$TREE/examples/results/step4/states" \
        --out "$OUT/${S}${TAG}.json" 2>&1 | tail -40
    echo "SHARD $S rc=$? -> $OUT/${S}${TAG}.json"
    [ -s "$OUT/${S}${TAG}.json" ] || { echo "SHARD $S PRODUCED NOTHING"; rc_all=1; }
  done
fi

# ---- Gate A, from the replayed shards, through the Step-4 certifier that owns the bounds ---------
# certify.py reads the bound for each check out of the frozen Step-2 spec and each env's own
# declaration; nothing here judges anything itself. --doc is pointed OUTSIDE the tree so the Step-4
# baseline document is never rewritten by a Step-6 job.
if [ -z "$CONFIGS" ]; then
  CDIR=$OUT/certify${TAG}
  mkdir -p "$CDIR/conformance"
  for S in $SHARDS; do cp -f "$OUT/${S}${TAG}.json" "$CDIR/conformance/${S}.json" 2>/dev/null; done
  PYTHONPATH=$TREE $(venv_for dmc)/bin/python scratch/step4/certify.py \
      --shards "$CDIR" --out "$CDIR" --doc "$SCR/logs/verify_step4_doc${TAG}_${SLURM_JOB_ID}.md" \
      2>&1 | tail -5
  echo "GATE_A_RC=$?"
fi
echo "REPLAY_DONE $(date -u +%FT%TZ)"
