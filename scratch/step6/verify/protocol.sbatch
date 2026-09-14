#!/bin/bash
#SBATCH --job-name=se6-verify-protocol
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The dmc shard and its Gate A, after 5402387 made check_resolution_consistency judge the bound's own
# 8-state protocol draw instead of the 64-state frozen corpus.
#
# TWO ARMS, because the four visual-tier cells are the whole question and the two arms answer
# different halves of it:
#
#   se-step3            mujoco 3.12 + mujoco_warp 3.12 + dm_control 1.0.45 -- the interpreter the dmc
#                       shard was FROZEN in, so the 10 parity cells' digests are comparable. The four
#                       visual cells are BLOCKED here BY DESIGN: their visual_fidelity is declared
#                       against 3.11 and texture-fidelity's provenance gate refuses to judge it
#                       against a drifted reference.
#   se6-verify-vis311   mujoco 3.11 + mujoco_warp 3.11 + dm_control 1.0.44 -- the pair those four
#                       cells' numbers were DECLARED against, which is the arm in which "the visual
#                       tier passes" is a statement rather than a hope. Built by this workstream the
#                       same way as the others: --without-pip, one .pth, nothing installed. Run over
#                       the four cells only -- the other ten declare against 3.12 and judging them
#                       here would just move the provenance gate onto a different set of rows.
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify6}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
TAG=${TAG:-_protocol}
VIS=${VIS:-DMCCheetahRun,DMCHopperHop,DMCHumanoidStand,DMCWalkerWalk}
OUT=$OUTROOT/replay
mkdir -p "$OUT"
cd "$TREE" || exit 1
banner

echo "######################## arm 3.12 (se-step3): the whole dmc shard"
PYTHONPATH=$TREE $(venv_for dmc)/bin/python examples/fidelity_step4.py --shard dmc --device cuda \
    --states-dir "$TREE/examples/results/step4/states" --out "$OUT/dmc312${TAG}.json" 2>&1 | tail -25
echo "ARM312 rc=$?"
C312=$OUT/certify312${TAG}; mkdir -p "$C312/conformance"
cp -f "$OUT/dmc312${TAG}.json" "$C312/conformance/dmc.json"
PYTHONPATH=$TREE $(venv_for dmc)/bin/python scratch/step4/certify.py --shards "$C312" --out "$C312" \
    --doc "$SCR/logs/protocol312_${SLURM_JOB_ID}.md" 2>&1 | tail -3

echo "######################## arm 3.11 (se6-verify-vis311): the four visual cells"
PY311=$SCR/venvs/se6-verify-vis311/bin/python
$PY311 -c "import mujoco,mujoco_warp,importlib.metadata as md;print(' mujoco',mujoco.__version__,'mjwarp',mujoco_warp.__version__,'dm_control',md.version('dm-control'))"
PYTHONPATH=$TREE $PY311 examples/fidelity_step4.py --configs "$VIS" --device cuda \
    --states-dir "$TREE/examples/results/step4/states" --out "$OUT/dmc311${TAG}.json" 2>&1 | tail -20
echo "ARM311 rc=$?"
C311=$OUT/certify311${TAG}; mkdir -p "$C311/conformance"
cp -f "$OUT/dmc311${TAG}.json" "$C311/conformance/dmc.json"
PYTHONPATH=$TREE $PY311 scratch/step4/certify.py --shards "$C311" --out "$C311" \
    --doc "$SCR/logs/protocol311_${SLURM_JOB_ID}.md" 2>&1 | tail -3
echo "PROTOCOL_DONE $(date -u +%FT%TZ)"
