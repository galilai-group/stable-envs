#!/bin/bash
#SBATCH --job-name=se6-verify-cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The CPU half of the integration certification, on a batch node at ONE torch thread (AGENT_COMMON:
# login-node and all-core numbers are not certification numbers).
#
# WHAT A CPU REGRESSION STATEMENT CAN EVEN BE HERE. Every digest in examples/results/step4 was taken
# on cuda (raycast3d findings §"no frozen CPU oracle"), so "reproduces the frozen digest" is not
# available on CPU. Two statements are:
#   * BASELINE tree vs INTEGRATION tree, same node, same job, same frozen states, byte for byte.
#     That is the regression test, and it is a pixel diff rather than a hash comparison because a
#     moved digest with no diff beside it says nothing about how far it moved.
#   * CPU vs the frozen CUDA digest, reported as information. raster2d established that five of its
#     six configs agree across devices and one legitimately does not, so a disagreement here is a
#     fact to record, not a bar to fail.
#
# The renderer is scratch/step6/raycast3d/replay_step4.py, which the raycast3d workstream wrote for
# exactly this and which APPENDS its own repo root to sys.path instead of inserting it -- so the
# baseline leg run out of this tree really imports $SCR/se6/baseline and not this one. A script that
# inserts its root at position 0 makes both legs measure the same tree and report 1.00x.
#
# Overridable for the coordinator's post-merge re-run:
#     sbatch --export=ALL,RAY='',RASTER='PushT,TwoRoom',TAG='_swmnav' cpu_verify.sbatch
#     sbatch --export=ALL,GFX=1,TAG='_post_gfxdraw' cpu_verify.sbatch
set -uo pipefail
export SCR=/oscar/scratch/$USER
TREE=${TREE:-$SCR/se6/verify}
source "$TREE/scratch/step6/verify/_env.sh" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
TAG=${TAG:-}
OUT=$OUTROOT/cpu
D=$SCR/se6/_dumps/verify_cpu${TAG}
mkdir -p "$OUT"
cd "$TREE" || exit 1
banner

# raster2d's six + CraftaxClassic (tilemap) -- the 2-D CPU paths that were merged.
RASTER=${RASTER:-"PushT SimplePointMaze TwoRoom TwoRoomTarget Piecewise PiecewiseZones CraftaxClassic"}
# raycast3d's 21 LIVE configs. The other four DM Control cells (Cheetah/Hopper/Humanoid/Walker) are
# MJWarp-only and have no CPU renderer at all, which is the cpu-3d workstream's open problem.
RAY=${RAY:-"AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"}
# gfxdraw's five. OFF by default: dashboard.yaml has gfxdraw at status "review", not merged.
GFXC=${GFXC:-"Acrobot CartPole MountainCar MountainCarContinuous Pendulum"}
ALL="$RASTER $RAY"
[ "${GFX:-0}" = 1 ] && ALL="$ALL $GFXC"
PY=$(venv_for verify)/bin/python
R=$TREE/scratch/step6/raycast3d/replay_step4.py
echo "configs: $ALL"
rm -rf "$D"; mkdir -p "$D/baseline" "$D/integration"
$PY -c "import torch;torch.set_num_threads(1);print('threads',torch.get_num_threads())"

PYTHONPATH=$BASE $PY "$R" --device cpu --configs $ALL --dump-dir "$D/baseline" \
    --out "$OUT/cpu_baseline${TAG}.json" 2>&1 | tail -35
echo "BASELINE_LEG rc=$?"
PYTHONPATH=$TREE $PY "$R" --device cpu --configs $ALL --dump-dir "$D/integration" \
    --out "$OUT/cpu_integration${TAG}.json" 2>&1 | tail -35
echo "INTEGRATION_LEG rc=$?"
PYTHONPATH=$TREE $PY "$R" --compare-dumps "$D/baseline" "$D/integration" \
    --out "$OUT/cpu_pixel_diff${TAG}.json" 2>&1 | tail -35
echo "PIXEL_DIFF rc=$?"
rm -rf "$D"

# --- tests/unit, in the verify interpreter -------------------------------------------------------
PYTHONPATH=$TREE $PY -m pytest tests/unit -v -p no:randomly 2>&1 | keep "$OUT/test_unit${TAG}.results.txt"
echo "CPU_VERIFY_DONE $(date -u +%FT%TZ)"
