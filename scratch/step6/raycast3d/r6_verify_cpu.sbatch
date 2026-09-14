#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6vcpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-6 CPU verification. The "before" tree is $SCR/se6/integration -- the tree this branch was
# merged onto -- and NOT the pre-Step-6 baseline: the comparison this ticket owes is
# current-integration against the change, not against a tree five tickets older.
#
# Four legs, and the third and fourth are the ones that would catch a defect the others cannot:
#   A  COMPOSITE_COMPACT_MERGE on vs off on this branch  -- the switch against its own off-path
#   B  branch vs integration, DEFAULTS on both legs      -- what a caller actually gets
#   C  the four DMControl trackcom cells, which reach raycast3d through cpu3d.py rather than through
#      composite: a control that the new consumer is untouched
#   D  tests
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r6cm
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
rm -rf "$D"; mkdir -p "$D"

ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"
TRACKCOM="DMCCheetahRun DMCHopperHop DMCHumanoidStand DMCWalkerWalk"

echo "######## A. COMPOSITE_COMPACT_MERGE on == off, all 21 live configs"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --compact-merge on  --dump-dir "$D/on"  --out "$OUT/replay/r6cm_cpu_on.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --compact-merge off --dump-dir "$D/off" --out "$OUT/replay/r6cm_cpu_off.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/off" "$D/on" \
    --out "$OUT/replay/r6cm_cpu_switch_identity.json" || echo "SWITCH rc=$?"

echo "######## B. branch vs integration, defaults on BOTH legs -- neither pins the switch"
PYTHONPATH=$INT $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/int" --out "$OUT/replay/r6cm_cpu_integration.json" || echo "INT rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/br"  --out "$OUT/replay/r6cm_cpu_branch.json" || echo "BR rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/int" "$D/br" \
    --out "$OUT/replay/r6cm_cpu_branch_vs_integration.json" || echo "DIFF rc=$?"

echo "######## C. the trackcom cells: cpu3d.py is a NEW consumer of cast_nearest/any_blocking"
PYTHONPATH=$INT $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $TRACKCOM \
    --dump-dir "$D/tc_int" --out "$OUT/replay/r6cm_trackcom_integration.json" || echo "TCI rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $TRACKCOM \
    --dump-dir "$D/tc_br"  --out "$OUT/replay/r6cm_trackcom_branch.json" || echo "TCB rc=$?"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/tc_int" "$D/tc_br" \
    --out "$OUT/replay/r6cm_trackcom_identity.json" || echo "TCD rc=$?"

echo "######## D. tests"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_raycast3d_composite_compact.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -6
echo "R6_VCPU done"
