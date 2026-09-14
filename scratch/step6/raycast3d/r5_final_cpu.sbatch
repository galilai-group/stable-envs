#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5finalcpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The identity that matters once PRIMITIVE_BOUNDS defaults to True: NEITHER leg pins the switch, so
# this is the pristine baseline against the branch exactly as a user gets it. Every earlier identity
# run had to force the bound on, because with the default off a replay taking the default would have
# compared the unbounded path against itself and reported ALL_IDENTICAL for the wrong reason.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r5final
set -x
cd "$WS"
rm -rf "$D"; mkdir -p "$D"
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

$PY -c "import sys; sys.path.insert(0,'$WS'); from stable_envs.render import raycast3d as R3; \
print('BRANCH DEFAULT PRIMITIVE_BOUNDS =', R3.PRIMITIVE_BOUNDS, 'kinds', sorted(R3._BOUNDED_KINDS))"

PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/base" --out "$OUT/replay/r5final_cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/branch" --out "$OUT/replay/r5final_cpu_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" \
    --out "$OUT/replay/r5final_cpu_identity.json" || echo "DIFF rc=$?"

PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -8
echo "R5_FINAL_CPU done"
