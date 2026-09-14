#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-5 verification on the device it targets, one node, one job.
#   1. PRIMITIVE_BOUNDS on vs off, all 21 -- the identity, and it must be FORCED on now that the
#      module default is False, or the "bounded" leg is the unbounded one and the check is vacuous;
#   2. branch (bound forced on) vs pristine baseline, all 21;
#   3. the acceptance sweep: same branch, bound off then on, one thread, eight rows. >= 0.97x
#      everywhere is the bar, and Pointmaze Large/Medium must come back to ~1.00x;
#   4. tests/unit, including the batch-invariance and per-kind cases.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r5cpu
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
rm -rf "$D"; mkdir -p "$D"

ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## 1. PRIMITIVE_BOUNDS on == off, per-kind predicate live, all 21"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --primitive-bounds on  --dump-dir "$D/on"  --out "$OUT/replay/r5_cpu_bounds_on.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --primitive-bounds off --dump-dir "$D/off" --out "$OUT/replay/r5_cpu_bounds_off.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/off" "$D/on" \
    --out "$OUT/replay/r5_cpu_bounds_identity.json" || echo "IDENT rc=$?"

echo "######## 2. branch (bound ON) vs pristine baseline, all 21"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/base" --out "$OUT/replay/r5_cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/on" \
    --out "$OUT/replay/r5_cpu_branch_vs_baseline.json" || echo "DIFF rc=$?"

echo "######## 3. acceptance sweep: bound off vs on, same branch, one thread"
for E in DMCReacherHard DMCFingerSpin DMCCartpoleSwingup DMCPendulumSwingup DMCAcrobotSwingup \
         DMCBallInCupCatch PointmazeLargeNavigate PointmazeMediumNavigate; do
  for S in off on; do
    PYTHONPATH=$WS $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
        --batch 1 16 64 256 --repeat 5 --warmup 2 --primitive-bounds $S \
        --states-dir "$WS/examples/results/step4/states" \
        --out "$OUT/profile_after_r5/bounds_$S" || echo "ISO $E $S rc=$?"
  done
done

echo "######## 4. tests/unit"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_raycast3d_primitive_bounds.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -8
echo "R5_CPU done"
