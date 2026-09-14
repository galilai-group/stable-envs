#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The changes independent review of 3b9e81c required.
#   1. _UNIT_D_TOL 1e-4 -> 1e-6. Tightening an ADMISSION check has its own failure mode: a real
#      bundle that now falls outside it takes the unbounded path, the pixels stay identical because
#      that path is the definition, and the bound silently stops applying where it was measured to
#      pay. So the probe instruments _unit_rays itself -- rejected bundles must be zero -- and the
#      identity replay covers all 21 configs with nothing excluded.
#   2. The per-kind cost table, re-priced in the SHAPE THAT SHIPS: cast_nearest, which gathers best
#      and scatters best/nrm/idx, not the cheaper any_blocking shape the first run timed.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r6
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
rm -rf "$D"; mkdir -p "$D"
ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## A. does any real bundle fail the tightened check?"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_unit_ray_probe.py --device cpu \
    --out "$OUT/experiments/r6_unit_ray_probe_cpu.json" || echo "PROBE rc=$?"

echo "######## B. the per-kind cost table in the shape that ships"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r5_kind_cost.py --threads 1 --shape nearest \
    --out "$OUT/experiments/r6_kind_cost_nearest.json" || echo "COSTN rc=$?"

echo "######## C. PRIMITIVE_BOUNDS on == off, all 21, nothing excluded"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --primitive-bounds on  --dump-dir "$D/on"  --out "$OUT/replay/r6_cpu_bounds_on.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --primitive-bounds off --dump-dir "$D/off" --out "$OUT/replay/r6_cpu_bounds_off.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/off" "$D/on" \
    --out "$OUT/replay/r6_cpu_bounds_identity.json" || echo "IDENT rc=$?"

echo "######## D. and against the pristine baseline, defaults on both legs"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/base" --out "$OUT/replay/r6_cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/branch" --out "$OUT/replay/r6_cpu_branch.json" || echo "BR rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" \
    --out "$OUT/replay/r6_cpu_branch_vs_baseline.json" || echo "DIFF rc=$?"

echo "######## E. tests"
PYTHONPATH=$WS $PY -m pytest tests/unit/test_raycast3d_primitive_bounds.py -q -p no:randomly 2>&1 | tail -6
PYTHONPATH=$WS $PY -m pytest tests/unit -q -p no:randomly 2>&1 | tail -6
echo "R6_CPU done"
