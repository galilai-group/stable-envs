#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r4cpu
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-4 on the device it targets. Three things, one job, one node:
#   1. all 21 live configs on CPU, pristine baseline vs branch, in pixels -- the regression check,
#      since the frozen Step-4 oracle is cuda-only and has nothing to say about CPU frames;
#   2. PRIMITIVE_BOUNDS on vs off on the branch, same 21 -- the identity the switch claims;
#   3. the 1-thread certification sweep, before and after, for the eight rows the ticket names.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
D=$SCR/se6/_dumps/r4cpu
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
rm -rf "$D"; mkdir -p "$D"

ALL="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate \
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCFingerSpin \
DMCFingerTurnEasy DMCFingerTurnHard DMCPendulumSwingup DMCReacherEasy DMCReacherHard"

echo "######## 1. CPU frames, pristine baseline vs branch, all 21"
PYTHONPATH=$BASE $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/base" --out "$OUT/replay/r4_cpu_baseline.json" || echo "BASE rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --dump-dir "$D/branch" --out "$OUT/replay/r4_cpu_branch.json" || echo "BRANCH rc=$?"
PYTHONPATH=$WS   $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/base" "$D/branch" \
    --out "$OUT/replay/r4_cpu_branch_vs_baseline.json" || echo "DIFF rc=$?"

echo "######## 2. PRIMITIVE_BOUNDS on == off, on CPU, all 21"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --primitive-bounds on  --dump-dir "$D/on"  --out "$OUT/replay/r4_cpu_bounds_on.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --device cpu --configs $ALL \
    --primitive-bounds off --dump-dir "$D/off" --out "$OUT/replay/r4_cpu_bounds_off.json"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/replay_step4.py --compare-dumps "$D/off" "$D/on" \
    --out "$OUT/replay/r4_cpu_bounds_identity.json" || echo "IDENT rc=$?"

echo "######## 3. the 1-thread CPU sweep, before and after"
for E in DMCReacherHard DMCFingerSpin DMCCartpoleSwingup DMCPendulumSwingup DMCAcrobotSwingup \
         DMCBallInCupCatch PointmazeLargeNavigate PointmazeMediumNavigate; do
  PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_r4/before_baseline" || echo "SB $E rc=$?"
  PYTHONPATH=$WS   $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_r4/after" || echo "SA $E rc=$?"
done
echo "R4_CPU done"
