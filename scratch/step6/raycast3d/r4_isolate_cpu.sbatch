#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r4iso
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-4, attribution. The main sweep's "before" leg is the PRISTINE pre-Step-6 baseline, so its
# ratios are cumulative over 2a + 2b + R4 and none of them belongs to the bound alone. This job runs
# the SAME branch twice, PRIMITIVE_BOUNDS off then on, one thread, one node, one job -- the only leg
# that can say what the bound itself is worth.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1

for E in DMCReacherHard DMCFingerSpin DMCCartpoleSwingup DMCPendulumSwingup DMCAcrobotSwingup \
         DMCBallInCupCatch PointmazeLargeNavigate PointmazeMediumNavigate; do
  for S in off on; do
    PYTHONPATH=$WS $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
        --batch 1 16 64 256 --repeat 5 --warmup 2 --primitive-bounds $S \
        --states-dir "$WS/examples/results/step4/states" \
        --out "$OUT/profile_after_r4/bounds_$S" || echo "ISO $E $S rc=$?"
  done
done
echo "R4_ISO done"
