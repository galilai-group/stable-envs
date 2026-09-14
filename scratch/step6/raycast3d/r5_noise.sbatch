#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5noise
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Is the residual a COST or is it the harness's spread? Four legs, INTERLEAVED off/on/off/on, same
# branch, same node, same job, more repeats. off2/off1 is a ratio of two runs of IDENTICAL code, so
# it measures nothing but the harness; if that spread is wider than the 0.97x bar, the bar cannot be
# evaluated at that row and saying "regression" would be reading noise.
#
# The rows are the ones still under the bar after the call-level short circuit, plus DMCFingerSpin as
# a control that was comfortably over it (1.20x): a noise estimate is only worth having if it is also
# taken where the answer is not in doubt.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1
cat /proc/loadavg

for E in PointmazeMediumNavigate PointmazeLargeNavigate DMCReacherHard DMCCartpoleSwingup DMCFingerSpin; do
  for LEG in off1 on1 off2 on2; do
    S=${LEG%[12]}
    PYTHONPATH=$WS $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
        --batch 1 16 64 256 --repeat 15 --warmup 4 --primitive-bounds $S \
        --states-dir "$WS/examples/results/step4/states" \
        --out "$OUT/profile_after_r5/noise_$LEG" || echo "NOISE $E $LEG rc=$?"
  done
done
echo "R5_NOISE done"
