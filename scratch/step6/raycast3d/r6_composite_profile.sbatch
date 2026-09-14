#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6prof
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-6 design step: per-op attribution inside composite at ss=2, one thread, on the CPU
# certification partition -- the residual RAYCAST-2b left is 25.3 ms of "composite's own machinery"
# and this is the job that says which statements those are.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1

PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_composite_profile.py \
    --configs PointmazeLargeNavigate,PointmazeMediumNavigate \
    --batches 16,64,256 --repeats 7 --device cpu --threads 1 \
    --out "$WS/examples/results/step6/raycast3d/experiments/r6_composite_profile_cpu.json"

echo "R6_PROFILE done"
