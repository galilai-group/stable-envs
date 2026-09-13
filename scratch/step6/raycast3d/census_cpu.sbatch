#!/bin/bash
#SBATCH --job-name=se6-raycast3d-census
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 6C ticket 1, deliverable 1: the LIVE raycast3d replay set, observed rather than read off a
# declaration. certification.json's `backends` is a module-level field and dmcontrol.py serves two
# renderer families, so this renders every declared configuration once with the raycast3d entry
# points wrapped and records which of them actually ran.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
export PYTHONPATH=$WS
PY=$SCR/venvs/se-step6/bin/python
set -x
cd "$WS"
$PY scratch/step6/raycast3d/backend_census.py --device cpu --batch 2 \
    --out "$WS/examples/results/step6/raycast3d/backend_census_cpu.json"
echo "CENSUS done"
