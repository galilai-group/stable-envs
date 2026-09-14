#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6design
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-6's design evidence, re-taken: the per-op attribution inside composite (both bodies, same
# block names, so the columns subtract) and the window every caller actually needs.
#
# The BEFORE column runs against $SCR/se6/integration, so it is the tree the change is measured
# against rather than a reconstruction of it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
PY2D=$SCR/venvs/se6-raycast3d-maze2d/bin/python
EXP=$WS/examples/results/step6/raycast3d/experiments
mkdir -p "$EXP"
set -x
cd "$WS"
lscpu | grep -E "^Model name" | head -1

CFG=PointmazeLargeNavigate,PointmazeMediumNavigate,AntmazeGiantNavigate,HumanoidmazeMediumNavigate

PYTHONPATH=$INT $PY scratch/step6/raycast3d/r6_composite_profile.py --configs "$CFG" \
    --batches 16,64,256 --repeats 7 --device cpu --threads 1 --variant window \
    --out "$EXP/r6_composite_profile_before_cpu.json"
PYTHONPATH=$WS  $PY scratch/step6/raycast3d/r6_composite_profile.py --configs "$CFG" \
    --batches 16,64,256 --repeats 7 --device cpu --threads 1 --variant compact \
    --out "$EXP/r6_composite_profile_after_cpu.json"

OGB="AntmazeGiantNavigate,AntmazeLargeExplore,AntmazeMediumExplore,AntmazeTeleportExplore,HumanoidmazeGiantNavigate,HumanoidmazeLargeNavigate,HumanoidmazeMediumNavigate,PointmazeGiantNavigate,PointmazeLargeNavigate,PointmazeMediumNavigate,PointmazeTeleportNavigate"
PYTHONPATH=$INT $PY scratch/step6/raycast3d/r6_span_probe.py --configs "$OGB" \
    --batches 16,64,256 --device cpu --threads 1 --out "$EXP/r6_span_ogbench_cpu.json"
PYTHONPATH=$INT $PY2D scratch/step6/raycast3d/r6_span_probe.py \
    --configs PointMazeLarge,PointMazeMedium,PointMazeOpen,PointMazeUMaze \
    --batches 16,64 --device cpu --threads 1 --out "$EXP/r6_span_maze2d_cpu.json"
echo "R6_DESIGN done"
