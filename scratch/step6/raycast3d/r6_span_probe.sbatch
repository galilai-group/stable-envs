#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6span
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-6 design step: the window every composite caller actually needs, against the 32 it asks for.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WS=$SCR/se6/raycast3d
PY=$SCR/venvs/se-step6/bin/python
PY2D=$SCR/venvs/se6-raycast3d-maze2d/bin/python
EXP=$WS/examples/results/step6/raycast3d/experiments
set -x
cd "$WS"

OGB="AntmazeGiantNavigate,AntmazeLargeExplore,AntmazeMediumExplore,AntmazeTeleportExplore,HumanoidmazeGiantNavigate,HumanoidmazeLargeNavigate,HumanoidmazeMediumNavigate,PointmazeGiantNavigate,PointmazeLargeNavigate,PointmazeMediumNavigate,PointmazeTeleportNavigate"
PYTHONPATH=$WS $PY scratch/step6/raycast3d/r6_span_probe.py --configs "$OGB" \
    --batches 16,64,256 --device cpu --threads 1 --out "$EXP/r6_span_ogbench_cpu.json"

PYTHONPATH=$WS $PY2D scratch/step6/raycast3d/r6_span_probe.py \
    --configs PointMazeLarge,PointMazeMedium,PointMazeOpen,PointMazeUMaze \
    --batches 16,64 --device cpu --threads 1 --out "$EXP/r6_span_maze2d_cpu.json"

echo "R6_SPAN done"
