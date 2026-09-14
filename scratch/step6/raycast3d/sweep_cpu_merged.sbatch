#!/bin/bash
#SBATCH --job-name=se6-raycast3d-cpusweep
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# RAYCAST-2b on the device it targets: the CPU.
#
# The frozen Step-4 oracle is CUDA-ONLY -- every digest in it was taken on a 3090 -- so the CPU path
# has no frozen contract to replay against. What it does have is two properties that can be checked
# here and are the ones that matter:
#
#   1. BRANCH == BASELINE on the same frozen states, on this node, in this job. That is the
#      regression check, and it is the strongest statement available without a CPU oracle.
#   2. COVER_COMPACTION on == off. The compaction is admissible only because it is the identity, and
#      the CPU is where it changes the most work, so it is asserted here as well as on the GPU.
#
# Plus the two OUT-OF-CORPUS callers of R3.composite (maze2d's PointMaze*, mujoco_locomotion's
# Mujoco*) which have no frozen states at all and are held to before-vs-after on a seeded draw.
#
# CPU certification budget is ONE torch thread; --cpus-per-task=4 only gives the job room.
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
D=$SCR/se6/_dumps/2bcpu
set -x
cd "$WS"
mkdir -p "$OUT/profile_after_2b"

OGB="AntmazeGiantNavigate AntmazeLargeExplore AntmazeMediumExplore AntmazeTeleportExplore \
HumanoidmazeGiantNavigate HumanoidmazeLargeNavigate HumanoidmazeMediumNavigate \
PointmazeGiantNavigate PointmazeLargeNavigate PointmazeMediumNavigate PointmazeTeleportNavigate"
OOB="PointMazeLarge PointMazeMedium PointMazeOpen PointMazeUMaze \
MujocoReacher MujocoPusher MujocoInvertedPendulum MujocoInvertedDoublePendulum"

echo "######## the 1-thread CPU certification sweep on the MERGED code, before and after"
for E in PointmazeLargeNavigate PointmazeMediumNavigate AntmazeMediumExplore HumanoidmazeMediumNavigate; do
  PYTHONPATH=$BASE $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2b/before_baseline" || echo "SB $E rc=$?"
  PYTHONPATH=$WS   $PY scratch/step6/raycast3d/profile_phases.py --env $E --device cpu --threads 1 \
      --batch 1 16 64 256 --repeat 5 --warmup 2 \
      --states-dir "$WS/examples/results/step4/states" \
      --out "$OUT/profile_after_2b/after" || echo "SA $E rc=$?"
done
echo "CPU_SWEEP done"
