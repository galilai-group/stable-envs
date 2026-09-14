#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r6gnr
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The GPU no-regression check, taken the way this workstream learned to take a CPU one.
#
# Job 6366547 ran each leg ONCE and returned 0.53x on AntmazeMediumExplore at B=1024 and 1.39x on
# PointmazeMediumNavigate at the same batch -- two numbers that cannot both be a property of a change
# that is bit-identical on both devices. A single pair per cell cannot tell a cost from the harness,
# so this is INTERLEAVED int/br/int/br on one card in one job, and int2/int1 is a ratio of two runs
# of identical code.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se6/raycast3d
INT=$SCR/se6/integration
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d/experiments/r6cm_gpu_interleaved
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

for E in AntmazeMediumExplore PointmazeMediumNavigate PointmazeLargeNavigate HumanoidmazeMediumNavigate; do
  for LEG in int1 br1 int2 br2; do
    case $LEG in int*) T=$INT;; *) T=$WS;; esac
    PYTHONPATH=$T $PY scratch/step6/raycast3d/profile_phases.py --env "$E" --device cuda \
        --batch 64 256 1024 4096 --repeat 12 --warmup 5 \
        --out "$OUT/$LEG" 2>&1 | tail -6
    echo "GNR $E $LEG rc=$?"
  done
done
echo "R6_GNR done"
