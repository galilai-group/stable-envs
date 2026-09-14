#!/bin/bash
#SBATCH --job-name=se6-raycast3d-r5shardgpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The GPU half of the shards: DMCBallInCupCatch, whose GPU_FAST claim the dashboard cannot currently
# support because there is no Step-3-shaped shard to classify. DMCFingerSpin comes along as the
# same-card control the RAYCAST-2a work was measured against.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TMPDIR=$SCR/tmp; mkdir -p "$TMPDIR"
WS=$SCR/se6/raycast3d
BASE=$SCR/se6/baseline
PY=$SCR/venvs/se-step6/bin/python
OUT=$WS/examples/results/step6/raycast3d/sweep
mkdir -p "$OUT"
set -x
cd "$WS"
nvidia-smi --query-gpu=name,driver_version --format=csv
ENVS="DMCBallInCupCatch,DMCFingerSpin"
PYTHONPATH=$BASE $PY examples/bench_step3.py --device cuda --envs "$ENVS" \
    --reps 25 --max-cell-s 20 --out "$OUT/gpu_3090_before_baseline.json" || echo "BEFORE rc=$?"
PYTHONPATH=$WS   $PY examples/bench_step3.py --device cuda --envs "$ENVS" \
    --reps 25 --max-cell-s 20 --out "$OUT/gpu_3090_after_branch.json" || echo "AFTER rc=$?"
echo "R5_SHARD_GPU done"
