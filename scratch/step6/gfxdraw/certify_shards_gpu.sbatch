#!/bin/bash
#SBATCH --job-name=se6-gfxdraw-shardgpu
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# GPU certification shards with a paired denominator, both arms, one card, one job.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export TRITON_CACHE_DIR=$SCR/.cache/triton
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
PY=$SCR/venvs/se-step6/bin/python
OUT=$SCR/se6/gfxdraw/examples/results/step6/gfxdraw/sweep2
NODE=$(hostname -s)
mkdir -p "$OUT"
cd "$SCR/se6/gfxdraw"
echo "NODE $NODE JOB $SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
set -x
for ARM in before after; do
  PYTHONPATH=$SCR/se6/gfxdraw $PY scratch/step6/gfxdraw/certify_shard.py \
      --device cuda --arm $ARM --max-cell-s 30 \
      --out "$OUT/gpu_paired_${ARM}_${NODE}.json"
  echo "GPU SHARD $ARM rc=$?"
done
echo "SHARD GPU JOB DONE"
