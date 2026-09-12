#!/bin/bash
#SBATCH --job-name=se-s5-qualify
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Two things that do not need a trained checkpoint.
#
# 1. The corpus inspection, with enough memory this time -- the first attempt was OOM-killed at
#    24 GB while indexing a 92 GiB HDF5.
#
# 2. The harness end to end on the RANDOM policy. It exercises everything the gate depends on --
#    config composition, the three-surface swap, the pairing, the per-episode capture -- and it
#    produces the number the trained arms have to beat. A world model that does not beat random
#    action selection is not an instrument, and knowing the floor costs one short allocation.
#
# The published LeWM Reacher checkpoint was the original plan here and cannot be loaded: it was
# saved under transformers 4.x, whose ViT parameters are named encoder.encoder.layer.N.attention.
# attention.query.*, and transformers 5.x renamed them to encoder.layers.N.attention.q_proj.*.
# Recorded as a finding rather than worked around by rewriting somebody's weights.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-random}
SEED=${SEED:-42}
N=${N:-50}
TAG=${TAG:-random}
OUT=${OUT:-$SCR/step5/out/reacher/runs_$TAG}
set -x

cd "$SCR/step5"
$PY inspect_dataset.py --dataset dmc/reacher_random.h5 --out "$SCR/step5/out/reacher"
echo "INSPECT rc=$?"

cd "$SCR/stable-envs"
for ARM in official stable_envs blank; do
  $PY examples/downstream_step5/reacher/eval_reacher.py --arm $ARM --seed $SEED --n $N \
      --policy "$POLICY" --swm "$SCR/swm-repo" --target-mode offscreen --out "$OUT"
  echo "ARM $ARM rc=$?"
done
ls -la "$OUT" 2>/dev/null
