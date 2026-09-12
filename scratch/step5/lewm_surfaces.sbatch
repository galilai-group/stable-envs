#!/bin/bash
#SBATCH --job-name=se-s5-surf
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Localise the LeWM Reacher drop: 52% official against 22% with every pixel surface swapped.
#
# Three surfaces feed the model an image -- the live observation at every step, the stored corpus
# frame at the start step, and the stored goal frame. Swapping them separately says WHICH one costs
# the points, and the two answers mean opposite things. Live-only is a statement about our
# renderer. Dataset-only is a statement about the corpus: the model was trained on stored frames,
# and if those are not what our renderer produces from the same state, replacing them moves the
# model off its training distribution without either renderer being wrong.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTHONPATH=$SCR/stable-envs
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
POLICY=${POLICY:-reacher/lewm_tf5/weights.pt}
OUT=$SCR/step5/out/reacher/runs_lewm
set -x
cd "$SCR/stable-envs"
for SFC in live dataset; do
  $PY examples/downstream_step5/reacher/eval_reacher.py --arm stable_envs --seed 42 --n 50 \
      --policy "$POLICY" --swm "$SCR/swm-repo" --target-mode offscreen --surfaces $SFC --out "$OUT"
  echo "SURFACE $SFC rc=$?"
done
ls -la "$OUT"
