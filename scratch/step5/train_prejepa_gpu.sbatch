#!/bin/bash
#SBATCH --job-name=se-s5-train
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Train the DINO-WM / PreJEPA Reacher checkpoint with stable_worldmodel's OWN training script.
# Nothing is reimplemented: the config is the repo's PushT DINO-WM config with the dataset and the
# encoding keys changed, and it is passed through hydra's --config-dir so the swm checkout stays
# untouched. spt.Manager resumes from <run_dir>/<name>_weights.ckpt if it is there, which is what
# makes a preemptible condo allocation usable for a multi-hour training run.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
export STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
PY=$SCR/venvs/se-step5/bin/python
EPOCHS=${EPOCHS:-3}
set -x
cd "$SCR/swm-repo"
# Through the shim: the committed script names a callback that no released stable-pretraining
# provides, so it raises before the first batch. See scratch/step5/train_prejepa_shim.py.
$PY "$SCR/step5/train_prejepa_shim.py" --swm "$SCR/swm-repo" -- \
    --config-dir "$SCR/step5/train_config" \
    --config-name prejepa_reacher \
    trainer.max_epochs=$EPOCHS
echo "TRAIN rc=$?"
ls -la "$STABLEWM_HOME/checkpoints/reacher/prejepa" 2>/dev/null
