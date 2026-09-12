#!/bin/bash
#SBATCH --job-name=se-s5-remap
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Rename the LeWM Reacher checkpoint's ViT keys into the transformers 5.x layout. CPU only: this
# loads 72 MB, renames dictionary keys and writes them back. The guards are in the script; the
# real verification is the qualification run afterwards, against a recorded 86%.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export STABLEWM_HOME=$SCR/.stable_worldmodel
export PYTHONUNBUFFERED=1
set -x
cd "$SCR/step5"
$SCR/venvs/se-step5/bin/python remap_lewm.py --policy quentinll/lewm-reacher --out reacher/lewm_tf5
echo "REMAP rc=$?"
