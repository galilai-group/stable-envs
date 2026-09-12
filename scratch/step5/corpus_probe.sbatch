#!/bin/bash
#SBATCH --job-name=se-s5-corpus
#SBATCH --partition=3090-gcondo
#SBATCH --qos=cs-3090-gcondo
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp STABLEWM_HOME=$SCR/.stable_worldmodel
export MUJOCO_GL=egl MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONPATH=$SCR/stable-envs
set -x
cd "$SCR/stable-envs"
$SCR/venvs/se-step5/bin/python examples/downstream_step5/reacher/corpus_probe.py \
    --n 32 --out "$SCR/step5/out/reacher"
echo "CORPUS rc=$?"
