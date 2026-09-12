#!/bin/bash
#SBATCH --job-name=se-s5-inspect
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# What resolution and dtype are the corpus frames, and do they show the target? The two earlier
# attempts were OOM-killed asking get_dim() for the width of a 92 GiB pixels column; that call is
# gone.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache HF_HOME=$SCR/.cache/huggingface
export STABLEWM_HOME=$SCR/.stable_worldmodel
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
set -x
cd "$SCR/step5"
$SCR/venvs/se-step5/bin/python inspect_dataset.py --dataset dmc/reacher_random.h5 \
    --out "$SCR/step5/out/reacher" --frames 8
echo "INSPECT rc=$?"
