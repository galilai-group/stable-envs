#!/bin/bash
#SBATCH --job-name=se-s5-data
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache HF_HOME=$SCR/.cache/huggingface
export HF_HUB_ENABLE_HXET=0
# The dataset cache is on scratch, not in HOME: HOME is over quota and a 40 GiB HDF5 would fail
# silently halfway.
export STABLEWM_HOME=$SCR/.stable_worldmodel
mkdir -p "$STABLEWM_HOME"
PY=$SCR/venvs/se-step4-swm/bin/python
set -x
cd "$SCR/step5"
$PY fetch_dataset.py --cache "$STABLEWM_HOME" --work "$SCR/step5/dl"
echo "DATA rc=$?"
df -h /oscar/scratch/$USER | tail -1
