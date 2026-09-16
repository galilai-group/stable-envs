#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-subbatch
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Localise the closure sweep's sub-batch and empty-batch findings before fixing them.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_subbatch_diag.py \
  --out "$WS/examples/results/step7/ogb-manipulation/m6" --cells OGBCube OGBScene || echo "DIAG rc=$?"
[ -s "$WS/examples/results/step7/ogb-manipulation/m6/subbatch_diag.json" ] || { echo MISSING; exit 4; }
echo OK
