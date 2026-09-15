#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-m3corr
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# M3 design gate: does re-typing the out-of-range spot to POINT make MJWarp agree with MuJoCo?
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/oscar/m3
mkdir -p "$OUT"
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name --format=csv || true
eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname)"; exit 9; }
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_m3_corrections.py --out "$OUT" \
  --env swm/OGBCube-v0 --states 8 || echo "M3 rc=$?"
[ -s "$OUT/m3_corrections.json" ] || { echo "MISSING m3_corrections.json"; exit 4; }
echo OK
