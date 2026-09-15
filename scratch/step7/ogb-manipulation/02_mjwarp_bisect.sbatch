#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-bisect
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W5 phase 0: WHY does OGBScene crash MJWarp?
#
# Jobs 6395461 and 6395736 both died with `Warp CUDA error 700: an illegal memory access`, the
# second in a fresh process with the error surfacing at `wp_cuda_context_synchronize` -- during the
# warm-up frame, before the graph capture -- so it is OGBScene's own fault, not contamination.
# BatchRenderer exposes the megakernel's switches, so bisect rather than guess. Each configuration
# runs in its own subprocess because an illegal access poisons the context for everything after it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/oscar/mjwarp
mkdir -p "$OUT"
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname) -- node fault"; exit 9; }
"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_mjwarp_bisect.py --out "$OUT" \
  --env swm/OGBScene-v0 --ws "$WS" || echo "BISECT rc=$?"
[ -s "$OUT/mjwarp_bisect_OGBScene.json" ] || { echo "MISSING bisect json"; exit 4; }
echo OK
