#!/bin/bash
#SBATCH --job-name=se7-ogb-manipulation-mjwarp
#SBATCH --partition=gpu
#SBATCH --qos=norm-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce3090
#SBATCH --exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W5 phase 0, cluster half 2: can the OGBench manipulation scene go through MJWarp
# the way `stable_envs/envs/fetch.py` puts the Fetch scene through it, and at what batch scaling?
#
# THE TWO QUESTIONS, both of which the local census can only bound.
#
# 1. DOES IT BUILD.  Fetch's mover pass is 18 meshes / 31,720 triangles. These scenes are 28 and 42
#    meshes / 193,113 and 196,633 VISIBLE triangles -- 6.1x Fetch -- so the BVH build, the device
#    memory and the per-ray traversal depth are all outside the range any Step-6 number covers.
#
# 2. BATCH_SCALES.  Step 6 closed `fetch-gpu` as a demonstrated blocker on
#    BATCH_SCALES = fps(4096)/fps(1) = 7.6-11.7x against a required 20x, and recorded that the SAME
#    scene PASSES at <=240 px (26-99x) and fails only at the frozen 480 px. These two envs render at
#    224 px and 200 px -- inside the passing regime -- so the Fetch verdict does NOT transfer and the
#    ratio has to be measured. This job measures fps at B = 1, 16, 64, 256, 1024, 4096 on ONE card.
#
# A 3090 keeps this comparable with the Step-3/4/6 GPU rows. One GPU job at a time per agent.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export XDG_CACHE_HOME=$SCR/.cache PIP_CACHE_DIR=$SCR/.cache/pip
export TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface WARP_CACHE_PATH=$SCR/.cache/warp
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 MUJOCO_GL=egl
WS=$SCR/se7/ogb-manipulation
VENV=$SCR/venvs/se7-ogb
OUT=$WS/examples/results/step7/ogb-manipulation/oscar/mjwarp
mkdir -p "$OUT"
export PYTHONPATH=$WS:$SCR/se7/swm-src
set -x
cd "$WS"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
# EGL and nothing else. This runs on a GPU node, where EGL is the correct backend; if it fails
# here it is a NODE FAULT (W4 found exactly that on gpu2104, torch/CUDA healthy, EGL broken) and
# must not be papered over with a software fallback that would silently change the reference.
eval "$(PYTHONPATH=$PYTHONPATH "$VENV/bin/python" \
  scratch/step7/ogb-manipulation/gl_probe.py --require egl 2>&1 | tee /dev/stderr | tail -1)"
[ "$MUJOCO_GL" = "__NONE__" ] && { echo "FATAL: EGL unusable on $(hostname) -- node fault"; exit 9; }

"$VENV/bin/python" scratch/step7/ogb-manipulation/probe_mjwarp_trial.py --out "$OUT" \
  --batches 1 16 64 256 1024 || echo "MJWARP rc=$?"
ls -l "$OUT" | tail -10
