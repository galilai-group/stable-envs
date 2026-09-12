#!/bin/bash
#SBATCH --job-name=se-s5-tf4
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# CONTINGENCY, not part of the plan. Build a second interpreter pinned to transformers 4.x so the
# published LeWM Reacher checkpoint can be loaded.
#
# Only needed if the PreJEPA checkpoint fails to qualify: the LeWM weights are a competent,
# externally recorded instrument (86% in swm's docs/baselines.md under this same 50-step budget),
# but using them would cost the second world-model family section 14 asks for on this environment,
# so they are a fallback rather than an addition.
#
# The pin is the whole point. transformers 5 renamed ViT parameters from
# encoder.encoder.layer.N.attention.attention.query.* to encoder.layers.N.attention.q_proj.*, and
# these weights were saved before that. stable-pretraining 0.1.8 predates the rename too, so 4.x
# should suit it better rather than worse.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
VENV=$SCR/venvs/se-step5-tf4
set -x
rm -rf "$VENV"
uv venv --python 3.12 "$VENV" || exit 1
uv pip install --python "$VENV/bin/python" \
   --index-url https://download.pytorch.org/whl/cu126 \
   --extra-index-url https://pypi.org/simple \
   --index-strategy unsafe-best-match \
   "torch==2.14.0+cu126" "torchvision==0.29.0+cu126" \
   "transformers<5" \
   -e "$SCR/swm-repo[train]" \
   h5py hdf5plugin \
   "mujoco==3.12.0" "dm_control==1.0.45" "mujoco-warp==3.12.0" "warp-lang==1.17.0" \
   scikit-learn scipy huggingface-hub opencv-python || exit 1
cd "$SCR/step5"
"$VENV/bin/python" - <<'PY'
import torch, transformers, stable_worldmodel as swm
print("torch", torch.__version__, "transformers", transformers.__version__)
m = swm.wm.utils.load_pretrained("quentinll/lewm-reacher")
print("LeWM reacher loaded:", type(m).__name__,
      sum(p.numel() for p in m.parameters()), "params")
PY
echo "TF4 rc=$?"
