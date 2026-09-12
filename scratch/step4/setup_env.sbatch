#!/bin/bash
#SBATCH --job-name=se-step4-setup
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# Step 3's caches, unchanged and for the same reason: HOME is SOFT_EXCEEDED at 101/100 GB, and every
# tool here defaults its cache under it. HF_HOME is new -- the DINOv2 tripwire pulls a ViT-B/14
# checkpoint, which is 350 MB in a directory that would otherwise be $HOME/.cache/huggingface.
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export WARP_CACHE_PATH=$SCR/.cache/warp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$HF_HOME" "$SCR/logs"

source /etc/profile.d/modules.sh 2>/dev/null
module load python/3.12.8-zr3c 2>/dev/null

# 1. The shard that Step 3 never needed: craftax's atlas and stable_worldmodel's own nav envs are the
#    OFFICIAL renderers for six Step 4 configurations, and neither is in any existing venv. Its own
#    interpreter because craftax pulls the jax stack, which is what the Overcooked carve-out in
#    stable_envs/testing/gate.py is about.
VENV=$SCR/venvs/se-step4-swm
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV" || exit 1
  uv pip install --python "$VENV/bin/python" \
      --index-url https://download.pytorch.org/whl/cu126 torch || exit 1
  uv pip install --python "$VENV/bin/python" \
      numpy "opencv-python-headless" gymnasium imageio pillow scipy scikit-image \
      craftax stable-worldmodel \
      || echo "PARTIAL: some reference packages did not install; the shard records that per config"
fi

# 2. The tripwire needs an encoder in every interpreter that holds a reference renderer.
for V in se-step3 se-s3b-fetch311 se-step4-swm; do
  uv pip install --python "$SCR/venvs/$V/bin/python" timm || echo "timm failed in $V"
done

# 3. Pull the checkpoint ONCE, here, into the shared cache. A compute node that cannot reach the hub
#    would otherwise fail three separate jobs with a network error dressed as a missing model.
"$SCR/venvs/se-step3/bin/python" - <<'PY'
import timm
m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
print("dinov2 ok:", sum(p.numel() for p in m.parameters()), "params")
PY

for V in se-step3 se-s3b-fetch311 se-step4-swm; do
  echo "=== $V ==="
  "$SCR/venvs/$V/bin/python" - <<'PY'
import importlib
for m in ("torch","numpy","timm","craftax","stable_worldmodel","dm_control","ogbench",
          "gym_pusht","gymnasium_robotics","mujoco","mujoco_warp","skimage"):
    try:
        mod = importlib.import_module(m); print(f"  ok      {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  MISSING {m}: {type(e).__name__}")
PY
done
echo "SETUP DONE"
