#!/bin/bash
#SBATCH --job-name=se-step4-fix
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# Two gaps the first setup left, both of which would have shown up as a FIDELITY failure rather than
# as the environment problem they are:
#
#   torchvision   `uv pip install timm` pulled the PyPI torchvision, which is built against the
#                 default-index torch and not against the cu126 torch 2.14.0 these venvs carry. It
#                 imports far enough to raise `operator torchvision::nms does not exist` from timm's
#                 first import, and stable_worldmodel (which depends on torchvision) with it. The fix
#                 is the matching wheel from the same index the torch came from.
#   scipy         conformance.pixel_agreement -> interior_mask and conformance.ssim both import
#                 scipy.ndimage. Without it EVERY parity and visual measurement in the Fetch shard
#                 would come back as an error on a renderer that is fine.
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
source /etc/profile.d/modules.sh 2>/dev/null
module load python/3.12.8-zr3c 2>/dev/null

for V in se-step3 se-s3b-fetch311 se-step4-swm; do
  P="$SCR/venvs/$V/bin/python"
  echo "=== $V: torch before ==="; "$P" -c "import torch;print(torch.__version__)"
  uv pip install --python "$P" --index-url https://download.pytorch.org/whl/cu126 torchvision \
      || echo "torchvision failed in $V"
  uv pip install --python "$P" scipy || echo "scipy failed in $V"
  echo "--- after ---"
  "$P" - <<'PY'
import importlib
for m in ("torch", "torchvision", "timm", "scipy", "stable_worldmodel", "craftax"):
    try:
        mod = importlib.import_module(m); print(f"  ok      {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  MISSING {m}: {type(e).__name__}: {str(e)[:110]}")
PY
done

# The checkpoint, once, into the shared cache. A compute node that cannot reach the hub would
# otherwise fail three separate jobs with a network error dressed as a missing model.
"$SCR/venvs/se-step3/bin/python" - <<'PY'
import timm
m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
print("dinov2 ok:", sum(p.numel() for p in m.parameters()), "params")
PY
echo "FIX DONE"
