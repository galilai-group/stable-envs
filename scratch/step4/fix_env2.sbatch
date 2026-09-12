#!/bin/bash
#SBATCH --job-name=se-step4-fix2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# The version was never the problem. `timm` had pulled torchvision 0.29.0 from PyPI -- the right
# version, the wrong BUILD -- and its compiled ops do not register against the cu126 torch these venvs
# carry, so the very first import raises `operator torchvision::nms does not exist`. uv saw a
# satisfying version already present and audited rather than replaced it, which is why the last job
# changed nothing. --reinstall-package names the one wheel to swap: 0.29.0 -> 0.29.0+cu126.
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
source /etc/profile.d/modules.sh 2>/dev/null
module load python/3.12.8-zr3c 2>/dev/null

for V in se-step3 se-s3b-fetch311 se-step4-swm; do
  P="$SCR/venvs/$V/bin/python"
  echo "=== $V ==="
  uv pip install --python "$P" --reinstall-package torchvision \
      --index-url https://download.pytorch.org/whl/cu126 torchvision || echo "torchvision failed in $V"
  uv pip install --python "$P" timm || echo "timm failed in $V"
  [ "$V" = "se-step4-swm" ] && uv pip install --python "$P" stable-worldmodel
  "$P" - <<'PY'
import importlib
for m in ("torch", "torchvision", "timm", "scipy", "stable_worldmodel", "craftax"):
    try:
        mod = importlib.import_module(m); print(f"  ok      {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  MISSING {m}: {type(e).__name__}: {str(e)[:110]}")
PY
done

"$SCR/venvs/se-step3/bin/python" - <<'PY'
import timm
m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
print("dinov2 ok:", sum(p.numel() for p in m.parameters()), "params")
PY
echo "FIX2 DONE"
