#!/bin/bash
#SBATCH --job-name=se-s5-fixtorch
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# The editable install of stable-worldmodel pulled torch from PyPI, whose default build is CUDA 13;
# this cluster's driver is 12.9, so torch.cuda.init() fails outright. Put back the cu126 wheels the
# rest of this project runs on. Same class of failure as Step 4's torchvision: uv considers a
# satisfying VERSION installed and will not swap the BUILD without --reinstall-package.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
VENV=$SCR/venvs/se-step5
set -x
uv pip install --python "$VENV/bin/python" \
   --reinstall-package torch --reinstall-package torchvision \
   --index-url https://download.pytorch.org/whl/cu126 \
   torch==2.14.0 torchvision==0.29.0
cd "$SCR/step5"
"$VENV/bin/python" -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
echo "FIX rc=$?"
