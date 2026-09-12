#!/bin/bash
#SBATCH --job-name=se-s5-setup2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Rebuild the Step 5 interpreter in ONE resolution.
#
# The first attempt installed torch from the cu126 wheel index and then installed
# stable-worldmodel, whose `torch` requirement uv satisfied from PyPI -- whose default build is
# CUDA 13. That silently replaced the whole stack, and this cluster's driver is 12.9, so
# torch.cuda.init() failed outright. Reinstalling torch alone afterwards was worse: it left the
# cu13 NCCL in place and libtorch_cuda.so came up with an undefined ncclCommGrow.
#
# The fix is not a repair, it is an ordering: pin the LOCAL version (+cu126, which exists only on
# the pytorch index) and resolve everything together, so there is no second pass to disagree with
# the first.
#
# The `format` extra is not installed either. It pulls lerobot, which caps torchvision below 0.21
# and makes the whole resolution unsatisfiable against the torch this project runs on. What that
# extra is actually needed for here is h5py, so h5py is what is installed; the LeRobot Hub adapter
# reads datasets this arm does not use.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
VENV=$SCR/venvs/se-step5
set -x

rm -rf "$VENV"
uv venv --python 3.12 "$VENV" || exit 1

uv pip install --python "$VENV/bin/python" \
   --index-url https://download.pytorch.org/whl/cu126 \
   --extra-index-url https://pypi.org/simple \
   --index-strategy unsafe-best-match \
   "torch==2.14.0+cu126" "torchvision==0.29.0+cu126" \
   -e "$SCR/swm-repo[train]" \
   h5py hdf5plugin "imageio[ffmpeg]" \
   "mujoco==3.12.0" "dm_control==1.0.45" "mujoco-warp==3.12.0" "warp-lang==1.17.0" \
   scikit-learn scipy zstandard huggingface-hub opencv-python || exit 1

cd "$SCR/step5"
"$VENV/bin/python" - <<'PY'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
import stable_worldmodel as swm, stable_pretraining, hydra, h5py, sklearn, mujoco, dm_control
print("swm", swm.__file__, "mujoco", mujoco.__version__)
PY
echo "SETUP2 rc=$?"
uv pip list --python "$VENV/bin/python" | grep -i -E "^(torch|nvidia)" | head -30
