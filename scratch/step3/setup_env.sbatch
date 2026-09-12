#!/bin/bash
#SBATCH --job-name=se-step3-setup
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail

# HOME is at 100.88 GB against a 100 GB soft limit (checkquota, SOFT_EXCEEDED, 10-day grace), so every
# cache this job could fill has to be pointed at scratch BEFORE the first install. uv, pip and torch
# all default under $HOME and a multi-GB CUDA wheel set would push HOME past its hard limit mid-build.
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache
export PIP_CACHE_DIR=$SCR/.pip-cache
export TORCH_HOME=$SCR/.torch
export XDG_CACHE_HOME=$SCR/.cache
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$SCR/logs" "$SCR/venvs"

source /etc/profile.d/modules.sh 2>/dev/null
module load python/3.12.8-zr3c 2>/dev/null

VENV=$SCR/venvs/se-step3
uv venv --python 3.12 "$VENV" || exit 1
export VIRTUAL_ENV=$VENV

# The renderer stack. cu126 wheels rather than the default index: the node driver has to be at least
# as new as the CUDA runtime the wheel was built against, and the newest build is the usual way a
# torch install that "succeeded" fails at the first kernel launch.
uv pip install --python "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cu126 torch || exit 1

# Everything else from PyPI. These are the REFERENCE packages -- the renderer itself needs none of
# them -- so a failure here costs the official-renderer baseline for one family, not the family.
uv pip install --python "$VENV/bin/python" \
    numpy "opencv-python-headless" gymnasium imageio pillow \
    gym-pusht pymunk pygame \
    "mujoco>=3.12" "dm-control>=1.0.44" ogbench \
    || echo "PARTIAL: some reference packages did not install; the sweep records that per env"

"$VENV/bin/python" - <<'PY'
import torch, platform
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.version.cuda)
for m in ("numpy","cv2","gymnasium","mujoco","dm_control","ogbench","gym_pusht","pymunk","pygame","imageio"):
    try:
        __import__(m); print(f"  ok      {m}")
    except Exception as e:
        print(f"  MISSING {m}: {type(e).__name__}: {str(e)[:90]}")
PY
echo "SETUP DONE"
