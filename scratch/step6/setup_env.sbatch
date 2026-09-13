#!/bin/bash
#SBATCH --job-name=se-s6-env
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# A SEPARATE interpreter for Step 6 CPU profiling.
#
# Five envs failed the first profiling pass for missing pygame (classic control's reference
# renderer) and ogbench (the maze reference). Both are cheap CPU-only wheels -- but se-step5 is the
# venv twelve running Step 5 jobs are importing from, and pip transiently replacing files under a
# live job is how ten hours of campaign gets lost. So: new venv, same pins.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
VENV=$SCR/venvs/se-step6
set -x
rm -rf "$VENV"
uv venv --python 3.12 "$VENV" || exit 1
uv pip install --python "$VENV/bin/python" \
   --index-url https://download.pytorch.org/whl/cu126 \
   --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
   "torch==2.14.0+cu126" "torchvision==0.29.0+cu126" \
   "mujoco==3.12.0" "dm_control==1.0.45" \
   numpy scipy scikit-learn h5py imageio pygame pymunk shapely ogbench opencv-python || exit 1
cd "$SCR/stable-envs"
PYTHONPATH=$SCR/stable-envs MUJOCO_GL=egl "$VENV/bin/python" - <<'PY'
import stable_envs as se, stable_envs.envs as E, torch, mujoco
print("torch", torch.__version__, "mujoco", mujoco.__version__)
print("unavailable families:", sorted(E.import_errors()))
for name in ("MountainCar", "Pendulum", "PointmazeLargeNavigate", "PushT"):
    try:
        e = se.make(name, batch=1, size=64, device="cpu")
        print(f"  {name:24s} ok  state_dim={e.state_dim}")
    except Exception as ex:
        print(f"  {name:24s} FAIL {type(ex).__name__}: {str(ex)[:90]}")
PY
echo "S6ENV rc=$?"
