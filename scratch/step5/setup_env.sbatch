#!/bin/bash
#SBATCH --job-name=se-s5-setup
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Build the Step 5 interpreter: stable_worldmodel and its training stack, dm_control pinned to the
# mujoco it works with, and stable-envs on PYTHONPATH exactly as Step 4 ran it.
set -uo pipefail
export SCR=/oscar/scratch/$USER
# HOME is over quota on this account, so every cache that would otherwise land there is pinned to
# scratch. A pip cache in a full HOME fails halfway through an install and leaves a broken venv.
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache
export XDG_CACHE_HOME=$SCR/.cache TORCH_HOME=$SCR/.torch HF_HOME=$SCR/.cache/huggingface
export UV_PYTHON_PREFERENCE=only-managed
VENV=$SCR/venvs/se-step5
set -x

uv venv --python 3.12 "$VENV" || exit 1
export VIRTUAL_ENV=$VENV

# The exact torch Steps 3 and 4 measured on. A different CUDA wheel would make this a different
# machine wearing the same label.
uv pip install --python "$VENV/bin/python" \
   torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126 || exit 1

# stable_worldmodel itself, editable from the synced checkout so the git commit is knowable, plus
# the train extra (stable-pretraining, hydra, transformers) and the format extra (h5py, lance).
# The `env` extra is deliberately NOT installed: it pulls craftax/jax/ogbench/box2d for environments
# this arm never builds, and every one of them is a way for the install to fail on something the
# experiment does not use.
uv pip install --python "$VENV/bin/python" -e "$SCR/swm-repo[train,format]" || exit 1

# dm_control 1.0.45 with mujoco 3.12.0 is the pair se-step3 runs; dm_control's named-index layer is
# built from a generated table of MjModel fields and breaks silently across a mujoco minor.
uv pip install --python "$VENV/bin/python" \
   "mujoco==3.12.0" "dm_control==1.0.45" "mujoco-warp==3.12.0" "warp-lang==1.17.0" \
   scikit-learn scipy zstandard "huggingface-hub" opencv-python || exit 1

cd "$SCR/step5"
PYTHONPATH=$SCR/stable-envs MUJOCO_GL=egl "$VENV/bin/python" - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch, numpy, sklearn, hydra, h5py
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import stable_worldmodel as swm
print("swm", swm.__file__)
import stable_pretraining as spt
print("spt ok")
import dm_control, mujoco
print("dm_control", dm_control.__version__ if hasattr(dm_control, "__version__") else "?", "mujoco", mujoco.__version__)
import stable_envs as se
import stable_envs.envs as E
errs = E.import_errors()
print("stable_envs families unavailable:", sorted(errs))
e = se.make("DMCReacherHard", batch=2, size=224, device="cuda" if torch.cuda.is_available() else "cpu")
print("DMCReacherHard state_dim", e.state_dim)
PY
echo "SETUP rc=$?"
