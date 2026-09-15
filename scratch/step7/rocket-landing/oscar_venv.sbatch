#!/bin/bash
#SBATCH --job-name=se7-rocket-venv
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
# Step 7 / W6: the interpreter for PFRocketLanding on Oscar -- the reference (PyFlyt + PyBullet + cvxpy +
# stable-worldmodel), our renderer (torch cu126 + nvdiffrast from GitHub; its CUDA plugin JIT-compiles on
# first import ON A GPU NODE with nvcc, so the smoke test below only checks the Python side), and the six
# locomotion visual cells' reference (gymnasium[mujoco], mujoco 3.12.0) for the nvdiff byte-identity job.
# Same pins as se-step6 where they overlap; a NEW venv because se-step6 is in use by other agents.
# HOME is at quota: every cache lives under $SCR.
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.cache/uv PIP_CACHE_DIR=$SCR/.cache/pip XDG_CACHE_HOME=$SCR/.cache
export NUMBA_CACHE_DIR=$SCR/.cache/numba TORCH_EXTENSIONS_DIR=$SCR/.cache/torch_extensions
mkdir -p "$SCR/logs" "$NUMBA_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
VENV=$SCR/venvs/se7-rocket
echo "NODE $(hostname -s) JOB $SLURM_JOB_ID"
set -x
rm -rf "$VENV"
uv venv --python 3.12 "$VENV" || exit 1
uv pip install --python "$VENV/bin/python" \
   --index-url https://download.pytorch.org/whl/cu126 \
   --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
   "torch==2.14.0+cu126" \
   "mujoco==3.12.0" "gymnasium[mujoco]" numpy scipy Pillow imageio pytest ninja \
   PyFlyt pybullet cvxpy stable-worldmodel opencv-python-headless trimesh || exit 1
# nvdiffrast is not on PyPI; the locomotion visual tier was measured with 0.4.0
uv pip install --python "$VENV/bin/python" "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0" \
  || uv pip install --python "$VENV/bin/python" "git+https://github.com/NVlabs/nvdiffrast.git" || exit 1
cd "$SCR/se7/rocket-landing"
PYTHONPATH=$SCR/se7/rocket-landing MUJOCO_GL=egl "$VENV/bin/python" - <<'PY'
import importlib.metadata as md, torch
for k in ("torch", "mujoco", "gymnasium", "PyFlyt", "pybullet", "cvxpy", "stable-worldmodel", "nvdiffrast", "numba", "scipy"):
    try:
        print(f"{k:20s} {md.version(k)}")
    except Exception as e:
        print(f"{k:20s} MISSING {e}")
import nvdiffrast  # the python package; the CUDA plugin compiles on the GPU job
import stable_envs as se, stable_envs.envs as E
print("rocket_landing import error:", E.import_errors().get("rocket_landing", "none"))
from stable_envs.envs import rocket_landing as RL
print("missing_backend on a CPU node (expected: nvdiffrast RasterizeCudaContext needs CUDA at import? no -> None):", RL.missing_backend())
v, n, f, vg, uv = RL.build_scene_arrays(); print("scene", len(v), "verts", len(f), "tris")
import gymnasium, stable_worldmodel  # noqa
print("swm registered:", "swm/PFRocketLanding-v0" in gymnasium.registry)
PY
echo "S7VENV rc=$?"
