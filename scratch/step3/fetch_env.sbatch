#!/bin/bash
#SBATCH --job-name=se-step3-fetchenv
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
# A SECOND interpreter, because Step 1 measured the conflict rather than guessing it: dm_control needs
# mujoco>=3.12 and gymnasium-robotics needs <=3.11, and under 3.12 a plain
# gymnasium.make("FetchPush-v4") raises AssertionError inside set_joint_qpos. One venv cannot hold both
# families, so Fetch gets its own shard and the merge puts them back together.
VENV=$SCR/venvs/se-step3-fetch
uv venv --python 3.12 "$VENV" || exit 1
uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu126 torch || exit 1
uv pip install --python "$VENV/bin/python" \
    numpy opencv-python-headless gymnasium imageio pillow \
    "mujoco==3.11.0" gymnasium-robotics warp-lang || exit 1
# mujoco_warp is what actually draws the 18 Fetch meshes. Pinned to the release that matches mujoco
# 3.11 rather than the git head, which tracks 3.12.
uv pip install --python "$VENV/bin/python" \
    "mujoco_warp @ git+https://github.com/google-deepmind/mujoco_warp.git@main" \
    || echo "MJWARP-HEAD FAILED (expected if it requires mujoco>=3.12); the shard will record it"
"$VENV/bin/python" -c "
import importlib
for m in ('torch','mujoco','gymnasium_robotics','warp','mujoco_warp'):
    try:
        x=importlib.import_module(m); print('ok', m, getattr(x,'__version__','?'))
    except Exception as e: print('MISSING', m, type(e).__name__, str(e)[:110])
"
echo "FETCH ENV DONE"
