#!/bin/bash
#SBATCH --job-name=se-step3-warp
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
VENV=$SCR/venvs/se-step3
# mujoco_warp is what the DM Control mjwarp path and the whole Fetch family render through; without it
# those envs are a dependency blocker rather than a measurement. It needs mujoco>=3.11, which this venv
# already satisfies -- the conflict Step 1 found is with gymnasium-robotics, not with dm_control.
uv pip install --python "$VENV/bin/python" warp-lang || echo "WARP-LANG FAILED"
uv pip install --python "$VENV/bin/python" \
    "mujoco_warp @ git+https://github.com/google-deepmind/mujoco_warp.git" || echo "MJWARP FAILED"
"$VENV/bin/python" -c "
for m in ('warp','mujoco_warp'):
    try:
        mod=__import__(m); print('ok', m, getattr(mod,'__version__','?'))
    except Exception as e: print('MISSING', m, type(e).__name__, str(e)[:120])
"
echo "WARP SETUP DONE"
