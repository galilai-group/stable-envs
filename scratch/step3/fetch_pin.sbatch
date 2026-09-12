#!/bin/bash
#SBATCH --job-name=se-step3-fetchpin
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
V=$SCR/venvs/se-step3-fetch
# The pin has to be applied LAST and with --no-deps. Installing "mujoco==3.11.0" alongside
# mujoco_warp let the resolver satisfy mujoco_warp's own floor by pulling mujoco 3.13 back over the
# top, which is precisely the version that makes gymnasium-robotics raise a bare AssertionError inside
# set_joint_qpos -- the blocker Step 1 documented and this job is trying to get under.
uv pip install --python "$V/bin/python" --no-deps --reinstall "mujoco==3.11.0" || exit 1
"$V/bin/python" -c "
import mujoco; print('mujoco', mujoco.__version__)
import gymnasium as gym, gymnasium_robotics  # noqa
e = gym.make('FetchPush-v4'); e.reset(seed=0); print('FetchPush-v4 reset OK under', mujoco.__version__)
try:
    import mujoco_warp; print('mujoco_warp still imports:', mujoco_warp.__version__)
except Exception as ex: print('mujoco_warp BROKEN by the downgrade:', type(ex).__name__, str(ex)[:150])
"
echo "FETCH PIN DONE"
