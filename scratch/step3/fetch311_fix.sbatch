#!/bin/bash
#SBATCH --job-name=se-s3b-fix
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
P=$SCR/venvs/se-s3b-fetch311/bin/python
# gymnasium-robotics went in with --no-deps to stop it dragging mujoco forward; that also skipped its
# harmless pure-python deps. Add those explicitly, naming each one, so nothing can pull mujoco again.
uv pip install --python "$P" packaging cloudpickle typing_extensions farama-notifications || exit 1
"$P" -c "import importlib.metadata as md; print('mujoco', md.version('mujoco'), '| mujoco-warp', md.version('mujoco-warp'))"
"$P" - <<'PY'
import traceback
res = {}
try:
    import mujoco; res['mujoco'] = mujoco.__version__
except Exception as e: res['mujoco'] = f'FAIL {e}'
try:
    import mujoco_warp; res['mujoco_warp'] = 'imports'
except Exception as e: res['mujoco_warp'] = f'FAIL {type(e).__name__}: {str(e)[:110]}'
try:
    import gymnasium as gym, gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    e = gym.make('FetchPush-v4'); e.reset(seed=0)
    res['FetchPush-v4'] = 'reset OK'
except Exception as ex:
    res['FetchPush-v4'] = f'FAIL {type(ex).__name__}: {str(ex)[:110]}'; traceback.print_exc()
for k, v in res.items(): print(f'  {k:16s} {v}')
print('MATCHED STACK WORKS' if not any(str(v).startswith('FAIL') for v in res.values())
      else 'MATCHED STACK FAILS')
PY
echo "S3B FIX DONE"
