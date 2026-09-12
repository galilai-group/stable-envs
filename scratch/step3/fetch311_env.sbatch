#!/bin/bash
#SBATCH --job-name=se-s3b-env
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
V=$SCR/venvs/se-s3b-fetch311
rm -rf "$V"
uv venv --python 3.12 "$V" || exit 1
P="$V/bin/python"

# The combination Step 3 never tried. It tested mujoco 3.13 + gymnasium-robotics (reference raises
# AssertionError) and mujoco 3.11 + mujoco_warp HEAD (renderer raises AttributeError on mjtCtrlInput).
# mujoco_warp publishes a v3.11.0 TAG whose whole point is to match mujoco 3.11, so the matched pair
# is a real third option rather than a rearrangement of the two that failed.
uv pip install --python "$P" --index-url https://download.pytorch.org/whl/cu126 torch || exit 1
uv pip install --python "$P" numpy opencv-python-headless imageio pillow "gymnasium>=1.0" warp-lang || exit 1

# Order matters. mujoco goes in FIRST and pinned; mujoco_warp goes in with --no-deps so its own floor
# cannot drag mujoco forward again -- that is exactly how the earlier attempt ended up at 3.13.
uv pip install --python "$P" "mujoco==3.11.0" || exit 1
uv pip install --python "$P" --no-deps \
    "mujoco_warp @ git+https://github.com/google-deepmind/mujoco_warp.git@v3.11.0" || exit 1
uv pip install --python "$P" --no-deps gymnasium-robotics || exit 1

echo "=== resolved versions ==="
"$P" -c "
import importlib.metadata as md
for n in ('torch','mujoco','mujoco-warp','gymnasium','gymnasium-robotics','warp-lang','numpy'):
    try: print(f'  {n:22s} {md.version(n)}')
    except Exception as e: print(f'  {n:22s} MISSING')
"
echo "=== the three-way test ==="
"$P" - <<'PY'
import traceback
ok = {}
try:
    import mujoco; ok['mujoco'] = mujoco.__version__
except Exception as e: ok['mujoco'] = f'FAIL {type(e).__name__}: {e}'
try:
    import mujoco_warp; ok['mujoco_warp'] = 'imports'
except Exception as e:
    ok['mujoco_warp'] = f'FAIL {type(e).__name__}: {str(e)[:120]}'
try:
    import gymnasium as gym, gymnasium_robotics  # noqa
    gym.register_envs(gymnasium_robotics)
    e = gym.make('FetchPush-v4'); e.reset(seed=0)
    ok['gymnasium_robotics'] = 'FetchPush-v4 reset OK'
except Exception as ex:
    ok['gymnasium_robotics'] = f'FAIL {type(ex).__name__}: {str(ex)[:120]}'
    traceback.print_exc()
for k, v in ok.items(): print(f'  {k:22s} {v}')
print('ALL THREE OK' if not any(str(v).startswith('FAIL') for v in ok.values()) else 'AT LEAST ONE FAILED')
PY
echo "S3B ENV DONE"
