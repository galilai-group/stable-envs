#!/bin/bash
#SBATCH --job-name=se-s3b-fix2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/oscar/scratch/%u/logs/%x_%j.out
set -uo pipefail
export SCR=/oscar/scratch/$USER
export UV_CACHE_DIR=$SCR/.uv-cache PIP_CACHE_DIR=$SCR/.pip-cache XDG_CACHE_HOME=$SCR/.cache
P=$SCR/venvs/se-s3b-fetch311/bin/python

# Stop chasing --no-deps casualties one import at a time. Let gymnasium-robotics resolve its FULL
# dependency set (it will drag mujoco forward doing so), then force mujoco back to 3.11 afterwards.
# That is the same move that failed in Step 3 -- with one decisive difference: mujoco_warp here is the
# v3.11.0 tag, built against mujoco 3.11, so pinning mujoco back to 3.11 is putting it where this
# mujoco_warp already wanted it rather than stranding a HEAD build that needed 3.13.
uv pip install --python "$P" gymnasium-robotics || exit 1
echo "after full resolve: $("$P" -c 'import importlib.metadata as m;print(m.version("mujoco"))')"
uv pip install --python "$P" --no-deps --reinstall "mujoco==3.11.0" || exit 1
uv pip install --python "$P" numpy opencv-python-headless imageio pillow || exit 1

"$P" -c "import importlib.metadata as m; print('FINAL mujoco', m.version('mujoco'), '| mujoco-warp', m.version('mujoco-warp'), '| gym-robotics', m.version('gymnasium-robotics'))"
"$P" - <<'PY'
import traceback
res = {}
for name, fn in (("mujoco", lambda: __import__("mujoco").__version__),
                 ("mujoco_warp", lambda: (__import__("mujoco_warp"), "imports")[1])):
    try: res[name] = fn()
    except Exception as e: res[name] = f"FAIL {type(e).__name__}: {str(e)[:110]}"
try:
    import gymnasium as gym, gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    e = gym.make("FetchPush-v4"); e.reset(seed=0)
    res["FetchPush-v4"] = "reset OK"
except Exception as ex:
    res["FetchPush-v4"] = f"FAIL {type(ex).__name__}: {str(ex)[:110]}"; traceback.print_exc()
for k, v in res.items(): print(f"  {k:16s} {v}")
print("MATCHED STACK WORKS" if not any(str(v).startswith("FAIL") for v in res.values())
      else "MATCHED STACK FAILS")
PY
echo "S3B FIX2 DONE"
