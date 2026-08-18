"""Is the reference's checker floor MINIFICATION-FILTERED, or would nearest-texel reproduce it?

The distinct-colour count alone does not decide it (shading and the mirror contribute too), so this
asks the question the filter answers directly. A checker of rgb1=(0,0,0) and rgb2=(0.8,0.8,0.8) at
texrepeat 60 on a 40 m plane is magnified near the camera and heavily minified toward the horizon.

  * A nearest-texel sampler keeps the FULL black/white contrast at every distance: the far field
    aliases into speckle, so its per-pixel variance stays as high as the near field's.
  * A mipmapped sampler averages many texels per pixel out there, so the far field converges to one
    mid-grey and its variance collapses.

So: split the floor by image row and report the spread of luminance in the nearest and farthest
bands. The ratio is the answer, and it does not depend on modelling the UVs.
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import gymnasium as gym
import mujoco

ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "Swimmer-v5"]


def run(env_id, size=128):
    env = gym.make(env_id, render_mode="rgb_array", width=size, height=size)
    u = env.unwrapped
    env.reset(seed=0)
    f = u.render().copy()
    r = mujoco.Renderer(u.model, height=size, width=size)
    r.enable_segmentation_rendering()
    r.update_scene(u.data, camera=0)
    seg = r.render()
    r.disable_segmentation_rendering()
    floor = (seg[..., 1].astype(np.int64) == int(mujoco.mjtObj.mjOBJ_GEOM)) & (
        seg[..., 0].astype(np.int64) == 0)
    lum = f.astype(np.float64).mean(-1)
    rows = np.nonzero(floor.any(1))[0]
    if len(rows) < 10:
        return {"env": env_id, "error": "no floor"}
    lo, hi = rows.min(), rows.max()                 # the floor spans rows lo..hi; hi is NEAREST
    band = max(1, (hi - lo) // 10)
    near = floor.copy(); near[:hi - band] = False
    far = floor.copy(); far[lo + band:] = False
    out = {"env": env_id, "floor_rows": [int(lo), int(hi)]}
    for tag, msk in (("near", near), ("far", far)):
        v = lum[msk]
        out[tag] = {"n": int(v.size), "mean": round(float(v.mean()), 2),
                    "std": round(float(v.std()), 2),
                    "p5": round(float(np.percentile(v, 5)), 1),
                    "p95": round(float(np.percentile(v, 95)), 1)}
    out["far_over_near_std"] = round(out["far"]["std"] / max(out["near"]["std"], 1e-9), 4)
    return out


if __name__ == "__main__":
    out = []
    for e in (sys.argv[1:] or ENVS):
        try:
            r = run(e)
        except Exception as exc:
            import traceback; traceback.print_exc()
            r = {"env": e, "error": f"{type(exc).__name__}: {exc}"}
        out.append(r); print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_minification.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
