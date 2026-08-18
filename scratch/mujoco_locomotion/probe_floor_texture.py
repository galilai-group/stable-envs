"""Do the floor's 550-975 colours come from the TEXTURE, or from shading it?

The NO-GO for the six tracking domains rests on the claim that the reference's checker floor is
mipmap-filtered, and the evidence offered was the number of distinct colours it paints there. That
number is not decisive on its own: the floor material has ``specular="1" shininess="1"`` and
``reflectance="0.5"``, and both vary continuously across a plane, so a two-texel checker could show
many colours without any filtering at all.

This isolates it by overwriting the checker's texel data with a CONSTANT colour before the GL context
is built — same material, same lights, same camera, same mirror — and re-counting. If the count
collapses, the colours were the texture; if it does not, they were the shading and the claim is
wrong.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import gymnasium as gym
import mujoco

ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Humanoid-v5", "Swimmer-v5"]


def build(env_id, size, flat_texture=False, no_reflect=False):
    """A fresh env; texture and material edits applied BEFORE the first render, which is when the
    ``MjrContext`` uploads texture data and cannot be told about a later change."""
    env = gym.make(env_id, render_mode="rgb_array", width=size, height=size)
    u = env.unwrapped
    m = u.model
    if flat_texture:
        attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
        buf = getattr(m, attr)
        for t in range(m.ntex):
            if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_2D):
                a0 = int(m.tex_adr[t])
                n = 3 * int(m.tex_height[t]) * int(m.tex_width[t])
                buf[a0:a0 + n] = np.tile(np.array([102, 102, 102], buf.dtype), n // 3)
    if no_reflect:
        m.mat_reflectance[:] = 0.0
    env.reset(seed=0)
    u.render()
    return env, u


def floor_colours(env_id, size=128):
    out = {"env": env_id, "size": size}
    base, u0 = build(env_id, size)
    m0, d0 = u0.model, u0.data
    r = mujoco.Renderer(m0, height=size, width=size)
    r.enable_segmentation_rendering()
    r.update_scene(d0, camera=0)
    seg = r.render()
    r.disable_segmentation_rendering()
    floor = (seg[..., 1].astype(np.int64) == int(mujoco.mjtObj.mjOBJ_GEOM)) & (
        seg[..., 0].astype(np.int64) == 0)
    out["floor_frac"] = round(float(floor.mean()), 4)

    for tag, kw in (("checker", {}), ("flat_texture", dict(flat_texture=True)),
                    ("flat_texture_no_mirror", dict(flat_texture=True, no_reflect=True))):
        e, u = build(env_id, size, **kw)
        u.mujoco_renderer.viewer.make_context_current()
        f = u.render()
        px = f[floor]
        out[tag] = {"n_unique": int(len(np.unique(px.reshape(-1, 3), axis=0))),
                    "n_pixels": int(floor.sum())}
    return out


if __name__ == "__main__":
    out = []
    for e in (sys.argv[1:] or ENVS):
        try:
            r = floor_colours(e)
        except Exception as exc:                                            # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"env": e, "error": f"{type(exc).__name__}: {exc}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_floor_texture.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
