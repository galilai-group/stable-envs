"""Which light's specular actually reaches the frame, and what the shading constants really are.

``docs/research/ogbench.md`` fits the specular weight at 0.25-0.26 where the light table predicts
``(headlight 0.5 + model light 0.1) * material 0.5 = 0.30``, and records the discrepancy without
resolving it. 0.25 is exactly ``0.5 * 0.5``, i.e. what you get if the MODEL light's specular never
arrives -- so that is a hypothesis with a one-line experiment: zero each light's specular in turn, on
the REFERENCE, and see which one moves the picture.

The remaining OGBench floor residual is mean 3.9/255 with 99.996 % of pixels within 8, which is the
size a 0.05 error in the specular weight would have on a texture of ~128-255, so this is the term
worth resolving before anything else.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from common import emit, save  # noqa: E402


def render(m, d, cam, size):
    import mujoco
    r = mujoco.Renderer(m, height=size, width=size)
    r.update_scene(d, camera=cam)
    img = r.render().copy()
    del r
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", default="point")
    ap.add_argument("--maze", default="medium")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    import gymnasium
    import mujoco
    import ogbench  # noqa: F401

    env = gymnasium.make(f"{args.loco}maze-{args.maze}-v0", render_mode="rgb_array",
                         ob_type="pixels", width=args.size, height=args.size)
    env.reset(options={"task_id": 1})
    u = env.unwrapped
    m, d = u.model, u.data
    m.vis.quality.offsamples = 0
    cam = u.custom_camera
    mujoco.mj_forward(m, d)

    base = render(m, d, cam, args.size)
    keep = {"light_specular": np.array(m.light_specular, copy=True),
            "light_diffuse": np.array(m.light_diffuse, copy=True),
            "light_ambient": np.array(m.light_ambient, copy=True),
            "hl_spec": np.array(m.vis.headlight.specular, copy=True),
            "hl_diff": np.array(m.vis.headlight.diffuse, copy=True),
            "hl_amb": np.array(m.vis.headlight.ambient, copy=True),
            "mat_spec": np.array(m.mat_specular, copy=True)}

    def restore():
        m.light_specular[:] = keep["light_specular"]
        m.light_diffuse[:] = keep["light_diffuse"]
        m.light_ambient[:] = keep["light_ambient"]
        m.vis.headlight.specular[:] = keep["hl_spec"]
        m.vis.headlight.diffuse[:] = keep["hl_diff"]
        m.vis.headlight.ambient[:] = keep["hl_amb"]
        m.mat_specular[:] = keep["mat_spec"]

    rows = []
    cases = {
        "model_light_specular=0": lambda: m.light_specular.__setitem__(slice(None), 0.0),
        "headlight_specular=0": lambda: m.vis.headlight.specular.__setitem__(slice(None), 0.0),
        "material_specular=0": lambda: m.mat_specular.__setitem__(slice(None), 0.0),
        "model_light_diffuse=0": lambda: m.light_diffuse.__setitem__(slice(None), 0.0),
        "headlight_diffuse=0": lambda: m.vis.headlight.diffuse.__setitem__(slice(None), 0.0),
        "headlight_ambient=0": lambda: m.vis.headlight.ambient.__setitem__(slice(None), 0.0),
        "model_light_ambient=0": lambda: m.light_ambient.__setitem__(slice(None), 0.0),
    }
    for name, fn in cases.items():
        restore()
        fn()
        img = render(m, d, cam, args.size)
        dd = np.abs(img.astype(np.int16) - base.astype(np.int16)).max(-1)
        row = {"case": name, "pixels_changed": float((dd > 0).mean()), "mean": float(dd.mean()),
               "max": int(dd.max())}
        emit("ablate", row)
        rows.append(row)
    restore()
    save(f"spec_{args.loco}_{args.maze}.json", rows)


if __name__ == "__main__":
    main()
