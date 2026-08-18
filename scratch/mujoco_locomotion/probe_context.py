"""How much does an unbound GL context cost? Measured, because it is silent.

``gymnasium.envs.mujoco.mujoco_rendering.OffScreenViewer.render`` never binds its own EGL context;
``MujocoRenderer._get_viewer`` binds only when one renderer owns more than one viewer. Anything else
that makes a context current — a second gymnasium env, a ``mujoco.Renderer``, a dm_control physics —
therefore steals it, and every later frame from the first env is drawn against a context that does
not own its framebuffer.

This renders one env at one state, creates an interloper, renders the same state again, and diffs.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import gymnasium as gym
import mujoco

ENVS = ["Reacher-v5", "Pusher-v5", "InvertedPendulum-v5", "InvertedDoublePendulum-v5"]


def diff(x, y):
    d = np.abs(x.astype(np.int16) - y.astype(np.int16)).max(-1)
    return {"max": int(d.max()), "frac_changed": round(float((d > 0).mean()), 4)}


def probe(env_id, size=240):
    """Four interlopers, in increasing order of how much of MuJoCo they touch."""
    env = gym.make(env_id, render_mode="rgb_array", width=size, height=size)
    u = env.unwrapped
    env.reset(seed=0)
    a = u.render().copy()
    out = {"env": env_id, "size": size}
    keep = []

    r = mujoco.Renderer(u.model, height=size, width=size)
    r.update_scene(u.data)
    r.render()
    keep.append(r)
    out["after_plain_renderer"] = diff(u.render(), a)

    r2 = mujoco.Renderer(u.model, height=size, width=size)
    r2.enable_segmentation_rendering()
    r2.update_scene(u.data, camera=u.mujoco_renderer.viewer.cam)   # the VIEWER's own camera object
    r2.render()
    keep.append(r2)
    out["after_segmentation_renderer_on_viewer_cam"] = diff(u.render(), a)
    out["viewer_cam_after"] = dict(
        lookat=[float(x) for x in u.mujoco_renderer.viewer.cam.lookat],
        distance=float(u.mujoco_renderer.viewer.cam.distance),
        azimuth=float(u.mujoco_renderer.viewer.cam.azimuth),
        elevation=float(u.mujoco_renderer.viewer.cam.elevation))

    other = gym.make(env_id, render_mode="rgb_array", width=size, height=size)
    other.reset(seed=0)
    other.unwrapped.render()
    keep.append(other)
    out["after_second_gym_env"] = diff(u.render(), a)

    third = gym.make(env_id, render_mode="rgb_array", width=size, height=size)
    third.unwrapped.model.vis.quality.offsamples = 1
    third.reset(seed=0)
    third.unwrapped.render()
    keep.append(third)
    out["after_offsamples1_env"] = diff(u.render(), a)

    u.mujoco_renderer.viewer.make_context_current()
    out["after_explicit_rebind"] = diff(u.render(), a)
    probe.keep = keep
    return out


if __name__ == "__main__":
    out = [probe(e) for e in (sys.argv[1:] or ENVS)]
    for r in out:
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_context.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
