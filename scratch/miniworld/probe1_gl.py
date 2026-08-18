"""Probe 1: can MiniWorld render headless at all, and is its renderer deterministic?

Asks the cheapest questions first, in the order that makes a later answer worthless if an earlier
one fails:

  A. does ``import miniworld`` succeed on a GPU node, and under which GL platform?
  B. does a frame come out non-uniform? (the SDL_VIDEODRIVER=dummy trap, generalised: a blank
     surface makes every parity number a comparison against black)
  C. is the SAME env deterministic across repeated renders of the same state?
  D. DISPLAY LIST 1. ``MiniWorldEnv._render_static`` compiles the whole static scene into display
     list **1**, a hard-coded global id, and every pyglet window in a process shares an object
     space by default. So two MiniWorld envs alive at once may share one display list, and the
     last one to reset would own it. The conformance suite keeps one reference env per (class,
     size) alive for the whole run, so if this is true the family cannot be tested at all in the
     shape this repo tests things.
"""
from __future__ import annotations

import os
import sys

import numpy as np

RESULT = {}


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


def main():
    banner("A. import")
    print("PYOPENGL_PLATFORM =", os.environ.get("PYOPENGL_PLATFORM"))
    print("DISPLAY =", os.environ.get("DISPLAY"))
    import pyglet
    # MEASURED, not read off the source: miniworld/opengl.py sets pyglet.options["headless"] from
    # PYOPENGL_PLATFORM, but it is imported AFTER miniworld/entity.py has already done
    # `from pyglet.gl import ...`, and that import creates pyglet's shadow window at import time
    # (pyglet/window/__init__.py:1919 -> gl._create_shadow_window()). So the env var alone is too
    # late and `import miniworld` dies with NoSuchDisplayException. The option has to be set before
    # pyglet.gl is ever touched.
    pyglet.options["headless"] = True
    print("pyglet", pyglet.version, "headless option:", pyglet.options.get("headless"))
    import miniworld
    print("miniworld", miniworld.__file__)
    import gymnasium as gym
    ids = sorted(k for k in gym.registry if k.startswith("MiniWorld"))
    print(f"{len(ids)} ids:", ids)

    banner("B. one frame, and is it uniform?")
    env = gym.make("MiniWorld-OneRoom-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    print("obs", obs.shape, obs.dtype, "uniform:", len(np.unique(obs.reshape(-1, 3), axis=0)) == 1)
    img = u.render()
    print("render()", img.shape, img.dtype)
    uniq = np.unique(img.reshape(-1, 3), axis=0)
    print(f"render() distinct colours: {len(uniq)}   mean {img.mean():.2f}")
    assert len(uniq) > 1, "BLANK FRAME — every parity number would be measured against a constant"
    print("obs_fb", u.obs_fb.width, u.obs_fb.height, " vis_fb", u.vis_fb.width, u.vis_fb.height)
    print("agent cam_fov_y", u.agent.cam_fov_y, "cam_height", u.agent.cam_height)
    print("sky_color", u.sky_color, "light_pos", u.light_pos, "light_color", u.light_color,
          "light_ambient", u.light_ambient)

    banner("C. determinism: same env, same state, render N times")
    frames = [u.render() for _ in range(5)]
    d = max(int(np.abs(frames[0].astype(np.int16) - f.astype(np.int16)).max()) for f in frames)
    print(f"max abs diff over 5 renders of the same state: {d}/255")
    RESULT["determinism_same_env"] = d

    # and after a no-op detour through another render path
    u.render_obs()
    f2 = u.render()
    d2 = int(np.abs(frames[0].astype(np.int16) - f2.astype(np.int16)).max())
    print(f"after an intervening render_obs(): {d2}/255")
    RESULT["determinism_after_detour"] = d2

    banner("D. display list 1: does a second env clobber the first?")
    st = np.array([*u.agent.pos, u.agent.dir], np.float64)
    before = u.render()

    env2 = gym.make("MiniWorld-TMazeLeft-v0", render_mode="rgb_array")
    env2.reset(seed=0)
    u2 = env2.unwrapped
    print("env2 built and reset; now re-render env1 at the SAME state")

    u.agent.pos = st[:3].copy()
    u.agent.dir = float(st[3])
    after = u.render()
    d3 = int(np.abs(before.astype(np.int16) - after.astype(np.int16)).max())
    frac = float((np.abs(before.astype(np.int16) - after.astype(np.int16)).max(-1) > 0).mean())
    print(f"env1 frame after env2 exists: max {d3}/255 over {100*frac:.2f}% of pixels")
    RESULT["display_list_interference_max"] = d3
    RESULT["display_list_interference_frac"] = frac

    # Does re-running _render_static fix it?
    u._render_static()
    fixed = u.render()
    d4 = int(np.abs(before.astype(np.int16) - fixed.astype(np.int16)).max())
    print(f"after re-running env1._render_static(): max {d4}/255")
    RESULT["display_list_repair_max"] = d4

    banner("RESULT")
    for k, v in RESULT.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
