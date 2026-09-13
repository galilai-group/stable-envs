"""Is a scene's STATIC primitive set a function of that scene, or of the last bake?

The reviewer's reproduction, kept because it is the only thing that distinguishes the two. Render a
hidden-target scene at a supersample nothing has cached yet, twice: once from a fresh process state,
and once after another appearance has been baked in between. ``_static_t`` used to re-derive the
static set from the LIVE shared ``MjModel``, which carries the last-baked appearance, so the second
answer promoted the hidden element to a static primitive and the two renders disagreed.

The supersample change is load-bearing: ``static_t`` is cached per (scene, ss), and at the default
ss it is always cast during the scene's own first render, while that scene's appearance is still on
the model. That is why every default-ss suite passed.

    PYTHONPATH=$PWD MUJOCO_GL=glfw python probe_static_leak.py DMCReacherHard DMCFingerTurnHard
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np

import stable_envs as se

HIDDEN = {"visibility.target": 0}

for name in sys.argv[1:] or ["DMCReacherHard", "DMCFingerTurnHard"]:
    st = np.load(f"examples/results/step4/states/{name}.npz")["states"]
    e = se.make(name, batch=len(st), device="cpu")

    # (A) the hidden scene at ss=2, with nothing else baked in this process first
    import stable_envs.envs.dmcontrol as D
    D._SCENE_CACHE.clear()
    a = e.render(st, variation=HIDDEN, supersample=2).cpu().numpy()

    # (B) the same thing, with a DIFFERENT appearance baked in between
    D._SCENE_CACHE.clear()
    e.render(st, variation=HIDDEN)                       # default ss: bakes the hidden scene
    e.render(st, variation={"material.grid": [0.2, 0.9, 0.4]})   # another bake, another appearance
    b = e.render(st, variation=HIDDEN, supersample=2).cpu().numpy()

    d = np.abs(a.astype(int) - b.astype(int)).max(-1)
    print(f"{name:20} px={int((d > 0).sum())} max={int(d.max())} "
          f"rows={int((d.reshape(len(st), -1) > 0).any(1).sum())} of {len(st)}")
