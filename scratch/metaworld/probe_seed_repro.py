"""Is a Meta-World frame even a function of (task, seed)? Asked before blaming any renderer.

Two resets with the SAME seed, in one process, rendered at the same qpos. If they differ, the
episode draw is not determined by the seed, and no state vector recovered from one episode can be
checked against another.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scratch.metaworld.probe_metaworld import make                          # noqa: E402

TASKS = ["reach-v3", "pick-place-v3", "door-open-v3", "button-press-v3", "drawer-open-v3"]

out = []
for t in (sys.argv[1:] or TASKS):
    env = make(t, 128)
    u = env.unwrapped
    env.reset(seed=1)
    a = u.render().copy()
    ta = np.array(getattr(u, "_target_pos", np.zeros(3)), np.float64)
    env.reset(seed=1)
    b = u.render().copy()
    tb = np.array(getattr(u, "_target_pos", np.zeros(3)), np.float64)
    # metaworld's own docstring: "seed: The seed to use. Ignored, use `seed()` instead."
    u.seed(1)
    env.reset()
    c = u.render().copy()
    tc = np.array(getattr(u, "_target_pos", np.zeros(3)), np.float64)
    u.seed(1)
    env.reset()
    d2 = u.render().copy()
    td = np.array(getattr(u, "_target_pos", np.zeros(3)), np.float64)
    r = {"task": t,
         "reset_seed_frame_maxdiff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()),
         "reset_seed_target_maxdiff": float(np.abs(ta - tb).max()),
         "explicit_seed_frame_maxdiff": int(np.abs(c.astype(np.int16) - d2.astype(np.int16)).max()),
         "explicit_seed_target_maxdiff": float(np.abs(tc - td).max())}
    out.append(r)
    print(json.dumps(r), flush=True)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_seed_repro.json")
json.dump(out, open(p, "w"), indent=1)
print("wrote", p)
