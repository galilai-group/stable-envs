"""PointMaze: parity against the official renderer, where the residual lives, and throughput."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, ".")
import stable_envs as se
from stable_envs.testing import conformance as C

SIZE = 224
for name in ("PointMazeUMaze", "PointMazeOpen", "PointMazeMedium", "PointMazeLarge"):
    env = se.make(name, batch=8, size=SIZE, device="cuda")
    cls = type(env)
    ref = C.reference_env(cls, SIZE)
    st = env.sample_states(32, rng=np.random.default_rng(0))
    official = cls.reference_render(ref, st)
    ours = env.render(st).cpu().numpy()
    r = C.pixel_agreement(ours, official)
    print(f"\n{name}: {r}")

    # where does it live? re-render the reference with both sites hidden to get the plate mask
    u = ref.unwrapped
    m = u.model
    keep = m.site_rgba[:, 3].copy()
    m.site_rgba[:, 3] = 0.0
    plate_only = cls.reference_render(ref, st)
    m.site_rgba[:, 3] = keep
    moving = np.abs(official.astype(np.int16) - plate_only.astype(np.int16)).max(-1) > 0
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    print(f"  moving-primitive pixels {100*moving.mean():.3f}% of the frame")
    print(f"  static : exact {float((d[~moving] == 0).mean()):.6f} mean {d[~moving].mean():.5f} "
          f"max {d[~moving].max()}")
    print(f"  moving : exact {float((d[moving] == 0).mean()):.4f} "
          f"within1 {float((d[moving] <= 1).mean()):.4f} mean {d[moving].mean():.3f} "
          f"max {d[moving].max()}")

    # a state where the ball sits on its own goal, i.e. the occlusion fallback
    on_goal = st[:4].copy()
    on_goal[:, 2:4] = on_goal[:, 0:2]
    o2 = env.render(on_goal).cpu().numpy()
    f2 = cls.reference_render(ref, on_goal)
    d2 = np.abs(o2.astype(np.int16) - f2.astype(np.int16)).max(-1)
    print(f"  ball on goal: exact {float((d2 == 0).mean()):.6f} mean {d2.mean():.4f} max {d2.max()}")

    if name == "PointMazeMedium":
        for B in (256, 1024, 4096):
            s = torch.as_tensor(env.sample_states(B), device="cuda")
            out = torch.empty(B, SIZE, SIZE, 3, dtype=torch.uint8, device="cuda")
            env.render(s, out=out)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            env.render(s, out=out)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            print(f"  B={B:5d} {B/dt:10,.0f} f/s")
        t0 = time.perf_counter()
        cls.reference_render(ref, st)
        print(f"  official MuJoCo EGL: {len(st)/(time.perf_counter()-t0):,.0f} f/s")
