"""Are the '2-D' DM Control domains actually 2-D vector art under their fixed camera?

The question that decides whether they belong in a polygon/disc rasterizer is not "does the physics
live in a plane" -- cartpole, pendulum, reacher, finger and ball_in_cup all do -- but "is the
rendered image an affine function of the plane". MuJoCo renders through a perspective camera with a
directional light, a shadow map, a textured checker floor and a gradient skybox, so the test is
whether the image shows depth cues (shading gradients, shadows, texture) that a flat rasterizer
cannot emit. Counting distinct colours and checking for smooth gradients answers it.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def main():
    from dm_control import suite

    rows = []
    for dom, task in [("cartpole", "swingup"), ("pendulum", "swingup"), ("reacher", "easy"),
                      ("finger", "spin"), ("ball_in_cup", "catch"), ("point_mass", "easy"),
                      ("acrobot", "swingup")]:
        try:
            env = suite.load(dom, task, task_kwargs={"random": 0})
        except Exception as e:                                     # noqa: BLE001
            print(f"{dom}/{task}: load failed {e}")
            continue
        ts = env.reset()
        m = env.physics.model
        img = env.physics.render(camera_id=0, height=240, width=320)
        uniq = len(np.unique(img.reshape(-1, 3), axis=0))
        # gradient content: mean absolute Laplacian away from edges, a flat-shaded vector image is 0
        g = img.astype(np.int32)
        lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        soft = float(((lap > 0) & (lap < 24)).mean())
        ncam = m.ncam
        cams = []
        for c in range(ncam):
            cams.append((m.camera(c).name, float(m.cam_fovy[c]),
                         np.asarray(m.cam_pos[c]).round(3).tolist()))
        nq = m.nq
        ngeom = m.ngeom
        gtypes = {}
        for gi in range(ngeom):
            t = int(m.geom_type[gi])
            gtypes[t] = gtypes.get(t, 0) + 1
        rows.append((f"{dom}/{task}", nq, ngeom, ncam, uniq, soft, gtypes, cams[:1]))
        env.close() if hasattr(env, "close") else None

    print(f"{'domain/task':22s} {'nq':>3s} {'ngeom':>6s} {'ncam':>5s} {'colours':>8s} "
          f"{'soft-grad px':>13s}  geom types")
    print("-" * 100)
    TYPE = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
            5: "cylinder", 6: "box", 7: "mesh"}
    for name, nq, ng, nc, uq, soft, gt, cams in rows:
        gs = ",".join(f"{TYPE.get(k, k)}x{v}" for k, v in sorted(gt.items()))
        print(f"{name:22s} {nq:3d} {ng:6d} {nc:5d} {uq:8d} {100*soft:12.1f}% {gs}")
    print("\n  fovy > 0 means a PERSPECTIVE camera (no affine world->pixel map):")
    for name, *_r, cams in rows:
        for cn, fovy, pos in cams:
            print(f"    {name:22s} cam '{cn}' fovy={fovy:.1f} pos={pos}")

    # a flat-shaded 2-D renderer emits a handful of colours; check what pygame classic control does
    print("\n  for scale, gymnasium classic control (pure gfxdraw vector art):")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import gymnasium as gym
    for nm in ("CartPole-v1", "Pendulum-v1", "Acrobot-v1", "MountainCar-v0"):
        e = gym.make(nm, render_mode="rgb_array").unwrapped
        e.reset(seed=0)
        im = e.render()
        g = im.astype(np.int32)
        lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        soft = float(((lap > 0) & (lap < 24)).mean())
        print(f"    {nm:16s} colours {len(np.unique(im.reshape(-1,3), axis=0)):6d}  "
              f"soft-grad px {100*soft:6.3f}%")
        e.close()


if __name__ == "__main__":
    main()
