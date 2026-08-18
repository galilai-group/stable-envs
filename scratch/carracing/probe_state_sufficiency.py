"""Is the CarRacing frame a function of a compact state, or does it carry history?

Two candidates for hidden history that a 28-float render state would NOT capture:

  1. ``FrictionDetector._contact`` does ``tile.color[:] = self.env.road_color`` -- an IN-PLACE write
     to the same numpy array ``road_poly`` holds, so DRIVING OVER a tile permanently repaints it.
  2. ``Car.particles`` (skid marks) -- but ``_render`` passes ``draw_particles=False`` for
     state_pixels, so they should not reach the observation.

Both are measured here by driving, then restoring the geometric state and re-rendering.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import gymnasium


def pose(u):
    """EVERY render input except the reward and the tile colours: the 27 numbers a compact state
    would carry -- hull pose + velocities, four wheel poses, phases and omegas."""
    return ([tuple(u.car.hull.position), u.car.hull.angle,
             tuple(u.car.hull.linearVelocity), u.car.hull.angularVelocity]
            + [(tuple(w.position), w.angle, w.phase, w.omega) for w in u.car.wheels])


def restore(u, p):
    u.car.hull.position = p[0]
    u.car.hull.angle = p[1]
    u.car.hull.linearVelocity = p[2]
    u.car.hull.angularVelocity = p[3]
    for w, (pos, ang, ph, om) in zip(u.car.wheels, p[4:]):
        w.position = pos
        w.angle = ang
        w.phase = ph
        w.omega = om


def main():
    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=0)
    u = env.unwrapped

    # colours BEFORE anything is driven over
    cols0 = np.array([np.asarray(c, np.float64).copy() for _, c in u.road_poly])
    p0 = pose(u)
    u.t = 2.0                                     # past the zoom ramp, so zoom is constant
    f0 = u.render().copy()

    for _ in range(120):
        env.step(np.array([0.0, 0.4, 0.0], np.float32))
    cols1 = np.array([np.asarray(c, np.float64).copy() for _, c in u.road_poly])
    n_changed = int((np.abs(cols1 - cols0).max(1) > 0).sum())

    # put the car back exactly where it started and re-render
    restore(u, p0)
    u.t = 2.0
    u.reward = 0.0
    f1 = u.render().copy()
    d = np.abs(f0.astype(np.int16) - f1.astype(np.int16)).max(-1)

    print("=== hidden history in the CarRacing observation ===")
    print(f"  road_poly entries whose colour was mutated by driving: {n_changed}/{len(cols0)}")
    if n_changed:
        i = int(np.nonzero(np.abs(cols1 - cols0).max(1) > 0)[0][0])
        print(f"    e.g. tile {i}: {tuple(cols0[i])} -> {tuple(cols1[i])}")
    print(f"  same geometric state + same reward, after driving: {int((d > 0).sum())} of {d.size} "
          f"pixels differ (max {int(d.max())}/255)")
    rows = np.nonzero(d.any(1))[0]
    print(f"    differing rows {rows.min()}..{rows.max()}; "
          f"in the indicator strip (rows>=84): {int(d[84:].sum() > 0)}, "
          f"count there {int((d[84:] > 0).sum())}, elsewhere {int((d[:84] > 0).sum())}")
    print(f"    max |d| outside the indicator strip: {int(d[:84].max())}")
    # control: is the reference deterministic at all?
    f2 = u.render().copy()
    print(f"    control, two renders in a row: "
          f"{int((np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max(-1) > 0).sum())} px differ")
    print(f"  particles alive: {len(u.car.particles)} (draw_particles is False for state_pixels)")
    env.close()


if __name__ == "__main__":
    main()
