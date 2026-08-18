"""Independent check of the parity-counter fill rule, on MY case set and on REAL car geometry.

probe_drawpoly_rule.py was written against its own random cases.  This re-runs the same transcribed
rule over (a) the 800 polygons from probe_drawpoly.py, which include heavily off-surface, thin and
axis-aligned quads, and (b) every polygon CarRacing actually hands to ``pygame.draw.polygon`` --
4 hull fixtures, 4 wheels, the wheel stripes and the 8 indicator quads -- transformed by real poses
taken from a real rollout.  If the rule holds there it holds where the port needs it.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pygame

from probe_drawpoly_rule import fill_mask

pygame.init()
W, H = 160, 120
BIG_W, BIG_H = 1000, 800


def ref(vx, vy, w, h):
    s = pygame.Surface((w, h))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), list(zip([float(v) for v in vx],
                                                     [float(v) for v in vy])))
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def my_cases():
    """The generator from probe_drawpoly.py, verbatim."""
    rng = np.random.default_rng(0)
    HULL3 = [(25, 20), (50, -10), (50, -40), (20, -90), (-20, -90), (-50, -40), (-50, -10), (-25, 20)]
    out = []
    for _ in range(400):
        c = rng.uniform(-40, W + 40, 2)
        r = rng.uniform(0.3, 90, 2)
        th = rng.uniform(0, 6.28)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        q = np.array([[-r[0], -r[1]], [r[0], -r[1]], [r[0], r[1]], [-r[0], r[1]]]) @ R.T + c
        out.append(q)
    for _ in range(200):
        c = rng.uniform(0, W, 2)
        sc = rng.uniform(0.05, 1.2)
        th = rng.uniform(0, 6.28)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        out.append((np.array(HULL3, float) * sc) @ R.T + c)
    for _ in range(200):
        x0, w, y0, hh = (rng.uniform(0, W), rng.uniform(0, 30), rng.uniform(0, H), rng.uniform(0, 3))
        out.append(np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + hh], [x0, y0 + hh]]))
    return out


def real_car_polys(n_frames=12):
    """Exactly the vertex lists ``Car.draw`` and ``_render_indicators`` pass to pygame.draw.polygon."""
    import gymnasium
    from gymnasium.envs.box2d.car_racing import SCALE, WINDOW_H, WINDOW_W, ZOOM
    from gymnasium.envs.box2d.car_dynamics import SIZE, WHEEL_R, WHEEL_W
    import math

    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    polys = []
    for _ in range(n_frames):
        for _ in range(10):
            env.step(np.array([float(rng.uniform(-0.6, 0.6)), 0.6, 0.0], np.float32))
        angle = -u.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - u.t, 0) + ZOOM * SCALE * min(u.t, 1)
        tr = pygame.math.Vector2((-u.car.hull.position[0] * zoom,
                                  -u.car.hull.position[1] * zoom)).rotate_rad(angle)
        tr = (WINDOW_W / 2 + tr[0], WINDOW_H / 4 + tr[1])
        for obj in u.car.drawlist:
            for f in obj.fixtures:
                path = [f.body.transform * v for v in f.shape.vertices]
                path = [pygame.math.Vector2((c[0], c[1])).rotate_rad(angle) for c in path]
                polys.append(np.array([[c[0] * zoom + tr[0], c[1] * zoom + tr[1]] for c in path]))
                if "phase" not in obj.__dict__:
                    continue
                a1, a2 = obj.phase, obj.phase + 1.2
                s1, s2, c1, c2 = math.sin(a1), math.sin(a2), math.cos(a1), math.cos(a2)
                if s1 > 0 and s2 > 0:
                    continue
                if s1 > 0:
                    c1 = np.sign(c1)
                if s2 > 0:
                    c2 = np.sign(c2)
                wp = [(-WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE), (+WHEEL_W * SIZE, +WHEEL_R * c1 * SIZE),
                      (+WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE), (-WHEEL_W * SIZE, +WHEEL_R * c2 * SIZE)]
                wp = [f.body.transform * v for v in wp]
                wp = [pygame.math.Vector2((c[0], c[1])).rotate_rad(angle) for c in wp]
                polys.append(np.array([[c[0] * zoom + tr[0], c[1] * zoom + tr[1]] for c in wp]))
        # the indicator quads, in post-flip coordinates
        s, h = WINDOW_W / 40.0, WINDOW_H / 40.0
        polys.append(np.array([[WINDOW_W, WINDOW_H], [WINDOW_W, WINDOW_H - 5 * h],
                               [0, WINDOW_H - 5 * h], [0, WINDOW_H]]))
        sp = float(np.sqrt(u.car.hull.linearVelocity[0] ** 2 + u.car.hull.linearVelocity[1] ** 2))
        for place, val in ((5, 0.02 * sp), (7, 0.01 * u.car.wheels[0].omega),
                           (8, 0.01 * u.car.wheels[1].omega), (9, 0.01 * u.car.wheels[2].omega),
                           (10, 0.01 * u.car.wheels[3].omega)):
            polys.append(np.array([[place * s, WINDOW_H - (h + h * val)],
                                   [(place + 1) * s, WINDOW_H - (h + h * val)],
                                   [(place + 1) * s, WINDOW_H - h], [place * s, WINDOW_H - h]]))
        for place, val in ((20, -10.0 * u.car.wheels[0].joint.angle),
                           (30, -0.8 * u.car.hull.angularVelocity)):
            polys.append(np.array([[place * s, WINDOW_H - 4 * h], [(place + val) * s, WINDOW_H - 4 * h],
                                   [(place + val) * s, WINDOW_H - 2 * h],
                                   [place * s, WINDOW_H - 2 * h]]))
    env.close()
    return polys


def run(cases, w, h, tag):
    ok = bad = 0
    worst = None
    for q in cases:
        vx = np.array([int(v) for v in q[:, 0]], np.int64)      # trunc toward zero, as measured
        vy = np.array([int(v) for v in q[:, 1]], np.int64)
        a = ref(vx, vy, w, h)
        b = fill_mask(vx, vy, w, h)
        d = int((a != b).sum())
        bad += d
        ok += d == 0
        if d and worst is None:
            worst = (vx.tolist(), vy.tolist(), d)
    print(f"  {tag}: {ok}/{len(cases)} polygons bit-identical, {bad} differing pixels"
          + (f"   first bad vx={worst[0]} vy={worst[1]} ({worst[2]} px)" if worst else ""))
    return bad


def main():
    print(f"pygame {pygame.version.ver}")
    print("=== the parity-counter rule, checked on cases it was NOT fitted to ===")
    total = run(my_cases(), W, H, "my 800 random/off-screen/thin quads and 8-gons")
    total += run(real_car_polys(), BIG_W, BIG_H,
                 "REAL car fixtures + wheel stripes + indicator quads, real poses")
    print(f"\ntotal differing pixels: {total}")


if __name__ == "__main__":
    main()
