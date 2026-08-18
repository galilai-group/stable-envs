"""PointMaze follow-ups: the Medium two-instance anomaly, and how the translucent target composites."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mujoco

gym.register_envs(gymnasium_robotics)


def build(eid, size=224):
    return gym.make(eid, render_mode="rgb_array", width=size, height=size, continuing_task=False)


def render_at(env, q, goal):
    u = env.unwrapped
    u.data.qpos[:] = np.asarray(q, np.float64)
    u.data.qvel[:] = 0.0
    u.goal = np.asarray(goal, np.float64)
    u.update_target_site_pos()
    mujoco.mj_forward(u.model, u.data)
    return u.point_env.mujoco_renderer.render("rgb_array").copy()


EID = "PointMaze_Medium-v3"
a = build(EID)
a.reset(seed=0)
q = np.array([0.3, -0.4])
g = np.array([1.0, 2.0])

f1 = render_at(a, q, g)
f2 = render_at(a, q, g)
print("same instance, twice:", int(np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max()))

b = build(EID)
b.reset(seed=123)
f3 = render_at(b, q, g)
d = np.abs(f1.astype(np.int16) - f3.astype(np.int16)).max(-1)
print("two instances:", int(d.max()), "differing px", int((d > 0).sum()), "/", d.size)
ys, xs = np.nonzero(d)
if len(ys):
    print("  bbox rows", ys.min(), ys.max(), "cols", xs.min(), xs.max())
    print("  a:", f1[ys[0], xs[0]], " b:", f3[ys[0], xs[0]])
    ma, mb = a.unwrapped.model, b.unwrapped.model
    for attr in ("geom_pos", "geom_size", "geom_rgba", "geom_type", "site_pos", "site_rgba",
                 "mat_rgba", "light_pos", "stat"):
        va, vb = getattr(ma, attr, None), getattr(mb, attr, None)
        if attr == "stat":
            print("  stat center", np.array(ma.stat.center), np.array(mb.stat.center),
                  "extent", ma.stat.extent, mb.stat.extent)
            continue
        same = np.array_equal(np.asarray(va), np.asarray(vb))
        if not same:
            print(f"  MODEL DIFFERS at {attr}")
    ca = a.unwrapped.point_env.mujoco_renderer._get_viewer("rgb_array").cam
    cb = b.unwrapped.point_env.mujoco_renderer._get_viewer("rgb_array").cam
    print("  cam a", np.array(ca.lookat), ca.distance, ca.azimuth, ca.elevation)
    print("  cam b", np.array(cb.lookat), cb.distance, cb.azimuth, cb.elevation)
    print("  maze map a == b:", a.unwrapped.maze.maze_map == b.unwrapped.maze.maze_map)

# ---- the translucent target: is the composite src-over with alpha 0.7?
u = a.unwrapped
m = u.model
tid = u.target_site_id
pid = 1 - tid
print("\ntarget site id", tid, "rgba", m.site_rgba[tid], " particle site rgba", m.site_rgba[pid])

keep = m.site_rgba[tid].copy()
with_t = render_at(a, q, g)
m.site_rgba[tid, 3] = 0.0
without_t = render_at(a, q, g)
m.site_rgba[tid] = keep
m.site_rgba[tid, 3] = 1.0
opaque_t = render_at(a, q, g)
m.site_rgba[tid] = keep

mask = np.abs(with_t.astype(np.int16) - without_t.astype(np.int16)).max(-1) > 0
print("target pixels:", int(mask.sum()), "of", mask.size)
if mask.any():
    A = with_t[mask].astype(np.float64)
    B = without_t[mask].astype(np.float64)
    C = opaque_t[mask].astype(np.float64)
    for al in (0.7, 0.699999988079071):
        pred = al * C + (1 - al) * B
        for rule, f in (("floor", np.floor), ("round", np.round)):
            e = np.abs(f(pred) - A)
            print(f"  alpha {al} {rule}: max {e.max():.3f} mean {e.mean():.4f} "
                  f"exact {float((e.max(-1) == 0).mean()):.4f}")

# is the target ever occluded by a wall or the ball at this camera?
print("\nsites visible? particle pixels:")
kp = m.site_rgba[pid].copy()
m.site_rgba[pid, 3] = 0.0
no_p = render_at(a, q, g)
m.site_rgba[pid] = kp
print("  ", int((np.abs(with_t.astype(np.int16) - no_p.astype(np.int16)).max(-1) > 0).sum()))
