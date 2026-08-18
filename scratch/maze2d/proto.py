"""PointMaze prototype: bake the plate, cast the two sites, blend the translucent one, measure.

Everything the env will do, in one script, so the constants are settled before any of it is written
into stable_envs/envs/maze2d.py.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mujoco

sys.path.insert(0, ".")
from stable_envs.render import raycast3d as R3

gym.register_envs(gymnasium_robotics)

MAZE = sys.argv[1] if len(sys.argv) > 1 else "Medium"
SIZE = 224
DEV = torch.device("cuda")


def build(maze, size):
    return gym.make(f"PointMaze_{maze}-v3", render_mode="rgb_array", width=size, height=size,
                    continuing_task=False)


env = build(MAZE, SIZE)
env.reset(seed=0)
u = env.unwrapped
m, d = u.model, u.data
viewer = u.point_env.mujoco_renderer._get_viewer("rgb_array")
TID = u.target_site_id
PID = 1 - TID
print("target site", TID, m.site_rgba[TID], "particle site", PID, m.site_rgba[PID])
print("headlight amb/dif/spec", np.array(m.vis.headlight.ambient), np.array(m.vis.headlight.diffuse),
      np.array(m.vis.headlight.specular), "offsamples", m.vis.quality.offsamples)

geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
static = [g for g, n in enumerate(geom_names) if n != "particle_geom"]
print("static geoms", len(static), "kinds", sorted({int(m.geom_type[g]) for g in static}))


def set_state(q, goal):
    d.qpos[:] = np.asarray(q, np.float64)
    d.qvel[:] = 0.0
    u.goal = np.asarray(goal, np.float64)
    u.update_target_site_pos()
    mujoco.mj_forward(m, d)


def official(q, goal):
    set_state(q, goal)
    return u.point_env.mujoco_renderer.render("rgb_array").copy()


def _mat_and_tex(g):
    mid = int(m.geom_matid[g])
    mat = np.asarray(m.geom_rgba[g], np.float64)[:3]
    tex = np.ones(3)
    if mid >= 0:
        if np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
            mat = np.asarray(m.mat_rgba[mid], np.float64)[:3]
        tid = m.mat_texid[mid]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        if tid >= 0:
            attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
            buf = getattr(m, attr)
            a0 = int(m.tex_adr[tid])
            h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
            reg = np.asarray(buf[a0:a0 + 3 * h * w], np.float64).reshape(h, w, 3) / 255.0
            uu = np.unique(reg.reshape(-1, 3), axis=0)
            tex = uu[0] if len(uu) == 1 else reg.reshape(-1, 3).mean(0)
    return mat, tex


# ---- bake the plate with both sites hidden -----------------------------------------------------
keep = m.site_rgba[:, 3].copy()
m.site_rgba[:, 3] = 0.0
plate = official(np.zeros(2), np.zeros(2))
m.site_rgba[:, 3] = keep
print("plate", plate.shape, "unique colours", len(np.unique(plate.reshape(-1, 3), axis=0)))

official(np.zeros(2), np.zeros(2))                    # repopulate the scene with the real appearance
scn = viewer.scn
cam = R3.Camera.from_mjv(scn.camera[0], scn.camera[1])
L = int(scn.nlight)
ldir = np.stack([-np.asarray(scn.lights[i].dir, np.float64) for i in range(L)])
print("nlight", L, "dir", ldir.tolist(), "headlight pos", np.asarray(scn.lights[0].pos).tolist(),
      "type", int(getattr(scn.lights[0], "type", -1)))

sprims = R3.Primitives(
    kind=np.array([int(m.geom_type[g]) for g in static]),
    size=np.stack([m.geom_size[g] for g in static]),
    mat_rgb=np.stack([_mat_and_tex(g)[0] for g in static]),
    tex_rgb=np.stack([_mat_and_tex(g)[1] for g in static]),
    spec=np.full(len(static), 0.5))
plate_t = R3.static_depth(cam, sprims, np.stack([d.geom_xpos[g] for g in static]),
                          np.stack([d.geom_xmat[g] for g in static]), SIZE, SIZE, DEV)
plate_g = torch.tensor(plate, device=DEV, dtype=torch.uint8)

SITE_R = float(m.site_size[PID][0])
TARGET_Z = float(u.maze.maze_height / 2 * u.maze.maze_size_scaling)
ALPHA = float(m.site_rgba[TID][3])
print("site radius", SITE_R, "target z", TARGET_Z, "target alpha", ALPHA)

BALL_RGB = np.asarray(m.site_rgba[PID][:3], np.float64)
TGT_RGB = np.asarray(m.site_rgba[TID][:3], np.float64)


def make_prims(with_target):
    ks = [R3.SPHERE] + ([R3.SPHERE] if with_target else [])
    rgb = [BALL_RGB] + ([TGT_RGB] if with_target else [])
    n = len(ks)
    return R3.Primitives(kind=np.array(ks), size=np.tile([SITE_R, 0, 0], (n, 1)),
                         mat_rgb=np.stack(rgb), tex_rgb=np.ones((n, 3)), spec=np.full(n, 0.5))


def lights_for(A, D, S, P):
    nA, nD, nS = 0.4, 0.8, 0.1
    return R3.Lights(dirs=ldir / np.linalg.norm(ldir, axis=1, keepdims=True),
                     ambient=np.array([A / nA * nA]), diffuse=np.array([D / nD * nD]),
                     specular=np.array([S / nS * nS]), exponent=P)


def render_ours(qs, goals, A=0.4, D=0.8, S=0.1, P=40.0):
    B = len(qs)
    lig = R3.Lights(dirs=ldir / np.linalg.norm(ldir, axis=1, keepdims=True),
                    ambient=np.array([A]), diffuse=np.array([D]), specular=np.array([S]),
                    exponent=P)
    eye = torch.eye(3, device=DEV).view(1, 1, 3, 3)
    xb = torch.zeros(B, 1, 3, device=DEV)
    xb[:, 0, :2] = torch.as_tensor(np.asarray(qs, np.float32), device=DEV)
    xt = torch.zeros(B, 1, 3, device=DEV)
    xt[:, 0, :2] = torch.as_tensor(np.asarray(goals, np.float32), device=DEV)
    xt[:, 0, 2] = TARGET_Z
    both = torch.cat([xb, xt], 1)
    with_t = R3.composite(plate_g, plate_t, cam, make_prims(True), both,
                          eye.expand(B, 2, 3, 3).contiguous(), lig, SIZE, SIZE,
                          supersample=2, tile=None).double()
    without = R3.composite(plate_g, plate_t, cam, make_prims(False), xb,
                           eye.expand(B, 1, 3, 3).contiguous(), lig, SIZE, SIZE,
                           supersample=2, tile=None).double()
    a = ALPHA
    out = (2 * a - a * a) * with_t + (1 - a) ** 2 * without
    return out.round().clamp(0, 255).to(torch.uint8).cpu().numpy()


# ---- states, and the measurement ---------------------------------------------------------------
rng = np.random.default_rng(0)
qs, goals, ref = [], [], []
for i in range(16):
    env.reset(seed=int(rng.integers(1 << 30)))
    qs.append(np.array(d.qpos[:], np.float64))
    goals.append(np.array(u.goal, np.float64))
    ref.append(official(qs[-1], goals[-1]))
ref = np.stack(ref)

ours = render_ours(qs, goals)
diff = np.abs(ours.astype(np.int16) - ref.astype(np.int16)).max(-1)
print(f"\nnominal light (A .4 D .8 S .1 p 40): exact {float((diff==0).mean()):.6f}  "
      f"mean {diff.mean():.4f}  max {diff.max()}")

# where does the residual live?
m.site_rgba[:, 3] = 0.0
plate_only = np.stack([official(q, g) for q, g in zip(qs, goals)])
m.site_rgba[:, 3] = keep
m.site_rgba[TID, 3] = ALPHA
moving = np.abs(ref.astype(np.int16) - plate_only.astype(np.int16)).max(-1) > 0
print(f"  moving-primitive pixels {int(moving.sum())} of {moving.size} "
      f"({100*moving.mean():.3f}%)")
print(f"  static pixels: exact {float((diff[~moving]==0).mean()):.6f} mean {diff[~moving].mean():.4f} "
      f"max {diff[~moving].max()}")
print(f"  moving pixels: exact {float((diff[moving]==0).mean()):.4f} "
      f"within1 {float((diff[moving]<=1).mean()):.4f} mean {diff[moving].mean():.3f} "
      f"max {diff[moving].max()}")

# ---- fit (A, D, S, p) on moving pixels, the way ogbench.py did ---------------------------------
best = None
for A in (0.30, 0.35, 0.40, 0.45, 0.50):
    for D in (0.70, 0.75, 0.80, 0.85, 0.90):
        for S in (0.0, 0.05, 0.1, 0.2):
            for P in (20.0, 40.0, 60.0):
                o = render_ours(qs, goals, A, D, S, P)
                dd = np.abs(o.astype(np.int16) - ref.astype(np.int16)).max(-1)
                sc = (float((dd[moving] <= 1).mean()), -dd[moving].mean())
                if best is None or sc > best[0]:
                    best = (sc, (A, D, S, P), float((dd == 0).mean()), dd.mean(), int(dd.max()))
print(f"\nbest (A,D,S,p) = {best[1]}: moving within1 {best[0][0]:.4f} mean {-best[0][1]:.3f}; "
      f"whole frame exact {best[2]:.6f} mean {best[3]:.4f} max {best[4]}")

# ---- throughput -------------------------------------------------------------------------------
for B in (256, 1024, 4096):
    q = np.tile(qs[0], (B, 1))
    g = np.tile(goals[0], (B, 1))
    render_ours(q, g)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    render_ours(q, g)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"B={B:5d} {B/dt:10,.0f} f/s")

t0 = time.perf_counter()
for i in range(32):
    official(qs[i % 16], goals[i % 16])
print(f"official MuJoCo EGL: {32/(time.perf_counter()-t0):,.0f} f/s")
