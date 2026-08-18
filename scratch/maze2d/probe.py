"""D4RL maze2d / gymnasium-robotics PointMaze: the four gates, before any renderer is written."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import mujoco

gym.register_envs(gymnasium_robotics)

IDS = ["PointMaze_UMaze-v3", "PointMaze_Medium-v3", "PointMaze_Large-v3", "PointMaze_Open-v3"]


def build(eid, size=224):
    return gym.make(eid, render_mode="rgb_array", width=size, height=size,
                    continuing_task=False)


def describe(eid):
    e = build(eid)
    e.reset(seed=0)
    u = e.unwrapped
    m = u.model
    print(f"\n=== {eid}")
    print(f"  nq={m.nq} nv={m.nv} ngeom={m.ngeom} nmesh={m.nmesh} nsite={m.nsite} "
          f"ncam={m.ncam} nlight={m.nlight} ntex={m.ntex} nmat={m.nmat}")
    print("  geoms:", [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                        int(m.geom_type[g])) for g in range(min(m.ngeom, 8))], "...")
    print("  sites:", [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s), int(m.site_type[s]),
                        m.site_size[s].tolist(), m.site_rgba[s].tolist())
                       for s in range(m.nsite)])
    print("  joints:", [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_type[j]))
                        for j in range(m.njnt)])
    cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)]
    print("  cameras:", cams, " default_camera_config:",
          getattr(u.point_env.mujoco_renderer, "default_cam_config", None))
    print("  maze_size_scaling:", u.maze.maze_size_scaling, " map rows:", len(u.maze.maze_map))
    return e, u, m


def gates(eid, size=224):
    e, u, m = describe(eid)
    frame = e.render()
    print(f"  frame {frame.shape} {frame.nbytes} B; unique colours {len(np.unique(frame.reshape(-1,3), axis=0))}")

    # ---- G1: is the frame a function of (ball qpos, goal xy)?
    def render_at(env, q, goal, qvel=None):
        uu = env.unwrapped
        uu.data.qpos[:] = np.asarray(q, np.float64)
        uu.data.qvel[:] = np.zeros(uu.model.nv) if qvel is None else qvel
        uu.goal = np.asarray(goal, np.float64)
        uu.update_target_site_pos()
        mujoco.mj_forward(uu.model, uu.data)
        return uu.point_env.mujoco_renderer.render("rgb_array").copy()

    rng = np.random.default_rng(0)
    cells = np.argwhere(np.array(u.maze.maze_map, dtype=object) == 0)
    qs, goals = [], []
    for _ in range(8):
        obs, _ = e.reset(seed=int(rng.integers(1 << 30)))
        qs.append(np.array(u.data.qpos[:], np.float64))
        goals.append(np.array(u.goal, np.float64))

    worst_hist, worst_vel = 0, 0
    a = [render_at(e, q, g) for q, g in zip(qs, goals)]
    b = [render_at(e, q, g) for q, g in zip(qs[::-1], goals[::-1])][::-1]
    for x, y in zip(a, b):
        worst_hist = max(worst_hist, int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()))
    for q, g in zip(qs, goals):
        x = render_at(e, q, g, np.zeros(m.nv))
        y = render_at(e, q, g, rng.normal(0, 100, m.nv))
        worst_vel = max(worst_vel, int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()))
    print(f"  G1: same (qpos, goal) from a different history -> max abs diff {worst_hist}; "
          f"qvel x100 -> {worst_vel}")

    # a fresh env instance, same state
    e2 = build(eid, size)
    e2.reset(seed=999)
    worst_inst = max(int(np.abs(render_at(e2, q, g).astype(np.int16) - x.astype(np.int16)).max())
                     for (q, g), x in zip(zip(qs, goals), a))
    print(f"  G1: a FRESH env instance at the same state -> max abs diff {worst_inst}")

    # ---- G2
    sd = m.nq + 2
    print(f"  G2: state {sd} float32 = {4*sd} B vs frame {frame.nbytes} B -> "
          f"{frame.nbytes/(4*sd):.0f}x")

    # ---- G3: model shape across resets and across instances
    shapes = set()
    for s in range(4):
        e.reset(seed=s)
        shapes.add((u.model.nq, u.model.ngeom, u.model.nmesh, u.model.nsite))
    shapes.add((e2.unwrapped.model.nq, e2.unwrapped.model.ngeom, e2.unwrapped.model.nmesh,
                e2.unwrapped.model.nsite))
    print(f"  G3: (nq, ngeom, nmesh, nsite) across 4 resets and 2 instances: {shapes}")

    # what does the camera look like, and is it fixed?
    cam = e.unwrapped.point_env.mujoco_renderer._get_viewer("rgb_array").cam
    print(f"  camera: type {cam.type} lookat {np.array(cam.lookat).tolist()} dist {cam.distance:.4f} "
          f"az {cam.azimuth} elev {cam.elevation}")
    return e


for eid in IDS:
    try:
        gates(eid)
    except Exception as e:                                            # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"{eid}: FAILED {type(e).__name__}: {e}")

# is d4rl's own maze2d installable/available at all?
try:
    import d4rl  # noqa: F401
    print("\nd4rl import: OK")
except Exception as e:                                                # noqa: BLE001
    print(f"\nd4rl import: {type(e).__name__}: {e}")
