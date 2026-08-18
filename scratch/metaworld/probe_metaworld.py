"""Meta-World: what is in the scene, and is the frame a function of a fixed-width state?

Three questions, in the order that decides whether a port is possible at all:

  1. **Geometry.** ``nmesh`` per task, and how much of the frame mesh geoms cover. The analytic
     ray-caster in this repo intersects planes, spheres, capsules, cylinders and boxes in closed
     form; a mesh has no closed form and is not a "primitive-only scene".
  2. **State.** Is the frame a function of ``qpos``? Meta-World writes the task's goal and object
     placement into the MODEL (``body_pos``/``site_pos``) at ``reset``, not into ``qpos``, so the
     same ``qpos`` in two episodes may draw two different pictures. Measured directly: render the
     same qpos under two tasks of the same env and diff.
  3. **Textures.** Which geoms carry a texture, since a textured non-plane/non-mesh geom is the one
     case even MJWarp cannot sample.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import gymnasium as gym
import mujoco

GEOM_NAMES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder",
              6: "box", 7: "mesh", 8: "sdf"}


def make(task: str, size: int = 128, camera: str = "corner"):
    """A single Meta-World task. The gymnasium ids Meta-World registers are BENCHMARKS (MT1/ML1 are
    vector envs), so a single task comes from the class table, which is what the benchmark wrappers
    instantiate anyway."""
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS
    env = ALL_V3_ENVIRONMENTS[task](
        render_mode="rgb_array", width=size, height=size, camera_name=camera)
    # what `_create_observable_goal_envs` does: a task must be set before the env will reset, and
    # the goal is drawn per reset rather than frozen at construction.
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    return env


def probe(task: str, size: int = 128) -> dict:
    env = make(task, size)
    u = env.unwrapped
    env.reset(seed=0)
    m, d = u.model, u.data
    rec = {"task": task, "nq": int(m.nq), "nv": int(m.nv), "ngeom": int(m.ngeom),
           "nmesh": int(m.nmesh), "ntex": int(m.ntex), "nmat": int(m.nmat), "ncam": int(m.ncam),
           "nlight": int(m.nlight), "nsite": int(m.nsite), "nbody": int(m.nbody),
           "geom_types": {}, "textured_nonmesh": []}
    for g in range(m.ngeom):
        t = GEOM_NAMES.get(int(m.geom_type[g]), str(int(m.geom_type[g])))
        rec["geom_types"][t] = rec["geom_types"].get(t, 0) + 1
        mid = int(m.geom_matid[g])
        if mid >= 0 and int(m.geom_type[g]) not in (0, 7):
            tid = m.mat_texid[mid]
            tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
            if tid >= 0:
                rec["textured_nonmesh"].append(
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}")

    # what fraction of the rendered frame is mesh?
    r = mujoco.Renderer(m, height=size, width=size)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera="corner")
    s = r.render()
    r.disable_segmentation_rendering()
    seg = np.where(s[..., 1].astype(np.int64) == int(mujoco.mjtObj.mjOBJ_GEOM),
                   s[..., 0].astype(np.int64), -1)
    ok = seg >= 0
    types = np.zeros_like(seg)
    types[ok] = np.asarray(m.geom_type)[seg[ok]]
    rec["frac_mesh_pixels"] = float((ok & (types == 7)).mean())
    rec["frac_background"] = float((~ok).mean())

    # STATE: same qpos, two episodes (i.e. two goal draws) -> same pixels?
    def render_now():
        return u.render().copy() if hasattr(u, "render") else env.render().copy()

    env.reset(seed=0)
    q = np.array(d.qpos, copy=True)
    v = np.zeros(m.nv)
    d.qpos[:] = q
    d.qvel[:] = v
    mujoco.mj_forward(m, d)
    a = render_now()
    goals = []
    diffs = []
    for sd in (1, 2, 3):
        env.reset(seed=sd)
        goals.append([float(x) for x in np.asarray(getattr(u, "_target_pos", [np.nan] * 3))])
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_forward(m, d)
        b = render_now()
        diffs.append(int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))
    rec["same_qpos_across_episodes_maxdiff"] = diffs
    rec["target_pos_per_episode"] = goals
    rec["model_body_pos_changes"] = None
    env.close()
    return rec


if __name__ == "__main__":
    import metaworld  # noqa: F401  (registers the gymnasium ids)
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS, ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    tasks = sys.argv[1:] or sorted(ALL_V3_ENVIRONMENTS)
    print(f"{len(ALL_V3_ENVIRONMENTS)} v3 tasks, "
          f"{len(ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE)} goal-observable", flush=True)
    out = []
    for t in tasks:
        try:
            r = probe(t)
        except Exception as e:                                           # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"task": t, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_metaworld.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
