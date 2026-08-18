"""Is Meta-World's baked plate a constant of the SCENE, or of the episode?

``probe_backend2.py`` measured the movers-hidden plate bit-identical across 12 states WITHIN one
episode (50/50 tasks), and ``probe_backend.py`` measured it differing by 255 ACROSS episodes. Both
used a mover set built from bodies whose own ``body_pos``/``body_quat`` row a reset writes. That
test is not closed under the kinematic tree: a reset that moves body B also moves every body WELDED
beneath B, and those children keep their (relative) rows unchanged, so they were classified static
and left in the plate.

If closing the mover set under the subtree makes the plate constant across episodes, the port bakes
ONE plate per appearance and the per-episode placement is carried entirely by the ray-cast — the
Fetch design exactly. If it does not, the plate is per-episode and has to be re-baked at every
``reset``, which is a MuJoCo render per episode rather than per appearance.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scratch.metaworld.probe_metaworld import make                          # noqa: E402
from stable_envs.render import mjwarp as MW                                 # noqa: E402

SIZE = 224
OUT = "/mnt/vast/home/randall.balestriero/se_logs/metaworld"


def subtree(m, roots: set[int]) -> set[int]:
    """``roots`` plus every body beneath them, so hiding the set hides everything that moves with it."""
    out = set(roots)
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x in out:
                out.add(b)
                break
            x = int(m.body_parentid[x])
    return out


def probe(task: str, n_ep: int = 6) -> dict:
    env = make(task, SIZE)
    u = env.unwrapped
    m, d = u.model, u.data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "corner")
    r = mujoco.Renderer(m, height=SIZE, width=SIZE)
    o = mujoco.MjvOption()
    o.sitegroup[:] = 0
    rng = np.random.default_rng(0)

    # which body rows does a reset write, over several episodes?
    snaps = []
    for _ in range(n_ep):
        env.reset()
        snaps.append((np.array(m.body_pos, copy=True), np.array(m.body_quat, copy=True)))
    moved = {b for b in range(m.nbody)
             if any(not np.array_equal(snaps[0][0][b], s[0][b])
                    or not np.array_equal(snaps[0][1][b], s[1][b]) for s in snaps)}

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    jointed = {g for g in vis if int(m.body_weldid[m.geom_bodyid[g]]) != 0
               or int(m.body_mocapid[m.geom_bodyid[g]]) >= 0}
    direct = sorted(jointed | {g for g in vis if int(m.geom_bodyid[g]) in moved})
    closed_bodies = subtree(m, moved)
    closed = sorted(jointed | {g for g in vis if int(m.geom_bodyid[g]) in closed_bodies})

    def plate(hide):
        g0 = np.array(m.geom_group, copy=True)
        m.geom_group[np.asarray(hide, np.int64)] = MW.HIDDEN_GROUP
        r.update_scene(d, camera=cam, scene_option=o)
        img = r.render().copy()
        m.geom_group[:] = g0
        return img

    pd, pc = [], []
    for _ in range(n_ep):
        env.reset()
        for _ in range(4):
            env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
        pd.append(plate(direct))
        pc.append(plate(closed))

    def drift(ps):
        a = np.stack(ps)
        e = np.abs(a.astype(np.int16) - a[0].astype(np.int16)).max(-1)
        return {"max": int(e.max()), "frac": round(float((e > 0).mean()), 6)}

    rec = {"task": task, "n_moved_bodies": len(moved), "n_moved_subtree": len(closed_bodies),
           "n_movers_direct": len(direct), "n_movers_subtree_closed": len(closed),
           "plate_across_episodes_direct": drift(pd),
           "plate_across_episodes_subtree_closed": drift(pc)}
    env.close()
    return rec


if __name__ == "__main__":
    tasks = sys.argv[1:]
    if not tasks or tasks == ["ALL"]:
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS
        tasks = sorted(ALL_V3_ENVIRONMENTS)
    out = []
    for t in tasks:
        try:
            rr = probe(t)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            rr = {"task": t, "error": f"{type(ex).__name__}: {ex}"}
        out.append(rr)
        print(json.dumps(rr), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "probe_backend4.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote")
