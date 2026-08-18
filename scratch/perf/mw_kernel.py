"""Why is Meta-World's megakernel 3.3x Fetch's on a scene of the same size?

7.59 ms per 64 worlds at 256 px against Fetch's 2.29 and HumanoidBench's 2.87, with 13 enabled geoms
against 11, one casting light each and the same resolution. The batch set is not it (every array is
free, ``scratch/perf/batched_cost.py``) and neither is the launch geometry (``block_dim=64`` is the
optimum, ``knobs.py``). What is left is the scene: how many lights are shaded per pixel, how many
triangles a primary ray has to intersect, and how many pixels hit a mesh at all.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se


def timed(fn, reps=15, warm=3):
    import warp as wp
    for _ in range(warm):
        fn()
    wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MetaWorldReach")
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--sizes", default="128,192,256")
    a = ap.parse_args()

    import mujoco

    fam = "metaworld" if a.env.startswith("MetaWorld") else "fetch"
    mod = __import__("stable_envs.envs." + fam, fromlist=["x"])
    env = se.make(a.env, batch=8, size=256, device="cuda")
    key = type(env).TASK if fam == "metaworld" else type(env).SCENE
    b = mod._build_bundle(key)
    m = b.model

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in mod.VISIBLE_GROUPS]
    tris = 0
    for g in vis:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            tris += int(m.mesh_facenum[int(m.geom_dataid[g])])
    facts = {"visible_geoms": len(vis),
             "mesh_geoms": sum(1 for g in vis if int(m.geom_type[g]) == 7),
             "visible_triangles": tris, "nlight": int(m.nlight),
             "light_active": [bool(x) for x in np.asarray(m.light_active).ravel()],
             "light_castshadow": [bool(x) for x in np.asarray(m.light_castshadow).ravel()],
             "light_diffuse": np.asarray(m.light_diffuse).tolist(),
             "headlight_active": bool(m.vis.headlight.active),
             "geom_types": sorted({int(m.geom_type[g]) for g in vis})}
    print("scene: " + json.dumps(facts), flush=True)

    rows = {}
    for size in [int(x) for x in a.sizes.split(",")]:
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()
        c = mod._context(key, size, a.nworld) if fam == "metaworld" else \
            mod._context(key, size, a.nworld)
        dt = timed(lambda: c.wp.capture_launch(c.graph))
        rays = a.nworld * size * size
        rows[f"res{size}"] = {"ms": dt * 1e3, "ns_per_ray": dt * 1e9 / rays}
        print(f"res {size:4d}  {dt*1e3:8.3f} ms  {dt*1e9/rays:7.3f} ns/ray  "
              f"{a.nworld/dt:9,.0f} f/s", flush=True)

    # How many lights are actually shaded, isolated by deactivating all but the casting one.
    mod._CTX_CACHE.clear()
    torch.cuda.empty_cache()
    la = np.array(m.light_active, copy=True)
    cast = [i for i in range(m.nlight) if bool(np.asarray(m.light_castshadow).ravel()[i])]
    try:
        m.light_active[:] = 0
        for i in cast:
            m.light_active[i] = 1
        c = mod._context(key, 256, a.nworld)
        dt = timed(lambda: c.wp.capture_launch(c.graph))
        rows["one_light"] = {"ms": dt * 1e3}
        print(f"only the casting light active: {dt*1e3:8.3f} ms  {a.nworld/dt:9,.0f} f/s", flush=True)
    finally:
        m.light_active[:] = la
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()

    # What fraction of primary rays hit a MESH rather than a plane/box/miss?
    mod._CTX_CACHE.clear()
    c = mod._context(key, 256, a.nworld)
    c.wp.capture_launch(c.graph)
    c.wp.synchronize()
    if hasattr(c, "seg"):
        seg = c.seg[:a.nworld]
        gt = torch.as_tensor(np.asarray(m.geom_type, np.int64), device=seg.device)
        hit = seg >= 0
        types = gt[seg.clamp(min=0).long()]
        frac = {int(t): float(((types == t) & hit).float().mean()) for t in sorted(set(gt.tolist()))}
        frac["miss"] = float((~hit).float().mean())
        rows["hit_type_fraction"] = frac
        print("primary-ray hit type fraction (mjtGeom -> share): " + json.dumps(frac), flush=True)
    print("JSON " + json.dumps({"env": a.env, "scene": facts, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
