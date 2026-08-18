"""Does rendering only the box the movers occupy actually cost proportionally less?

``scratch/perf/roi2.py`` says the dynamic half of a split decides 16-20% of a RoboCasa frame inside a
box covering 31-33% of it per state and 52% unioned over a chunk, and 5-8% of a HumanoidBench frame
inside 23-28% / 54-65%. Whether that box is worth launching is NOT the box's area: the rays a window
removes are the ones that hit nothing, which are the cheap ones, so a 2x cut in rays can easily be a
1.2x cut in time. This measures it instead of assuming it.

``use_precomputed_rays=True`` makes the window a data change rather than a code change: ``rc.ray`` is
a flat array of per-pixel directions the megakernel indexes by ray id, so a context built at
``cam_res=(W,H)`` whose ``rc.ray`` is GATHERED from the full-resolution table at ``(y0+j, x0+i)``
renders exactly that sub-window, with the same directions to the bit. The script asserts that: the
windowed frame must equal the full frame's crop, pixel for pixel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se
import stable_envs.render.mjwarp as MW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_static import build, family, model_of, module_for, moving_geoms  # noqa: E402


def timed(fn, reps, warm=3):
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
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--windows", default="0.9,0.75,0.6,0.5,0.35,0.25",
                    help="side fractions of the full frame to try")
    ap.add_argument("--keep", default="dyn", choices=["dyn", "full"],
                    help="'dyn': movers only in the BVH. 'full': the whole scene, which is what a "
                         "shadow-carrying family needs inside the box.")
    a = ap.parse_args()

    import warp as wp

    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    m = model_of(mod, env)
    fam = family(a.env)
    g0 = np.array(m.geom_group, copy=True)
    vis_groups = set(int(g) for g in mod.VISIBLE_GROUPS)
    mv, _ = moving_geoms(mod, env, m, k=16)
    mv_vis = set(int(g) for g in mv if int(g0[g]) in vis_groups)
    hide = np.array([g for g in range(m.ngeom)
                     if int(g0[g]) in vis_groups and int(g) not in mv_vis], np.int64) \
        if a.keep == "dyn" else np.array([], np.int64)

    rows = {}
    full_ray = None
    ref = None

    def ctx(side, rays=None):
        """A context whose CAMERA IS ``side`` px, then given the sub-window's ray directions.

        The context is built at ``res = side`` rather than by overriding ``cam_res``, because every
        family's ``_context`` reshapes ``rc.rgb_data`` to ``(nworld, res, res, 4)`` from the same
        ``res`` it passes as ``cam_res`` -- overriding one and not the other raises inside the
        family's own constructor.
        """
        nonlocal full_ray, ref
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()
        m.geom_group[:] = g0
        if len(hide):
            m.geom_group[hide] = MW.HIDDEN_GROUP
        try:
            c = build(mod, env, side, a.nworld)
            if rays is not None:
                wp.to_torch(c.rc.ray).copy_(rays)
                c.wp.capture_launch(c.graph)
                c.wp.synchronize()
            dt = timed(lambda: c.wp.capture_launch(c.graph), a.reps)
            frame = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(
                a.nworld, side, side, 4)[..., [2, 1, 0]].clone()
            if full_ray is None:
                full_ray = wp.to_torch(c.rc.ray).clone()
                ref = frame
            return dt, frame
        finally:
            m.geom_group[:] = g0
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()

    S = a.size
    dt, _ = ctx(S)
    rows["full"] = {"ms": dt * 1e3, "fps": a.nworld / dt, "side": S, "ray_frac": 1.0}
    print(f"{'full':10s} {S:4d}px  {dt*1e3:8.3f} ms  {a.nworld/dt:10,.0f} f/s", flush=True)

    for frac in [float(x) for x in a.windows.split(",")]:
        side = max(8, int(round(S * frac)) // 8 * 8)
        x0 = (S - side) // 2
        y0 = (S - side) // 2
        idx = ((torch.arange(side, device=full_ray.device) + y0).view(-1, 1) * S
               + (torch.arange(side, device=full_ray.device) + x0).view(1, -1)).reshape(-1)
        sub = full_ray.index_select(0, idx).contiguous()
        dt, frame = ctx(side, rays=sub)
        crop = ref[:, y0:y0 + side, x0:x0 + side]
        d = int((frame.to(torch.int16) - crop.to(torch.int16)).abs().max())
        rf = side * side / float(S * S)
        rows[f"win{side}"] = {"ms": dt * 1e3, "fps": a.nworld / dt, "side": side, "ray_frac": rf,
                              "maxdiff_vs_crop": d, "speedup": rows["full"]["ms"] / (dt * 1e3)}
        print(f"{'win'+str(side):10s} {side:4d}px  {dt*1e3:8.3f} ms  {a.nworld/dt:10,.0f} f/s  "
              f"rays {rf:.3f}  x{rows['full']['ms']/(dt*1e3):5.2f}  maxdiff {d}", flush=True)

    print("JSON " + json.dumps({"env": a.env, "keep": a.keep, "size": S, "nworld": a.nworld,
                                "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
