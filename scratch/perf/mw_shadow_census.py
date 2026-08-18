"""How much of Meta-World's shadow work is CONSTANT across a batch, and could a cache take it?

The shadow ray is 35% of this family's megakernel (``use_shadows=False``: 7.59 -> 4.91 ms at 64 worlds
and 256 px) and the composite's static half is baked by MuJoCo, so the tempting claim is that most of
those rays answer the same question every frame. This separates the pixels four ways and counts them,
because only one of the four classes is cacheable:

  MOVER pixel                      the Sawyer or its manipulandum. Its shadow ray starts somewhere
                                   that moves with ``qpos``; nothing about it is constant.
  STATIC pixel, blocked by STATIC  the table shadowing itself, the wall shadowing the floor. Constant
                                   for a given episode and camera, so the ray IS skippable.
  STATIC pixel, blocked by a MOVER the arm's shadow on the table. This is the term the composite
                                   exists to reproduce and it moves every frame.
  STATIC pixel, unblocked          lit. The ray still has to be cast to find that out, against the
                                   movers, every frame.

Four renders of the same context give all four counts, using ``metaworld.py``'s own trick of
re-pointing a BVH slot at another geom to remove geometry without rebuilding anything:

  A movers on,  shadows on      the shipped frame
  B movers off, shadows on      the plate's ``off``
  C movers off, shadows off
  D movers on,  shadows off
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
import stable_envs.render.mjwarp as MW


def timed(fn, reps=11, warm=3):
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
    ap.add_argument("--envs", default="MetaWorldReach,MetaWorldAssembly,MetaWorldSoccer,"
                                      "MetaWorldWindowOpen")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    a = ap.parse_args()

    import warp as wp

    from stable_envs.envs import metaworld as MWE

    orig = MW.create_render_context
    out = {}
    for name in a.envs.split(","):
        env = se.make(name, batch=8, size=a.size, device="cuda")
        task = type(env).TASK
        frames, times = {}, {}
        for shadows in (True, False):
            def crc(mm, _s=shadows, **kw):
                kw["use_shadows"] = _s
                return orig(mm, **kw)
            MW.create_render_context = crc
            MWE._CTX_CACHE.clear()
            torch.cuda.empty_cache()
            try:
                c = MWE._context(task, a.size, a.nworld)
                times["on" if shadows else "off"] = timed(lambda: c.wp.capture_launch(c.graph))
                c.wp.capture_launch(c.graph)
                c.wp.synchronize()
                frames["A" if shadows else "D"] = c.bgra[:1][..., [2, 1, 0]].clone()
                seg = c.seg[:1].clone()
                # Movers out of the ray-cast AND out of every shadow ray.
                c.bvh_ids[c.bvh_mover_slots] = c.bvh_substitute
                c.wp.capture_launch(c.graph)
                c.wp.synchronize()
                frames["B" if shadows else "C"] = c.bgra[:1][..., [2, 1, 0]].clone()
                c.bvh_ids.copy_(c.bvh_ids0)
                c.wp.synchronize()
                if shadows:
                    lut = torch.zeros(int(env._b.model.ngeom) + 1, dtype=torch.bool,
                                      device=seg.device)
                    lut[torch.as_tensor(np.asarray(env._b.movers, np.int64) + 1,
                                        device=seg.device)] = True
                    mover = lut[(seg + 1).clamp(min=0).long()]
                    hit = seg >= 0
            finally:
                MW.create_render_context = orig
                MWE._CTX_CACHE.clear()
                torch.cuda.empty_cache()

        A, B, Cf, D = frames["A"], frames["B"], frames["C"], frames["D"]
        stat = hit & ~mover
        by_static = stat & (B != Cf).any(-1)                 # the plate shadows itself: SKIPPABLE
        by_mover = stat & (A != B).any(-1) & ~by_static      # the arm's shadow: moves every frame
        lit = stat & ~by_static & ~by_mover                  # still has to be cast to find out
        mv_shadowed = mover & (A != D).any(-1)
        f = lambda t: float(t.to(torch.float32).mean())
        row = {
            "ms_shadows_on": times["on"] * 1e3, "ms_shadows_off": times["off"] * 1e3,
            "shadow_share": 1.0 - times["off"] / times["on"],
            "frac_miss": f(~hit), "frac_mover": f(mover), "frac_static": f(stat),
            "static_blocked_by_static": f(by_static), "static_blocked_by_mover": f(by_mover),
            "static_lit": f(lit), "mover_self_shadowed": f(mv_shadowed),
        }
        shaded = row["frac_mover"] + row["frac_static"]
        row["skippable_share_of_shaded"] = row["static_blocked_by_static"] / max(shaded, 1e-9)
        row["ceiling_if_skippable_free"] = 1.0 / (1.0 - row["shadow_share"]
                                                  * row["skippable_share_of_shaded"])
        row["ceiling_if_all_shadow_free"] = times["on"] / times["off"]
        out[name] = row
        print(f"{name:26s} shadow {100*row['shadow_share']:.1f}% of the kernel | "
              f"miss {row['frac_miss']:.3f} mover {row['frac_mover']:.3f} static {row['frac_static']:.3f}"
              f" | static: by-static {row['static_blocked_by_static']:.4f} "
              f"by-mover {row['static_blocked_by_mover']:.4f} lit {row['static_lit']:.4f} | "
              f"cacheable share of shaded {100*row['skippable_share_of_shaded']:.2f}% -> "
              f"x{row['ceiling_if_skippable_free']:.3f}  (all shadows free: "
              f"x{row['ceiling_if_all_shadow_free']:.2f})", flush=True)
        del env
        torch.cuda.empty_cache()
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
