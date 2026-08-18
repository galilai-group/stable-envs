"""Does RoboCasa's static/dynamic split reproduce the fused render, and what does it buy?

The split claims a pixel's colour is a function of its closest hit alone -- no shadow-casting light,
no secondary rays -- so drawing the room and the robot in two passes and keeping the nearer hit is a
rearrangement of the same arithmetic. This drives the FUSED context (``part="all"``, the code this
family shipped before) over the same states and compares it against ``env.render`` pixel for pixel,
then times both.

Whatever disagreement is left is a DEPTH TIE: two surfaces at exactly the same distance, which the
fused render settles by BVH visit order and the composite settles by preferring the plate. The script
reports how many pixels and how far, which is what belongs in ``visual_note``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import stable_envs as se


def timed(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def fused(env, mod, states, res, chunk):
    """The pre-split path: one context over the whole kitchen, driven exactly as ``render`` drove it."""
    B = len(states)
    out = torch.empty((B, res, res, 3), dtype=torch.uint8, device="cuda")
    q = env._full_qpos(torch.as_tensor(states, device="cuda", dtype=torch.float32))
    sty = env.resolve_style(None, None, batch=B)
    off = sty["camera.offset"].reshape(B, 3).cpu().numpy().astype(np.float64)
    pan = sty["camera.pan"].reshape(B, 1).cpu().numpy().astype(np.float64)
    tilt = sty["camera.tilt"].reshape(B, 1).cpu().numpy().astype(np.float64)
    cp, cq = env._camera_pose(off, pan, tilt)
    cp = torch.as_tensor(cp, device="cuda", dtype=torch.float32)
    cq = torch.as_tensor(cq, device="cuda", dtype=torch.float32)
    bl = torch.as_tensor(np.asarray(env._s.model.light_diffuse, np.float64), device="cuda",
                         dtype=torch.float32)
    light = (bl.unsqueeze(0) * sty["light.scale"].reshape(B, 1, 1)
             * sty["light.tint"].reshape(B, 1, 3))
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        n = hi - lo
        nw = 1 if n <= 1 else min(chunk, 1 << (n - 1).bit_length())
        c = mod._context(type(env).KITCHEN, type(env).CAMERA, res, nw, part="all")
        wp = c.wp
        wp.to_torch(c.wd.qpos)[:n] = q[lo:hi]
        wp.to_torch(c.wm.cam_pos)[:n, c.cam] = cp[lo:hi]
        wp.to_torch(c.wm.cam_quat)[:n, c.cam] = cq[lo:hi]
        wp.to_torch(c.wm.light_diffuse)[:n] = light[lo:hi]
        torch.cuda.synchronize()
        wp.capture_launch(c.graph)
        wp.synchronize()
        out[lo:hi] = c.bgra[:n][..., [2, 1, 0]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="RoboCasaCounterToSink,RoboCasaCounterToSinkRight,"
                                      "RoboCasaCounterToSinkWrist,RoboCasaCounterToSinkL45S33")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n", type=int, default=64, help="states for the pixel comparison")
    a = ap.parse_args()

    from stable_envs.envs import robocasa as RC

    res = {}
    for name in a.envs.split(","):
        try:
            env = se.make(name, batch=a.batch, size=a.size, device="cuda")
            st = env.sample_states(a.n)
            ours = env.render(st).clone()
            ref = fused(env, RC, st, a.size, RC.MJW_CHUNK)
            d = (ours.to(torch.int16) - ref.to(torch.int16)).abs()
            worst = d.amax(-1)
            row = {
                "n_movers": int(len(RC._mover_geoms(type(env).KITCHEN, type(env).CAMERA, a.size))),
                "max": int(d.max()), "mean": float(d.to(torch.float32).mean()),
                "exact_frac": float((worst == 0).to(torch.float32).mean()),
                "within8_frac": float((worst <= 8).to(torch.float32).mean()),
                "bad_px_per_frame": float((worst > 0).to(torch.float32).sum() / len(st)),
            }
            pool = env.sample_states(min(512, a.batch))
            big = torch.as_tensor(np.resize(pool, (a.batch, pool.shape[1])), device="cuda")
            out = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
            dt_new = timed(lambda: env.render(big, out=out))
            dt_old = timed(lambda: fused(env, RC, big.cpu().numpy(), a.size, RC.MJW_CHUNK), reps=3,
                           warm=1)
            row.update(fps_split=a.batch / dt_new, fps_fused=a.batch / dt_old,
                       speedup=dt_old / dt_new, ratio_vs_8worker_jpeg=a.batch / dt_new / 16889.0)
            res[name] = row
            print(f"{name:32s} split {a.batch/dt_new:9,.0f} f/s  fused {a.batch/dt_old:9,.0f} f/s  "
                  f"x{dt_old/dt_new:4.2f}   pixels: max {row['max']} exact {row['exact_frac']:.6f} "
                  f"({row['bad_px_per_frame']:.0f} px/frame)", flush=True)
            del env, out, big, ours, ref
            RC._CTX_CACHE.clear()
            RC._PLATE_CACHE.clear()
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            res[name] = {"error": f"{type(e).__name__}: {e}"}
    print("JSON " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
