"""Diff the current build against the frozen artifacts from ``dump_ref.py``.

Every ``Scene2D`` array is compared BYTE for byte, not to a tolerance. A vectorised builder is only
worth having if it issues the same arithmetic, and a tolerance would hide the one failure mode that
matters: an ulp on a vertex that happens to sit on a pixel boundary is a whole pixel in the frame and
nothing at all in a mean absolute error.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.render.raster2d import render_torch

REF = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/se_ref"
HIGHWAY = ["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"]


def cmp(tag, got, want):
    g, w = got.cpu(), want.cpu()
    if g.shape != w.shape:
        print(f"    {tag:24s} SHAPE {tuple(g.shape)} vs {tuple(w.shape)}", flush=True)
        return False
    if torch.equal(g, w):
        return True
    d = (g.to(torch.float64) - w.to(torch.float64)).abs()
    print(f"    {tag:24s} DIFFERS max {float(d.max()):.6g} on {int((d > 0).sum())}/{d.numel()} "
          f"elements", flush=True)
    return False


def check_highway(name):
    blob = torch.load(f"{REF}/{name}.pt", weights_only=True)
    st = blob["states"].cuda()
    B = st.shape[0]
    env = se.make(name, batch=B, device="cuda")
    ok = True
    for size in (None, 256):
        scr = env.screen(size)
        tag = f"{scr.H}x{scr.W}"
        sty = env.resolve_style(None, None, batch=B)
        sc = env._scene(st, sty, scr)
        for f in ("kind", "geom", "fill", "line", "line_w", "background"):
            ok &= cmp(f"{tag}.{f}", getattr(sc, f), blob[f"{tag}.{f}"])
        ok &= cmp(f"{tag}.frame_torch", render_torch(sc, (scr.H, scr.W), 1),
                  blob[f"{tag}.frame_torch"])
        ok &= cmp(f"{tag}.frame_triton", env.render(st, size=size), blob[f"{tag}.frame_torch"])
    print(f"  {name}: {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def check_carracing():
    blob = torch.load(f"{REF}/CarRacing.pt", weights_only=True)
    st = blob["states"].cuda()
    env = se.make("CarRacing", batch=st.shape[0], device="cuda")
    ok = cmp("frame", env.render(st), blob["frame"])
    ok &= cmp("frame_style", env.render(st, style=blob["style"].cuda()), blob["frame_style"])
    print(f"  CarRacing: {'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def speed(names, B=1024, size=256):
    print("\n=== end to end at 256px, batch 1024 ===", flush=True)
    for name in names:
        env = se.make(name, batch=1, size=size, device="cuda")
        st = torch.as_tensor(env.sample_states(B), device="cuda")
        scr = env.screen(size)
        sty = env.resolve_style(None, None, batch=B)
        out = torch.empty((B, scr.H, scr.W, 3), dtype=torch.uint8, device="cuda")
        for fn, lab in ((lambda: env.render(st, out=out), "all"),
                        (lambda: env._scene(st, sty, scr), "scene")):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(9):
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = float(np.median(ts))
            if lab == "all":
                print(f"  {name:14s} {scr.H}x{scr.W}  all {1000*t:8.2f} ms  {B/t:9.1f} f/s", end="",
                      flush=True)
            else:
                print(f"   scene {1000*t:7.2f} ms", flush=True)
        del env, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    bad = []
    if which in ("all", "highway"):
        for n in HIGHWAY:
            try:
                if not check_highway(n):
                    bad.append(n)
            except Exception as e:
                import traceback
                traceback.print_exc()
                bad.append(n)
        speed(HIGHWAY)
    if which in ("all", "carracing"):
        try:
            if not check_carracing():
                bad.append("CarRacing")
        except Exception:
            import traceback
            traceback.print_exc()
            bad.append("CarRacing")
    print(f"\n=== {'ALL MATCH' if not bad else 'MISMATCH: ' + ','.join(bad)} ===", flush=True)
