"""Phase attribution for the gfxdraw classic-control path.

``profile_note.py`` gives an OPERATOR trace, which answers "which aten op" and not "which part of
the renderer". Section 8 asks for the second, so this wraps the real public path
(``env.render(states)``) in timers placed at the renderer's own seams -- canvas/background,
polygon anti-aliasing (``clip_line`` vs the Wu scatter), the scanline fill mask, the opaque
``paint``, the circle stencil and fill, and the final flip -- and reports the split.

Wrappers only: nothing in ``stable_envs`` is edited, so the numbers describe the baseline.
"""
from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", nargs="+", default=["CartPole", "Acrobot", "Pendulum", "MountainCar"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    import stable_envs as se
    from stable_envs.render import gfxdraw as G

    T: dict[str, float] = collections.defaultdict(float)
    N: dict[str, int] = collections.defaultdict(int)
    cuda = args.device.startswith("cuda")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    class clk:
        def __init__(self, k):
            self.k = k

        def __enter__(self):
            sync()
            self.t = time.perf_counter()

        def __exit__(self, *a):
            sync()
            T[self.k] += time.perf_counter() - self.t
            N[self.k] += 1

    orig = dict(clip_line=G.clip_line, fill_polygon_mask=G.fill_polygon_mask,
                fill_circle_mask=G.fill_circle_mask, aaline=G.aaline,
                polygon=G.polygon, circle=G.circle)
    orig_paint = G.StyleCanvas.paint
    orig_blend_at = G.StyleCanvas.blend_at
    orig_rgb = G.Canvas.rgb
    orig_init = G.StyleCanvas.__init__

    def wrap(name, fn, key):
        def f(*a, **k):
            with clk(key):
                return fn(*a, **k)
        return f

    G.clip_line = wrap("clip_line", orig["clip_line"], "aa:clip_line")
    G.fill_polygon_mask = wrap("fpm", orig["fill_polygon_mask"], "fill:poly_mask")
    G.fill_circle_mask = wrap("fcm", orig["fill_circle_mask"], "fill:circle_mask")

    def aaline(*a, **k):
        with clk("aa:total"):
            return orig["aaline"](*a, **k)
    G.aaline = aaline

    def paint(self, *a, **k):
        with clk("fill:paint_where"):
            return orig_paint(self, *a, **k)
    G.StyleCanvas.paint = paint

    def blend_at(self, *a, **k):
        with clk("aa:blend_at"):
            return orig_blend_at(self, *a, **k)
    G.StyleCanvas.blend_at = blend_at

    def rgb(self, *a, **k):
        with clk("out:flip"):
            return orig_rgb(self, *a, **k)
    G.Canvas.rgb = rgb

    def init(self, *a, **k):
        with clk("canvas:alloc_bg"):
            return orig_init(self, *a, **k)
    G.StyleCanvas.__init__ = init

    print(f"device={args.device} threads={torch.get_num_threads()}")
    for name in args.env:
        npz = f"examples/results/step4/states/{name}.npz"
        pool = np.load(npz)["states"] if os.path.exists(npz) else None
        for B in args.batch:
            e = se.make(name, batch=B, device=args.device)
            src = pool if pool is not None else e.sample_states(64)
            st = np.asarray(src[np.arange(B) % len(src)], np.float32)
            for _ in range(3):
                e.render(st)
            T.clear(); N.clear()
            sync()
            t0 = time.perf_counter()
            for _ in range(args.repeat):
                e.render(st)
            sync()
            tot = (time.perf_counter() - t0) / args.repeat
            rows = sorted(T.items(), key=lambda kv: -kv[1])
            print(f"\n== {name} B={B}  render {tot*1e3:8.2f} ms  {B/tot:9.1f} fps")
            acc = 0.0
            for k, v in rows:
                v /= args.repeat
                if k in ("aa:total",):
                    print(f"   {k:22s} {v*1e3:8.3f} ms {100*v/tot:5.1f}%  "
                          f"calls {N[k]//args.repeat}   (contains clip_line+blend_at)")
                    continue
                acc += v
                print(f"   {k:22s} {v*1e3:8.3f} ms {100*v/tot:5.1f}%  calls {N[k]//args.repeat}")
            print(f"   {'[accounted]':22s} {acc*1e3:8.3f} ms {100*acc/tot:5.1f}%")


if __name__ == "__main__":
    main()
