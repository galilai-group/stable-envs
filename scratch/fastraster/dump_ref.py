"""Freeze what the two families currently produce, so a rewrite can be checked byte for byte.

Two artifacts per env, and they answer different questions:

  * the ``Scene2D`` tensors. A vectorised scene builder must issue the SAME arithmetic, not merely
    equivalent arithmetic -- ``einsum`` over one more axis and ``norm`` over a differently shaped
    tensor are each entitled to a different last bit, and one ulp on a vertex is a pixel wherever
    that vertex sits on a boundary. Comparing the built scene isolates that from the rasterizer.
  * the frame from ``render_torch``, which is the executable specification and is not being touched.
    It is the reference the fused path is diffed against, so freezing it pins the picture even if
    every kernel underneath changes.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.render.raster2d import render_torch

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/se_ref"
HIGHWAY = ["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"]


def dump_highway(name, sizes=(None, 256), B=6):
    env = se.make(name, batch=B, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(7)), device="cuda")
    blob = {"states": st.cpu()}
    for size in sizes:
        scr = env.screen(size)
        sty = env.resolve_style(None, None, batch=B)
        sc = env._scene(st, sty, scr)
        tag = f"{scr.H}x{scr.W}"
        for f in ("kind", "geom", "fill", "line", "line_w", "background"):
            blob[f"{tag}.{f}"] = getattr(sc, f).cpu()
        blob[f"{tag}.world"] = torch.tensor(sc.world, dtype=torch.float64)
        blob[f"{tag}.frame_torch"] = render_torch(sc, (scr.H, scr.W), 1).cpu()
        print(f"  {name} {tag} N={sc.n_prims}", flush=True)
    torch.save(blob, f"{OUT}/{name}.pt")


def dump_carracing(B=6):
    env = se.make("CarRacing", batch=B, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(7)), device="cuda")
    blob = {"states": st.cpu(), "frame": env.render(st).cpu()}
    # a second cell with a non-default style: the visited-tile mask repaints road quads in place and
    # is the one variation this env has that reaches a pixel
    sty = env.sample_style(B, rng=np.random.default_rng(3))
    blob["style"] = torch.as_tensor(sty)
    blob["frame_style"] = env.render(st, style=torch.as_tensor(sty, device="cuda")).cpu()
    torch.save(blob, f"{OUT}/CarRacing.pt")
    print(f"  CarRacing frames {tuple(blob['frame'].shape)}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "highway"):
        for n in HIGHWAY:
            try:
                dump_highway(n)
            except Exception as e:
                print(f"  {n}: {type(e).__name__}: {e}", flush=True)
    if which in ("all", "carracing"):
        dump_carracing()
    print("dumped to", OUT, flush=True)
