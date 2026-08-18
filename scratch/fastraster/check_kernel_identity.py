"""New rasterizer vs the pre-tiling kernel, on the FROZEN scenes, at ss=1 and ss=2.

The scenes come out of ``dump_ref.py`` rather than out of the current builder, so this isolates the
kernel: if the two disagree it is the rasterizer and not the scene.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from scratch.fastraster.orig_raster import render_orig
from stable_envs.render.raster2d import render_triton
from stable_envs.render.scene import Scene2D

REF = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/se_ref"
NAMES = ["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"]


def scenes(name):
    blob = torch.load(f"{REF}/{name}.pt", weights_only=True)
    for k in [k for k in blob if k.endswith(".kind")]:
        tag = k[:-5]
        g = lambda f: blob[f"{tag}.{f}"].cuda()                                  # noqa: E731
        H, W = (int(v) for v in tag.split("x"))
        yield tag, H, W, Scene2D(kind=g("kind"), geom=g("geom"), fill=g("fill"), line=g("line"),
                                 line_w=g("line_w"), background=g("background"),
                                 world=tuple(blob[f"{tag}.world"].tolist()))


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    bad = 0
    for name in NAMES:
        for tag, H, W, sc in scenes(name):
            for ss in (1, 2):
                a = render_orig(sc, (H, W), ss)
                b = render_triton(sc, (H, W), ss)
                d = (a.to(torch.int16) - b.to(torch.int16)).abs()
                mx = int(d.max())
                bad += mx != 0
                print(f"  {name:14s} {tag:9s} ss={ss}  max {mx}  "
                      f"px {int((d > 0).any(-1).sum())}/{a.numel()//3}", flush=True)
    print(f"=== {'IDENTICAL' if not bad else str(bad) + ' CELLS DIFFER'} ===", flush=True)
