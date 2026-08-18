"""Are ``pygame.draw.polygon`` and ``gfxdraw.filled_polygon`` the same rasterizer? Three-way diff.

This settles blocker 5.  CarRacing draws through BOTH:

    car_racing.py:784-785      gfxdraw.aapolygon + gfxdraw.filled_polygon   (road, grass, background)
    car_racing.py:704, 731     pygame.draw.polygon                          (indicator panel + bars)
    car_dynamics.py:300, 335   pygame.draw.polygon                          (hull, wheels, stripes)

``stable_envs/render/gfxdraw.py::fill_polygon_mask`` reimplements the FIRST of those.  The question
is whether it also reproduces the second.  Compared here on identical integer vertices:

    A  pygame.draw.polygon
    B  pygame.gfxdraw.filled_polygon
    C  stable_envs.render.gfxdraw.fill_polygon_mask   (the repo's 16.16 reimplementation of B)

If A == B the car can be drawn with the code the repo already has.  If A != B, the car needs a rule
that is not in the repo, and C == B only proves the repo reproduces the road.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pygame
from pygame import gfxdraw
import torch

pygame.init()
W, H = 160, 120


def mask_draw(pts):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), [(float(x), float(y)) for x, y in pts])
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def mask_gfx(pts):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    gfxdraw.filled_polygon(s, [(int(x), int(y)) for x, y in pts], (255, 255, 255))
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def mask_repo(pts, dev):
    from stable_envs.render import gfxdraw as G
    v = torch.tensor([[list(p) for p in pts]], dtype=torch.float64, device=dev)
    vx = torch.trunc(v[..., 0]).to(torch.int64)
    vy = torch.trunc(v[..., 1]).to(torch.int64)
    ys = torch.arange(H, device=dev)
    xs = torch.arange(W, device=dev)
    return G.fill_polygon_mask(vx, vy, ys, xs, W)[0].cpu().numpy()


def convex(pts):
    n = len(pts)
    sg = 0
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cr = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cr:
            if sg and (cr > 0) != (sg > 0):
                return False
            sg = cr
    return True


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"pygame {pygame.version.ver}, device {dev}")

    cases = [([(74, 102), (56, 53), (95, 39), (113, 88)], "the reported rotated rectangle"),
             ([(109, 27), (112, 20), (116, 21), (112, 29)], "quad A"),
             ([(20, 10), (60, 5), (70, 40), (30, 50)], "quad C")]
    rng = np.random.default_rng(0)
    n_rand = 0
    while n_rand < 401:
        c = rng.uniform(20, 140, 2)
        r = rng.uniform(3, 45, 2)
        th = rng.uniform(0, 6.28)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        q = np.array([[-r[0], -r[1]], [r[0], -r[1]], [r[0], r[1]], [-r[0], r[1]]]) @ R.T + c
        pts = [(int(a), int(b)) for a, b in q]
        if not convex(pts):
            continue
        cases.append((pts, f"random {n_rand}"))
        n_rand += 1

    tally = {"A_vs_B": [0, 0, 0], "A_vs_C": [0, 0, 0], "B_vs_C": [0, 0, 0]}
    first = {}
    for pts, tag in cases:
        a, b, c = mask_draw(pts), mask_gfx(pts), mask_repo(pts, dev)
        for k, (u, v) in (("A_vs_B", (a, b)), ("A_vs_C", (a, c)), ("B_vs_C", (b, c))):
            nd = int((u != v).sum())
            tally[k][0] += 1
            tally[k][1] += nd == 0
            tally[k][2] += nd
            if nd and k not in first:
                first[k] = (tag, pts, nd)

    print("\n           A = pygame.draw.polygon   B = pygame.gfxdraw.filled_polygon"
          "   C = stable_envs fill_polygon_mask")
    for k, (n, ok, npx) in tally.items():
        print(f"  {k}: {ok}/{n} polygons identical, {npx} differing pixels"
              + (f"   first: {first[k][0]} {first[k][1]} ({first[k][2]} px)" if k in first else ""))

    pts = cases[0][0]
    a, b, c = mask_draw(pts), mask_gfx(pts), mask_repo(pts, dev)
    print(f"\nrow-by-row for {pts}:")
    print("    y   pygame.draw   gfxdraw.filled   stable_envs")
    for y in range(85, 95):
        def span(m):
            i = np.nonzero(m[y])[0]
            return f"{int(i[0]):3d}..{int(i[-1]):3d}" if i.size else "   --  "
        print(f"  {y:4d}   {span(a)}        {span(b)}         {span(c)}")


if __name__ == "__main__":
    main()
