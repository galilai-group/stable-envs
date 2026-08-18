"""Blocker 2: what does the gfxdraw path actually cost at 1000x800, and what does the official cost?

Three measurements:
  1. the official renderer's frames/s for BOTH render modes (the bar to beat);
  2. how many primitives survive the reference's OWN cull, per frame, over a real rollout;
  3. the repo's ``stable_envs.render.gfxdraw`` primitives at 1000x800 on the GPU -- the per-polygon
     fill, the per-edge Wu pass, and a SPAN-based fill that only touches the pixels a polygon covers,
     at several batch sizes.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import math

import numpy as np
import torch

WINDOW_W, WINDOW_H = 1000, 800


# ------------------------------------------------------------------ official
def official():
    import gymnasium
    print("=== official renderer ===")
    for mode in ("state_pixels", "rgb_array"):
        env = gymnasium.make("CarRacing-v3", render_mode=mode)
        env.reset(seed=0)
        for _ in range(30):
            env.step(np.array([0.0, 0.3, 0.0], np.float32))
        for _ in range(5):
            env.render()
        t0 = time.perf_counter()
        for _ in range(100):
            env.render()
        dt = time.perf_counter() - t0
        print(f"  {mode:13s} {env.render().shape}  {100/dt:8.1f} frames/s "
              f"({1000*dt/100:.3f} ms/frame)")
        env.close()


# ------------------------------------------------------------------ primitive census
def census():
    import gymnasium
    from gymnasium.envs.box2d.car_racing import (GRASS_DIM, MAX_SHAPE_DIM, PLAYFIELD, SCALE, ZOOM)
    import pygame

    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=0)
    u = env.unwrapped
    grass = []
    for x in range(-20, 20, 2):
        for y in range(-20, 20, 2):
            grass.append([(GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                          (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                          (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                          (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM)])

    def visible(poly, zoom, trans, angle):
        p = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        p = [(c[0] * zoom + trans[0], c[1] * zoom + trans[1]) for c in p]
        return any((-MAX_SHAPE_DIM <= c[0] <= WINDOW_W + MAX_SHAPE_DIM)
                   and (-MAX_SHAPE_DIM <= c[1] <= WINDOW_H + MAX_SHAPE_DIM) for c in p)

    print("\n=== primitive census (seed 0) ===")
    print(f"  road_poly {len(u.road_poly)}  track tiles {len(u.track)}  grass {len(grass)}"
          f"  MAX_SHAPE_DIM {MAX_SHAPE_DIM:.1f} px  ZOOM*SCALE {ZOOM*SCALE:.1f} px/m"
          f"  PLAYFIELD {PLAYFIELD:.1f} m")
    gs, rs = [], []
    for step in range(300):
        env.step(np.array([0.0, 0.5, 0.0], np.float32))
        angle = -u.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - u.t, 0) + ZOOM * SCALE * min(u.t, 1)
        sx, sy = -u.car.hull.position[0] * zoom, -u.car.hull.position[1] * zoom
        tr = pygame.math.Vector2((sx, sy)).rotate_rad(angle)
        tr = (WINDOW_W / 2 + tr[0], WINDOW_H / 4 + tr[1])
        if step % 20 == 0:
            gs.append(sum(visible(p, zoom, tr, angle) for p in grass))
            rs.append(sum(visible(p, zoom, tr, angle) for p, _ in u.road_poly))
    print(f"  visible grass quads  min {min(gs)} med {int(np.median(gs))} max {max(gs)}")
    print(f"  visible road quads   min {min(rs)} med {int(np.median(rs))} max {max(rs)}")
    print(f"  drawn per frame = 1 bg + grass + road + 12 car + 8 indicators"
          f" ~= {1 + int(np.median(gs)) + int(np.median(rs)) + 20}")
    env.close()


# ------------------------------------------------------------------ GPU primitives
def gpu(dev="cuda"):
    from stable_envs.render import gfxdraw as G
    print("\n=== stable_envs.render.gfxdraw at 1000x800 (H200) ===")
    g = G.Gfx(WINDOW_H, WINDOW_W, dev)
    col = g.color((102, 204, 102))

    def bench(fn, iters=5, warmup=2):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    for B in (1, 8, 32, 128):
        cv = g.canvas(B, (102, 204, 102))
        bi = g.batch_index(B)
        rng = torch.Generator(device=dev).manual_seed(0)
        ctr = torch.rand((B, 1, 2), generator=rng, device=dev, dtype=torch.float64) * 1000
        loc = torch.tensor([[-135., -135.], [135., -135.], [135., 135.], [-135., 135.]],
                           dtype=torch.float64, device=dev).view(1, 4, 2)
        verts = (ctr + loc)

        t_fill = bench(lambda: cv.paint(G.fill_polygon_mask(
            torch.trunc(verts)[..., 0].to(torch.int64), torch.trunc(verts)[..., 1].to(torch.int64),
            cv.ys, cv.xs, cv.W), (102, 204, 102)))
        vi = torch.trunc(verts).to(torch.int64)
        t_aa = bench(lambda: G.aaline(cv, vi[:, 0, 0], vi[:, 0, 1], vi[:, 1, 0], vi[:, 1, 1],
                                      col, 300, bi))
        # span-based fill: only the covered pixels are written, via run-length decode + scatter
        t_span = bench(lambda: span_fill(cv, vi, B, dev))
        print(f"  B={B:4d}  mask-fill/poly {1000*t_fill:8.3f} ms   aaline/edge {1000*t_aa:7.3f} ms"
              f"   span-fill/poly {1000*t_span:7.3f} ms")
        n_poly, n_edge = 200, 800
        naive = n_poly * t_fill + n_edge * t_aa
        spanned = n_poly * t_span + n_edge * t_aa
        print(f"          200 polys + 800 edges: mask {B/naive:9.1f} f/s    span {B/spanned:9.1f} f/s")


def span_fill(cv, vi, B, dev):
    """Paint one convex quad by decoding its per-row spans, not by an (B,H,W) mask."""
    vx, vy = vi[..., 0], vi[..., 1]
    ys = cv.ys
    ay, by = vy.roll(1, 1), vy
    ax, bx = vx.roll(1, 1), vx
    swap = ay > by
    ylo = torch.where(swap, by, ay).unsqueeze(-1)
    yhi = torch.where(swap, ay, by).unsqueeze(-1)
    xlo = torch.where(swap, bx, ax).unsqueeze(-1)
    xhi = torch.where(swap, ax, bx).unsqueeze(-1)
    live = (ay != by).unsqueeze(-1)
    maxy = vy.max(1, keepdim=True).values.unsqueeze(-1)
    y = ys.view(1, 1, -1)
    act = live & (((y >= ylo) & (y < yhi)) | ((y == maxy) & (y > ylo) & (y <= yhi)))
    den = (yhi - ylo).clamp_min(1)
    X = torch.div(65536 * (y - ylo), den, rounding_mode="floor") * (xhi - xlo) + 65536 * xlo
    BIG = 1 << 40
    xa = ((torch.where(act, X, BIG).amin(1) + 1) >> 16).clamp_min(0)
    xb = ((torch.where(act, X, -BIG).amax(1) - 1) >> 16).clamp_max(cv.W - 1)
    ok = act.sum(1) >= 2
    n = torch.where(ok, (xb - xa + 1).clamp_min(0), torch.zeros_like(xa))        # (B,H)
    tot = int(n.sum())
    if tot == 0:
        return
    flat_n = n.reshape(-1)
    idx = torch.repeat_interleave(torch.arange(flat_n.numel(), device=dev), flat_n)
    off = torch.cumsum(flat_n, 0) - flat_n
    k = torch.arange(tot, device=dev) - off[idx]
    row = idx % cv.H
    b = idx // cv.H
    x = xa.reshape(-1)[idx] + k
    flat = (b * cv.H + row) * cv.W + x
    cv.flat.view(-1, 3)[flat] = torch.tensor([102, 204, 102], dtype=torch.uint8, device=dev)


if __name__ == "__main__":
    official()
    census()
    if torch.cuda.is_available():
        gpu()
