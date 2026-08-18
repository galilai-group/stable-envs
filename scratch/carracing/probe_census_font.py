"""Primitive census over the states the env actually reaches, the font atlas, and the AA cost bound.

(a) census: place the car on random track tiles (the distribution ``sample_states`` would use) and
    count how many primitives survive the reference's OWN cull.
(b) font: is the score text a composition of per-character glyphs, i.e. a finite atlas indexed by
    the state, or does the renderer kern it?
(c) AA: the Wu pass in ``stable_envs/render/gfxdraw`` costs 6.4 ms PER EDGE in eager mode
    (probe_throughput), batch-independent -- that is python op overhead, not pixels.  Measure the
    same work with every edge in ONE call to find the floor.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch

W, H = 1000, 800


def census():
    import gymnasium
    import pygame
    from gymnasium.envs.box2d.car_racing import GRASS_DIM, MAX_SHAPE_DIM, SCALE, ZOOM

    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=0)
    u = env.unwrapped
    grass = [[(GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y), (GRASS_DIM * x, GRASS_DIM * y),
              (GRASS_DIM * x, GRASS_DIM * y + GRASS_DIM),
              (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM)]
             for x in range(-20, 20, 2) for y in range(-20, 20, 2)]

    def vis(poly, zoom, tr, ang):
        p = [pygame.math.Vector2(c).rotate_rad(ang) for c in poly]
        p = [(c[0] * zoom + tr[0], c[1] * zoom + tr[1]) for c in p]
        return any((-MAX_SHAPE_DIM <= c[0] <= W + MAX_SHAPE_DIM)
                   and (-MAX_SHAPE_DIM <= c[1] <= H + MAX_SHAPE_DIM) for c in p)

    print("=== primitive census over the ON-TRACK state distribution (seed 0) ===")
    rng = np.random.default_rng(0)
    gs, rs = [], []
    for _ in range(64):
        beta, x, y = u.track[int(rng.integers(len(u.track)))][1:4]
        ang = -(-beta)                                     # hull.angle = beta at reset
        zoom = ZOOM * SCALE
        tr = pygame.math.Vector2((-x * zoom, -y * zoom)).rotate_rad(-beta)
        tr = (W / 2 + tr[0], H / 4 + tr[1])
        gs.append(sum(vis(p, zoom, tr, -beta) for p in grass))
        rs.append(sum(vis(p, zoom, tr, -beta) for p, _ in u.road_poly))
    print(f"  road_poly total {len(u.road_poly)}, grass total {len(grass)}")
    print(f"  visible grass: min {min(gs)} median {int(np.median(gs))} max {max(gs)}")
    print(f"  visible road : min {min(rs)} median {int(np.median(rs))} max {max(rs)}")
    print(f"  drawn/frame  = 1 bg + grass + road + 12 car + <=8 indicators, median "
          f"{1 + int(np.median(gs)) + int(np.median(rs)) + 20}, worst "
          f"{1 + max(gs) + max(rs) + 20}")
    env.close()


def font_atlas():
    import pygame
    pygame.font.init()
    f = pygame.font.Font(pygame.font.get_default_font(), 42)
    print(f"\n=== score text: font {pygame.font.get_default_font()} size 42 ===")
    sizes = {c: f.size(c) for c in "0123456789-"}
    print(f"  per-char size: {sizes}")
    whole = f.render("1234", True, (255, 255, 255), (0, 0, 0))
    print(f"  render('1234') size {whole.get_size()}   sum of char widths "
          f"{sum(sizes[c][0] for c in '1234')}")

    def arr(s):
        buf = pygame.Surface(s.get_size(), depth=32)
        buf.fill((0, 0, 0))
        buf.blit(s, (0, 0))
        return np.transpose(np.array(pygame.surfarray.pixels3d(buf)), (1, 0, 2))

    # Are the glyphs at fixed positions inside the rendered string?  Substitute one character at a
    # time and see whether the pixels that move stay inside one column band.
    print("  --- per-position substitution, 4-digit strings ---")
    base = arr(f.render("0000", True, (255, 255, 255), (0, 0, 0)))
    bands = []
    for pos in range(4):
        lo, hi = 10**9, -1
        for d in "123456789":
            t = list("0000")
            t[pos] = d
            a = arr(f.render("".join(t), True, (255, 255, 255), (0, 0, 0)))
            if a.shape != base.shape:
                print(f"    pos {pos} digit {d}: SIZE CHANGES {a.shape} vs {base.shape}")
                continue
            c = np.nonzero((np.abs(a.astype(np.int16) - base.astype(np.int16)).max(-1) > 0).any(0))[0]
            if c.size:
                lo, hi = min(lo, int(c[0])), max(hi, int(c[-1]))
        bands.append((lo, hi))
        print(f"    pos {pos}: changed columns {lo}..{hi}")
    disjoint = all(bands[i][1] < bands[i + 1][0] for i in range(3))
    print(f"  the four glyph bands are DISJOINT: {disjoint}")

    # kerning changes the string WIDTH, so a per-position band atlas cannot work; the atlas has
    # to be keyed by the whole string.  How many strings are reachable, and how big is the table?
    strings = ["%04i" % v for v in range(-100, 1001)]
    sz = {}
    for t in strings:
        sz[t] = f.render(t, True, (255, 255, 255), (0, 0, 0)).get_size()
    ws = sorted({w for w, h in sz.values()})
    hs = sorted({h for w, h in sz.values()})
    print(f"  reachable strings for reward in [-100, 1000]: {len(strings)}")
    print(f"  distinct rendered widths {ws}, heights {hs}")
    mx, mh = max(ws), max(hs)
    print(f"  a whole-string atlas is {len(strings)} x {mh} x {mx} grey bytes = "
          f"{len(strings)*mh*mx/1e6:.2f} MB")
    g8 = arr(f.render('8', True, (255, 255, 255), (0, 0, 0)))
    print(f"  glyph is grey (R==G==B): {bool((g8[...,0]==g8[...,1]).all() and (g8[...,1]==g8[...,2]).all())}"
          f", {len(np.unique(g8[...,0]))} distinct levels")

    # what does the observation actually see of it?
    r = f.render("0000", True, (255, 255, 255), (0, 0, 0)).get_rect()
    r.center = (60, H - H * 2.5 / 40.0)
    print(f"  blit rect for '0000': {r} (center (60, {H - H*2.5/40.0}))")
    print(f"  glyph greys: {sorted(set(np.unique(arr(f.render('8', True, (255,255,255), (0,0,0)))).tolist()))[:12]}...")


def aa_floor():
    if not torch.cuda.is_available():
        return
    from stable_envs.render import gfxdraw as G
    print("\n=== Wu anti-aliasing pass: per-edge loop vs one batched call ===")
    g = G.Gfx(H, W, "cuda")
    col = g.color((102, 204, 102))
    for E in (4, 64, 800, 3200):
        cv = g.canvas(E, (102, 204, 102))
        bi = g.batch_index(E)
        gen = torch.Generator(device="cuda").manual_seed(0)
        p = (torch.rand((4, E), generator=gen, device="cuda") * 900).to(torch.int64)

        def once():
            G.aaline(cv, p[0], p[1], p[2], p[3], col, 300, bi)
        for _ in range(2):
            once()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            once()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 5
        print(f"  one aaline call covering {E:5d} edges: {1000*dt:7.3f} ms "
              f"({1e6*dt/E:7.2f} us/edge)")
    print("  => the per-edge cost is python/launch overhead, not pixels: one call does 3200 edges")
    print("     for the same wall time as one edge, so the AA pass is bounded by how many")
    print("     SEQUENTIAL blend steps the draw order forces, not by the edge count.")


if __name__ == "__main__":
    census()
    font_atlas()
    aa_floor()
