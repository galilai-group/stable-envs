"""Layer 1-3 parity: background + 400 grass + N road quads, at 1000x800, PRE-flip, PRE-resample.

This is the layer with all the risk in it -- the camera transform, the reference's own cull, the
per-quad colours and 795 order-dependent anti-aliased blends. Comparing before the flip and before
the resample means a failure here is a geometry or blend error and cannot be a resample artifact.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pygame
import torch

import render_core as RC
from stable_envs.render import gfxdraw as G

W, H = RC.W, RC.H


def bake(seed=0):
    import gymnasium
    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=seed)
    u = env.unwrapped
    road_v = np.array([[list(p) for p in poly] for poly, _ in u.road_poly], np.float64)
    road_c = np.array([[int(c) for c in col] for _, col in u.road_poly], np.int64)
    grass = []
    for x in range(-20, 20, 2):
        for y in range(-20, 20, 2):
            g = RC.GRASS_DIM
            grass.append([(g * x + g, g * y + 0), (g * x + 0, g * y + 0),
                          (g * x + 0, g * y + g), (g * x + g, g * y + g)])
    b = RC.PLAYFIELD
    field = [(b, b), (b, -b), (-b, -b), (-b, b)]
    return env, u, road_v, road_c, np.array(grass, np.float64), np.array([field], np.float64)


def ref_road(u, st):
    """The reference's own layers 1-3, drawn on a fresh surface, pre-flip."""
    from gymnasium.envs.box2d.car_racing import SCALE, ZOOM
    set_state(u, st)
    surf = pygame.Surface((W, H))
    u.surf = surf
    angle = -u.car.hull.angle
    zoom = 0.1 * SCALE * max(1 - u.t, 0) + ZOOM * SCALE * min(u.t, 1)
    tr = pygame.math.Vector2((-(u.car.hull.position[0]) * zoom,
                              -(u.car.hull.position[1]) * zoom)).rotate_rad(angle)
    tr = (W / 2 + tr[0], H / 4 + tr[1])
    u._render_road(zoom, tr, angle)
    return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).copy()


def set_state(u, st):
    u.car.hull.position = (float(st[0]), float(st[1]))
    u.car.hull.angle = float(st[2])
    u.t = float(st[3])


def gpu_road(states, road_v, road_c, grass_v, field_v, dev="cuda"):
    B = states.shape[0]
    gfx = G.Gfx(H, W, dev)
    cv = RC.canvas(gfx, B, (0, 0, 0))
    cv.aa = torch.ones(B, dtype=torch.bool, device=dev)
    bi = gfx.batch_index(B)
    s = torch.as_tensor(states, device=dev, dtype=torch.float64)
    cos, sin, zoom, trans = RC.view_params(s[:, 0], s[:, 1], s[:, 2], s[:, 3])

    rv = torch.as_tensor(road_v, device=dev)
    gv = torch.as_tensor(grass_v, device=dev)
    fv = torch.as_tensor(field_v, device=dev)

    def col_of(c):
        t = torch.zeros(B, 3, dtype=torch.float32, device=dev)
        for k in range(3):
            t[:, k] = float(c[k])
        return t

    def draw(world_verts, color, enabled):
        RC.poly(cv, world_verts, col_of(color), enabled, bi)

    # 1 background, clip=False so it is always drawn
    scr = RC.to_screen(fv[0].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans)
    draw(scr, (102, 204, 102), torch.ones(B, dtype=torch.bool, device=dev))

    # 2 grass, in the reference's x-outer/y-inner order
    for i in range(gv.shape[0]):
        scr = RC.to_screen(gv[i].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans)
        draw(scr, (102, 230, 102), RC.visible(scr))

    # 3 road, in road_poly order, each with its own colour
    for i in range(rv.shape[0]):
        scr = RC.to_screen(rv[i].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans)
        draw(scr, tuple(int(c) for c in road_c[i]), RC.visible(scr))
    return cv.buf.cpu().numpy()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env, u, road_v, road_c, grass_v, field_v = bake(0)
    print(f"baked seed 0: {road_v.shape[0]} road quads, {grass_v.shape[0]} grass quads")

    rng = np.random.default_rng(0)
    states = []
    for k in range(4):
        beta, x, y = u.track[int(rng.integers(len(u.track)))][1:4]
        t = 2.0 if k < 3 else 0.62
        states.append([x, y, beta + float(rng.uniform(-0.4, 0.4)), t])
    states = np.array(states, np.float64)

    ref = np.stack([ref_road(u, st) for st in states])
    got = gpu_road(states, road_v, road_c, grass_v, field_v, dev)

    print("\n=== layers 1-3 (background + grass + road), 1000x800 pre-flip ===")
    tot = 0
    for i, st in enumerate(states):
        d = np.abs(got[i].astype(np.int32) - ref[i].astype(np.int32)).max(-1)
        tot += int((d > 0).sum())
        nd = int((d > 0).sum())
        print(f"  state {i} (t={st[3]:.2f}): {nd:6d} of {d.size} px differ, "
              f"max {int(d.max())}/255, exact {100*(d==0).mean():.4f}%")
        if d.max():
            ys, xs = np.nonzero(d)
            print(f"      first differing px (row,col) = ({ys[0]}, {xs[0]}), "
                  f"ours {got[i][ys[0], xs[0]]} ref {ref[i][ys[0], xs[0]]}")
    print(f"\ntotal differing pixels over {len(states)} frames: {tot}")
    env.close()


if __name__ == "__main__":
    main()
