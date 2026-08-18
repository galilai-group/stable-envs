"""End-to-end parity: the full 96x96 CarRacing observation, GPU vs the official renderer."""
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
from probe_smoothscale_gpu import Shrink
from stable_envs.render import gfxdraw as G

W, H = RC.W, RC.H
NS = 28


def bake(seed=0, dev="cuda"):
    import gymnasium
    from gymnasium.envs.box2d.car_dynamics import SIZE, WHEEL_R, WHEEL_W
    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=seed)
    u = env.unwrapped
    d = {}
    d["road_v"] = torch.tensor([[list(p) for p in poly] for poly, _ in u.road_poly],
                               dtype=torch.float64, device=dev)
    d["road_c"] = torch.tensor([[int(c) for c in col] for _, col in u.road_poly],
                               dtype=torch.float32, device=dev)
    g = RC.GRASS_DIM
    grass = [[(g * x + g, g * y), (g * x, g * y), (g * x, g * y + g), (g * x + g, g * y + g)]
             for x in range(-20, 20, 2) for y in range(-20, 20, 2)]
    d["grass_v"] = torch.tensor(grass, dtype=torch.float64, device=dev)
    b = RC.PLAYFIELD
    d["field_v"] = torch.tensor([[(b, b), (b, -b), (-b, -b), (-b, b)]],
                                dtype=torch.float64, device=dev)
    wl, hl = RC.car_local(u)
    assert all(w == wl[0] for w in wl), "the four wheels do not share one local polygon"
    d["wheel_local"] = torch.tensor(wl[0], dtype=torch.float64, device=dev)
    d["hull_local"] = [torch.tensor(h, dtype=torch.float64, device=dev) for h in hl]
    d["n_hull_v"] = [len(h) for h in hl]
    d["hull_color"] = [int(c * 255) for c in u.car.hull.color]
    d["wheel_color"] = [int(c * 255) for c in u.car.wheels[0].color]
    d["wheel_white"] = [77, 77, 77]
    d["wheel_w"] = WHEEL_W * SIZE
    d["wheel_r"] = WHEEL_R * SIZE
    at, wid, th, lo = RC.bake_text()
    d["atlas"] = torch.as_tensor(at, device=dev)
    d["widths"] = torch.as_tensor(wid, device=dev)
    d["th"], d["lo"] = th, lo
    d["sx"] = Shrink(W, 96, dev)
    d["sy"] = Shrink(H, 96, dev)
    return env, u, d


def col_of(c, B, dev):
    t = torch.empty(B, 3, dtype=torch.float32, device=dev)
    for k in range(3):
        t[:, k] = float(c[k])
    return t


def render(states, d, dev="cuda"):
    B = states.shape[0]
    gfx = G.Gfx(H, W, dev)
    cv = RC.canvas(gfx, B, (0, 0, 0))
    cv.aa = torch.ones(B, dtype=torch.bool, device=dev)
    bi = gfx.batch_index(B)
    s = torch.as_tensor(states, device=dev, dtype=torch.float64)
    cos, sin, zoom, trans = RC.view_params(s[:, 0], s[:, 1], s[:, 2], s[:, 27])
    on = torch.ones(B, dtype=torch.bool, device=dev)

    scr = RC.to_screen(d["field_v"][0].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans)
    RC.poly(cv, scr, col_of((102, 204, 102), B, dev), on, bi)
    for i in range(d["grass_v"].shape[0]):
        scr = RC.to_screen(d["grass_v"][i].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans)
        RC.poly(cv, scr, col_of((102, 230, 102), B, dev), RC.visible(scr), bi)
    for i in range(d["road_v"].shape[0]):
        scr = RC.to_screen(d["road_v"][i].unsqueeze(0).expand(B, 4, 2), cos, sin, zoom, trans)
        RC.poly(cv, scr, d["road_c"][i].view(1, 3).expand(B, 3), RC.visible(scr), bi)

    # 4 car: wheels (each followed by its stripe) then the four hull fixtures
    wc = col_of(d["wheel_color"], B, dev)
    ww = col_of(d["wheel_white"], B, dev)
    hc = col_of(d["hull_color"], B, dev)
    for k in range(4):
        px, py, ang = s[:, 6 + 3 * k], s[:, 7 + 3 * k], s[:, 8 + 3 * k]
        wv = RC.body_to_world(d["wheel_local"].unsqueeze(0).expand(B, 4, 2), px, py, ang)
        RC.draw_fill(cv, RC.to_screen(wv, cos, sin, zoom, trans), wc, on)
        sv, gate = RC.wheel_stripe(px, py, ang, s[:, 18 + k], d["wheel_w"], d["wheel_r"])
        RC.draw_fill(cv, RC.to_screen(sv, cos, sin, zoom, trans), ww, gate)
    for j, hl in enumerate(d["hull_local"]):
        n = hl.shape[0]
        hv = RC.body_to_world(hl.unsqueeze(0).expand(B, n, 2), s[:, 0], s[:, 1], s[:, 2])
        RC.draw_fill(cv, RC.to_screen(hv, cos, sin, zoom, trans), hc, on)

    RC.flip_in_place(cv)

    speed = torch.sqrt(s[:, 3] ** 2 + s[:, 4] ** 2)
    steer = s[:, 8] - s[:, 2]                      # revolute joint angle = wheel0.angle - hull.angle
    for verts, col, gate in RC.indicator_polys(speed, s[:, 22:26], steer, s[:, 5], dev):
        RC.draw_fill(cv, verts, col_of(col, B, dev), gate)

    RC.blit_text(cv, s[:, 26], d["atlas"], d["widths"], d["th"], d["lo"])
    return d["sy"](d["sx"](cv.buf, 2), 1)


def set_state(u, st):
    u.car.hull.position = (float(st[0]), float(st[1]))
    u.car.hull.angle = float(st[2])
    u.car.hull.linearVelocity = (float(st[3]), float(st[4]))
    u.car.hull.angularVelocity = float(st[5])
    for k, w in enumerate(u.car.wheels):
        w.position = (float(st[6 + 3 * k]), float(st[7 + 3 * k]))
        w.angle = float(st[8 + 3 * k])
        w.phase = float(st[18 + k])
        w.omega = float(st[22 + k])
    u.reward = float(st[26])
    u.t = float(st[27])
    # Box2D is a float32 engine: every pose it stores is rounded to float32, so the state has to be
    # float32 for the round-trip to be exact -- which is also the repo's own state dtype. Asserting
    # at float64 tolerance fails by ~2e-6 and would look like a renderer bug.
    got = np.array([*u.car.hull.position, u.car.hull.angle], np.float64)
    want = np.asarray(st[:3], np.float64)
    if np.abs(got - want).max() > 1e-6:
        raise AssertionError(f"reference did not reach the state: {got} vs {want}")
    j = u.car.wheels[0].joint.angle
    assert abs(j - (float(st[8]) - float(st[2]))) < 1e-5, \
        f"joint angle {j} is not wheel0.angle - hull.angle ({float(st[8]) - float(st[2])})"


def sample(u, n, rng):
    """States the env reaches: the car on a track tile, driving, with the zoom past its ramp."""
    out = []
    for _ in range(n):
        beta, x, y = u.track[int(rng.integers(len(u.track)))][1:4]
        ang = beta + float(rng.uniform(-0.5, 0.5))
        sp = float(rng.uniform(0, 25))
        st = [x, y, ang, sp * np.cos(ang), sp * np.sin(ang), float(rng.uniform(-1.5, 1.5))]
        for k, (wx, wy) in enumerate([(-55, 80), (55, 80), (-55, -82), (55, -82)]):
            lx, ly = wx * 0.02, wy * 0.02
            st += [x + lx * np.cos(ang) - ly * np.sin(ang),
                   y + lx * np.sin(ang) + ly * np.cos(ang),
                   ang + (float(rng.uniform(-0.4, 0.4)) if k < 2 else 0.0)]
        st += [float(rng.uniform(-10, 10)) for _ in range(4)]          # phases
        st += [float(rng.uniform(-5, 90)) for _ in range(4)]           # omegas
        st += [float(rng.uniform(-100, 999)), float(rng.uniform(0.55, 3.0))]
        out.append(st)
    return np.array(out, np.float32)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env, u, d = bake(0, dev)
    rng = np.random.default_rng(0)
    states = sample(u, 8, rng)

    ref = []
    for st in states:
        set_state(u, st)
        ref.append(u.render().copy())
    ref = np.stack(ref)
    got = render(states, d, dev).cpu().numpy()

    print(f"\n=== full 96x96 observation, {len(states)} states ===")
    dd = np.abs(got.astype(np.int32) - ref.astype(np.int32)).max(-1)
    for i in range(len(states)):
        n = int((dd[i] > 0).sum())
        print(f"  state {i}: {n:5d}/{dd[i].size} px differ, max {int(dd[i].max())}/255, "
              f"exact {100*(dd[i]==0).mean():7.3f}%")
    print(f"  ALL: exact {100*(dd==0).mean():.4f}%, mean |d| {dd.mean():.4f}, max {int(dd.max())}")
    if dd.max():
        i = int(np.argmax(dd.reshape(len(states), -1).max(1)))
        ys, xs = np.nonzero(dd[i])
        print(f"  worst frame {i}, first differing px (r,c)=({ys[0]},{xs[0]}) "
              f"ours {got[i][ys[0], xs[0]]} ref {ref[i][ys[0], xs[0]]}")
        rows = np.unique(ys)
        print(f"  differing rows {rows.min()}..{rows.max()} (indicator strip is >=84)")
    env.close()


if __name__ == "__main__":
    main()
