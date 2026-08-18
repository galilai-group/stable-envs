"""Reference re-implementations of four gymnasium classic-control ``render()`` methods.

Each one is a pure function of a compact state vector and uses only the exact primitives in
``ref_gfx``. They exist to prove, before any GPU code is written, that the primitive rules are
complete: if these reproduce the official frames bit-for-bit then everything left is arithmetic
that torch can do in parallel.

Minimal render state per env (velocities never reach the screen):
    CartPole      (x, theta)                       -- 2 of the 4 observed dims
    Pendulum      (theta, last_u)                  -- last_u is NOT in the observation
    Acrobot       (theta1, theta2)                 -- 2 of the 4 underlying dims
    MountainCar   (position,)                      -- 1 of 2
"""
from __future__ import annotations

import math

import numpy as np

import ref_gfx as G


def _rot(v, a):
    """``pygame.math.Vector2(v).rotate_rad(a)`` -- [[cos,-sin],[sin,cos]] in a y-down raster."""
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def _blank(h, w):
    return np.full((h, w, 3), 255, np.int64)


# --------------------------------------------------------------------------- CartPole-v1
CARTPOLE = dict(W=600, H=400, scale=600 / 4.8, polewidth=10.0, polelen=125.0,
                cartwidth=50.0, cartheight=30.0, carty=100.0, axleoffset=30.0 / 4.0,
                cart_c=(0, 0, 0), pole_c=(202, 152, 101), axle_c=(129, 132, 203))


def render_cartpole(x, theta, aa=True):
    P = CARTPOLE
    cv = _blank(P["H"], P["W"])
    cartx = x * P["scale"] + P["W"] / 2.0
    carty = P["carty"]
    l, r, t, b = -25.0, 25.0, 15.0, -15.0
    cart = [(l + cartx, b + carty), (l + cartx, t + carty),
            (r + cartx, t + carty), (r + cartx, b + carty)]
    _poly(cv, cart, P["cart_c"], aa)

    l, r, t, b = -5.0, 5.0, P["polelen"] - 5.0, -5.0
    pole = []
    for c in [(l, b), (l, t), (r, t), (r, b)]:
        c = _rot(c, -theta)
        pole.append((c[0] + cartx, c[1] + carty + P["axleoffset"]))
    _poly(cv, pole, P["pole_c"], aa)

    _circle(cv, int(cartx), int(carty + P["axleoffset"]), int(P["polewidth"] / 2), P["axle_c"], aa)
    cv[int(carty), 0:P["W"]] = np.array((0, 0, 0), np.int64)          # gfxdraw.hline, drawn LAST
    return np.flipud(cv).astype(np.uint8)


# --------------------------------------------------------------------------- Pendulum-v1
PENDULUM = dict(D=500, scale=500 / 4.4, rod_c=(204, 77, 77), axle_c=(0, 0, 0))


def render_pendulum(theta, aa=True):
    """Vector part only; the torque sprite (drawn iff ``last_u`` is set) is handled separately."""
    P = PENDULUM
    sc, off = P["scale"], 250
    cv = _blank(P["D"], P["D"])
    rod_length, rod_width = 1.0 * sc, 0.2 * sc
    l, r, t, b = 0.0, rod_length, rod_width / 2, -rod_width / 2
    rod = [(_rot(c, theta + np.pi / 2)[0] + off, _rot(c, theta + np.pi / 2)[1] + off)
           for c in [(l, b), (l, t), (r, t), (r, b)]]
    _poly(cv, rod, P["rod_c"], aa)
    _circle(cv, off, off, int(rod_width / 2), P["rod_c"], aa)
    re = _rot((rod_length, 0.0), theta + np.pi / 2)
    _circle(cv, int(re[0] + off), int(re[1] + off), int(rod_width / 2), P["rod_c"], aa)
    _circle(cv, off, off, int(0.05 * sc), P["axle_c"], aa)
    return np.flipud(cv).astype(np.uint8)


# --------------------------------------------------------------------------- Acrobot-v1
ACROBOT = dict(D=500, scale=500 / 4.4, L1=1.0, L2=1.0, link_c=(0, 204, 204), joint_c=(204, 204, 0))


def render_acrobot(th1, th2, aa=True):
    P = ACROBOT
    sc, off = P["scale"], 250.0
    cv = _blank(P["D"], P["D"])
    # ground: pygame.draw.line with float endpoints truncates the coordinate (363.6364 -> row 363),
    # it does not round -- rounding puts the whole line one row off and costs 1000 px a frame.
    cv[int(1 * sc + off), 0:P["D"]] = np.array((0, 0, 0), np.int64)
    p1 = [-P["L1"] * math.cos(th1) * sc, P["L1"] * math.sin(th1) * sc]
    p2 = [p1[0] - P["L2"] * math.cos(th1 + th2) * sc, p1[1] + P["L2"] * math.sin(th1 + th2) * sc]
    xys = [(0.0, 0.0), (p1[1], p1[0])]                       # note the [:, ::-1] in the source
    thetas = [th1 - np.pi / 2, th1 + th2 - np.pi / 2]
    for (x, y), th, llen in zip(xys, thetas, [P["L1"] * sc, P["L2"] * sc]):
        x, y = x + off, y + off
        l, r, t, b = 0.0, llen, 0.1 * sc, -0.1 * sc
        link = [(_rot(c, th)[0] + x, _rot(c, th)[1] + y) for c in [(l, b), (l, t), (r, t), (r, b)]]
        _poly(cv, link, P["link_c"], aa)
        _circle(cv, int(x), int(y), int(0.1 * sc), P["joint_c"], aa)
    _ = p2
    return np.flipud(cv).astype(np.uint8)


# --------------------------------------------------------------------------- MountainCar-v0
MOUNTAINCAR = dict(W=600, H=400, minp=-1.2, maxp=0.6, scale=600 / 1.8, carwidth=40.0,
                   carheight=20.0, clearance=10.0, car_c=(0, 0, 0), wheel_c=(128, 128, 128))


def mc_height(x):
    return np.sin(3 * np.asarray(x)) * 0.45 + 0.55


def render_mountaincar(pos, base=None, aa=True):
    """``base`` is the static terrain + flag layer (aalines / vline / flag triangle)."""
    P = MOUNTAINCAR
    cv = _blank(P["H"], P["W"]) if base is None else base.astype(np.int64).copy()
    ang = math.cos(3 * pos)
    cxp = (pos - P["minp"]) * P["scale"]
    cyp = P["clearance"] + float(mc_height(pos)) * P["scale"]
    l, r, t, b = -20.0, 20.0, 20.0, 0.0
    car = [(_rot(c, ang)[0] + cxp, _rot(c, ang)[1] + cyp) for c in [(l, b), (l, t), (r, t), (r, b)]]
    _poly(cv, car, P["car_c"], aa)
    for c in [(10.0, 0.0), (-10.0, 0.0)]:
        c = _rot(c, ang)
        _circle(cv, int(c[0] + cxp), int(c[1] + cyp), int(P["carheight"] / 2.5), P["wheel_c"], aa)
    return np.flipud(cv).astype(np.uint8)


# --------------------------------------------------------------------------- helpers
_AACIRCLE_CACHE = {}


def _poly(cv, verts, color, aa):
    vi = G.trunc_verts(verts)
    if aa:
        G.draw_aapolygon(cv, vi, color)
    cv[G.filled_polygon_mask(vi, cv.shape[0], cv.shape[1])] = np.asarray(color, np.int64)


def _circle(cv, cx, cy, r, color, aa):
    if aa:
        st = _aacircle_stencil(r, tuple(color))
        for p in (0, 1):                              # a few fringe pixels are written twice
            for (dy, dx, w, ps) in st:
                if ps != p:
                    continue
                y, x = cy + dy, cx + dx
                if 0 <= y < cv.shape[0] and 0 <= x < cv.shape[1]:
                    cv[y, x] = G.gfx_blend(np.asarray(color, np.int64), cv[y, x], w)
    cv[G.filled_circle_mask(cx, cy, r, cv.shape[0], cv.shape[1])] = np.asarray(color, np.int64)


def _aacircle_stencil(r, color):
    if (r, color) not in _AACIRCLE_CACHE:
        _AACIRCLE_CACHE[(r, color)] = G.bake_aacircle(r, color)
    return _AACIRCLE_CACHE[(r, color)]
