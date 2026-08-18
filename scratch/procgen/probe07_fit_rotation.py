"""Fit the rotated `drawImage` path against the measured samples.

Two independent questions, separated because the measurement answers them separately:

  A. COVERAGE -- which device pixels does the rotated quad write?
  B. SAMPLING -- for a covered pixel, which source pixel does it read?

Only the AGENT is used (sprite id 6, unique in the frame). The ring keys share their sprite ids
with the world keys lying in the maze, so a mask built from the id alone mixes a rotated draw with
an unrotated one -- which is what made the first pass of this probe report nonsense.

Candidates for B are the two forms Qt has for a transformed texture fetch, crossed with whether
`QSpanData::setupMatrix`'s `translate(1/65536, 1/65536)` delta is applied, and with whether the
fixed-point accumulator restarts per SPAN or is evaluated per PIXEL.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtblit as Q  # noqa: E402

F = np.float32
PI_F = F(3.14159265358979323846264338327950288)
AGENT_ID = 6
AGENT_WH = (37, 44)


def qtransform_rot(deg):
    if deg == 90.0 or deg == -270.0:
        return 1.0, 0.0
    if deg == 270.0 or deg == -90.0:
        return -1.0, 0.0
    if deg == 180.0 or deg == -180.0:
        return 0.0, -1.0
    b = np.deg2rad(np.float64(deg))
    return float(np.sin(b)), float(np.cos(b))


def source_to_device(rect, deg, sw, sh):
    rx, ry, rw, rh = rect
    cx, cy = rx + rw / 2.0, ry + rh / 2.0
    sina, cosa = qtransform_rot(deg)
    m11, m12, m21, m22 = cosa, sina, -sina, cosa
    dx, dy = cx, cy
    tx, ty = -rw / 2.0, -rh / 2.0
    dx += tx * m11 + ty * m21
    dy += tx * m12 + ty * m22
    sxs, sys_ = rw / sw, rh / sh
    m11, m12 = m11 * sxs, m12 * sxs
    m21, m22 = m21 * sys_, m22 * sys_
    return np.array([[m11, m12], [m21, m22]], np.float64), np.array([dx, dy], np.float64)


def invert(M, T, delta):
    if delta:
        d = 1.0 / 65536.0
        T = T + np.array([d, d]) @ M
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    inv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
    return inv, -T @ inv


def quad(rect, deg):
    M, T = source_to_device(rect, deg, 1.0, 1.0)
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64) @ M + T


def covered_centre(rect, deg, W, H):
    c = quad(rect, deg)
    ys, xs = np.mgrid[0:H, 0:W]
    px, py = xs + 0.5, ys + 0.5
    sgn = None
    inside = np.ones((H, W), bool)
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        cr = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        if sgn is None:
            s = 0.0
            for j in range(4):
                p, q = c[j], c[(j + 1) % 4]
                s += (q[0] - p[0]) * (q[1] + p[1])
            sgn = 1.0 if s < 0 else -1.0
        inside &= (cr * sgn) >= 0
    return inside


def sample_map(rect, deg, sw, sh, cover, delta, half, per_span):
    M, T = source_to_device(rect, deg, sw, sh)
    inv, invT = invert(M, T, delta)
    m11, m12 = inv[0, 0], inv[0, 1]
    m21, m22 = inv[1, 0], inv[1, 1]
    dx, dy = invT
    fdx, fdy = int(m11 * 65536), int(m12 * 65536)
    off = 32768 if half else 0
    H, W = cover.shape
    SX = -np.ones((H, W), np.int64)
    SY = -np.ones((H, W), np.int64)
    for y in range(H):
        idx = np.flatnonzero(cover[y])
        if idx.size == 0:
            continue
        if per_span:
            brk = np.flatnonzero(np.diff(idx) > 1)
            runs = np.split(idx, brk + 1)
        else:
            runs = [np.array([i]) for i in idx]
        for run in runs:
            x0 = run[0]
            X = int((m21 * (y + 0.5) + m11 * (x0 + 0.5) + dx) * 65536) - off
            Y = int((m22 * (y + 0.5) + m12 * (x0 + 0.5) + dy) * 65536) - off
            i = np.arange(len(run))
            SX[y, run] = (X + i * fdx) >> 16
            SY[y, run] = (Y + i * fdy) >> 16
    return SX, SY


def main():
    z = np.load(os.path.join(HERE, "rot_samples.npz"), allow_pickle=True)
    frames, meta = z["frames"], z["meta"]
    cases = []
    for f, m in zip(frames, meta):
        mw, mh = m["main"]
        unit, view_dim, x_off, y_off, _ = Q.prepare(mw, mh, False, (0, 0), 0.0, 64)
        for e in m["ents"]:
            if e["type"] != 0 or e["rotation"] == 0.0:
                continue
            rect = Q.screen_rect(e["x"] - e["rx"], e["y"] + e["ry"], 2 * e["rx"], 2 * e["ry"],
                                 unit, view_dim, x_off, y_off)
            deg = float(F(F(e["rotation"]) * F(180)) / PI_F)
            cases.append((f, rect, deg))

    cov_ok = cov_extra = cov_miss = n = 0
    for f, rect, deg in cases:
        meas = f[..., 2] == AGENT_ID
        pred = covered_centre(rect, deg, 64, 64)
        n += int(meas.sum())
        cov_ok += int((meas & pred).sum())
        cov_miss += int((meas & ~pred).sum())
        cov_extra += int((~meas & pred).sum())
    print(f"COVERAGE (pixel-centre-in-quad): {cov_ok}/{n} hit, {cov_miss} measured-but-not-"
          f"predicted, {cov_extra} predicted-but-not-measured, over {len(cases)} rotated agents")

    print("\nSAMPLING variants (fraction of measured agent pixels whose (sx,sy) match):")
    for delta, half, per_span in itertools.product([True, False], [True, False], [True, False]):
        ok = tot = 0
        for f, rect, deg in cases:
            meas = f[..., 2] == AGENT_ID
            SX, SY = sample_map(rect, deg, *AGENT_WH, meas, delta, half, per_span)
            good = (SX == f[..., 0]) & (SY == f[..., 1])
            ok += int((good & meas).sum())
            tot += int(meas.sum())
        print(f"  delta={int(delta)} half={int(half)} per_span={int(per_span)}: "
              f"{ok}/{tot} = {100*ok/max(tot,1):6.2f}%")


if __name__ == "__main__":
    main()
