"""Triton CUDA port of the PushT contact solver in ``jax_physics``: the same math in float32, one environment per thread.

Ports the solver that ``jax_physics`` adapts from fracapuano/envX at d26c817.
Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

#: Bias velocity per pixel of penetration beyond the slop, as in ``jax_physics``.
_RATE = tl.constexpr((1 - 0.9**0.6) / 0.01)
#: Scratch fields of one contact slot: point (0, 1), normal (2, 3), depth (4), ra (5, 6), rb (7, 8), the
#: cross products (9, 10), effective mass (11), bias target (12), impulse (13), bias impulse (14),
#: active (15) and feature id (16).
FIELDS = 17


@triton.jit
def _outward(x0, y0, x1, y1):
    ex, ey = x1 - x0, y1 - y0
    length = tl.sqrt(ex * ex + ey * ey)
    return ey / length, -ex / length


@triton.jit
def _world(vertices, lane, index, shape, k, px, py, c, s, mask, S: tl.constexpr):
    base = vertices + (((lane * 2 + index) * S + shape) * 4 + k) * 2
    vx = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    vy = tl.load(base + 1, mask=mask, other=0.0).to(tl.float32)
    return px + c * vx - s * vy, py + s * vx + c * vy


@triton.jit
def _put(scratch, slot, B, lane, mask, field: tl.constexpr, value):
    tl.store(scratch + (slot * 17 + field) * B + lane, value, mask=mask)


@triton.jit
def _get(scratch, slot, B, lane, mask, field: tl.constexpr):
    return tl.load(scratch + (slot * 17 + field) * B + lane, mask=mask, other=0.0)


@triton.jit
def _circle_polygon(cx, cy, radius, x0, y0, x1, y1, x2, y2, x3, y3):
    """Return point, normal (from the circle into the polygon), depth and feature id."""
    n0x, n0y = _outward(x0, y0, x1, y1)
    n1x, n1y = _outward(x1, y1, x2, y2)
    n2x, n2y = _outward(x2, y2, x3, y3)
    n3x, n3y = _outward(x3, y3, x0, y0)
    s0 = (cx - x0) * n0x + (cy - y0) * n0y
    s1 = (cx - x1) * n1x + (cy - y1) * n1y
    s2 = (cx - x2) * n2x + (cy - y2) * n2y
    s3 = (cx - x3) * n3x + (cy - y3) * n3y
    face = tl.where(s1 > s0, 1, 0)
    best = tl.maximum(s0, s1)
    face = tl.where(s2 > best, 2, face)
    best = tl.maximum(best, s2)
    face = tl.where(s3 > best, 3, face)
    best = tl.maximum(best, s3)
    nx = tl.where(face == 0, n0x, tl.where(face == 1, n1x, tl.where(face == 2, n2x, n3x)))
    ny = tl.where(face == 0, n0y, tl.where(face == 1, n1y, tl.where(face == 2, n2y, n3y)))
    sx = tl.where(face == 0, x0, tl.where(face == 1, x1, tl.where(face == 2, x2, x3)))
    sy = tl.where(face == 0, y0, tl.where(face == 1, y1, tl.where(face == 2, y2, y3)))
    tx = tl.where(face == 0, x1, tl.where(face == 1, x2, tl.where(face == 2, x3, x0)))
    ty = tl.where(face == 0, y1, tl.where(face == 1, y2, tl.where(face == 2, y3, y0)))
    ex, ey = tx - sx, ty - sy
    t = ((cx - sx) * ex + (cy - sy) * ey) / (ex * ex + ey * ey)
    t = tl.minimum(tl.maximum(t, 0.0), 1.0)
    px, py = sx + t * ex, sy + t * ey
    dx, dy = cx - px, cy - py
    distance = tl.maximum(tl.sqrt(dx * dx + dy * dy), 1e-12)
    inside = best <= 0
    nx = tl.where(inside, nx, dx / distance)
    ny = tl.where(inside, ny, dy / distance)
    depth = radius - tl.where(inside, best, distance)
    ident = face * 3 + tl.where(t == 0, 1, tl.where(t == 1, 2, 0))
    return px, py, -nx, -ny, depth, ident


@triton.jit
def _face_depth(vx0, vy0, vx1, vy1, ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3):
    nx, ny = _outward(vx0, vy0, vx1, vy1)
    low = tl.minimum(
        tl.minimum(ox0 * nx + oy0 * ny, ox1 * nx + oy1 * ny), tl.minimum(ox2 * nx + oy2 * ny, ox3 * nx + oy3 * ny)
    )
    return low - (vx0 * nx + vy0 * ny)


@triton.jit
def _reference(x0, y0, x1, y1, x2, y2, x3, y3, ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3):
    d0 = _face_depth(x0, y0, x1, y1, ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3)
    d1 = _face_depth(x1, y1, x2, y2, ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3)
    d2 = _face_depth(x2, y2, x3, y3, ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3)
    d3 = _face_depth(x3, y3, x0, y0, ox0, oy0, ox1, oy1, ox2, oy2, ox3, oy3)
    face = tl.where(d1 > d0, 1, 0)
    best = tl.maximum(d0, d1)
    face = tl.where(d2 > best, 2, face)
    best = tl.maximum(best, d2)
    face = tl.where(d3 > best, 3, face)
    best = tl.maximum(best, d3)
    return face, best


@triton.jit
def _corner(face, x0, x1, x2, x3):
    return tl.where(face == 0, x0, tl.where(face == 1, x1, tl.where(face == 2, x2, x3)))


@triton.jit
def _polygon_polygon(ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3, bx0, by0, bx1, by1, bx2, by2, bx3, by3):
    """Return two contact points, the shared normal (from ``a`` into ``b``), two depths and the first id."""
    ai, ad = _reference(ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3, bx0, by0, bx1, by1, bx2, by2, bx3, by3)
    bi, bd = _reference(bx0, by0, bx1, by1, bx2, by2, bx3, by3, ax0, ay0, ax1, ay1, ax2, ay2, ax3, ay3)
    flip = bd > ad
    rx0, ry0 = tl.where(flip, bx0, ax0), tl.where(flip, by0, ay0)
    rx1, ry1 = tl.where(flip, bx1, ax1), tl.where(flip, by1, ay1)
    rx2, ry2 = tl.where(flip, bx2, ax2), tl.where(flip, by2, ay2)
    rx3, ry3 = tl.where(flip, bx3, ax3), tl.where(flip, by3, ay3)
    ix0, iy0 = tl.where(flip, ax0, bx0), tl.where(flip, ay0, by0)
    ix1, iy1 = tl.where(flip, ax1, bx1), tl.where(flip, ay1, by1)
    ix2, iy2 = tl.where(flip, ax2, bx2), tl.where(flip, ay2, by2)
    ix3, iy3 = tl.where(flip, ax3, bx3), tl.where(flip, ay3, by3)
    face = tl.where(flip, bi, ai)
    fx0, fy0 = _corner(face, rx0, rx1, rx2, rx3), _corner(face, ry0, ry1, ry2, ry3)
    fx1, fy1 = _corner((face + 1) % 4, rx0, rx1, rx2, rx3), _corner((face + 1) % 4, ry0, ry1, ry2, ry3)
    nx, ny = _outward(fx0, fy0, fx1, fy1)
    m0x, m0y = _outward(ix0, iy0, ix1, iy1)
    m1x, m1y = _outward(ix1, iy1, ix2, iy2)
    m2x, m2y = _outward(ix2, iy2, ix3, iy3)
    m3x, m3y = _outward(ix3, iy3, ix0, iy0)
    q0, q1, q2, q3 = m0x * nx + m0y * ny, m1x * nx + m1y * ny, m2x * nx + m2y * ny, m3x * nx + m3y * ny
    edge = tl.where(q1 < q0, 1, 0)
    low = tl.minimum(q0, q1)
    edge = tl.where(q2 < low, 2, edge)
    low = tl.minimum(low, q2)
    edge = tl.where(q3 < low, 3, edge)
    gx0, gy0 = _corner(edge, ix0, ix1, ix2, ix3), _corner(edge, iy0, iy1, iy2, iy3)
    gx1, gy1 = _corner((edge + 1) % 4, ix0, ix1, ix2, ix3), _corner((edge + 1) % 4, iy0, iy1, iy2, iy3)
    tx, ty = -ny, nx
    r0 = fx0 * tx + fy0 * ty
    r1 = fx1 * tx + fy1 * ty
    lo, hi = tl.minimum(r0, r1), tl.maximum(r0, r1)
    denom = (gx1 - gx0) * tx + (gy1 - gy0) * ty
    safe = tl.where(tl.abs(denom) > 1e-12, denom, 1.0)
    first = gx0 * tx + gy0 * ty
    e0, e1 = (lo - first) / safe, (hi - first) / safe
    e0, e1 = tl.minimum(e0, e1), tl.maximum(e0, e1)
    t0 = tl.minimum(tl.maximum(e0, 0.0), 1.0)
    t1 = tl.minimum(tl.maximum(e1, 0.0), 1.0)
    p0x, p0y = gx0 + t0 * (gx1 - gx0), gy0 + t0 * (gy1 - gy0)
    p1x, p1y = gx0 + t1 * (gx1 - gx0), gy0 + t1 * (gy1 - gy0)
    d0 = (fx0 - p0x) * nx + (fy0 - p0y) * ny
    d1 = (fx0 - p1x) * nx + (fy0 - p1y) * ny
    overlap = (tl.maximum(ad, bd) <= 0) & ((tl.abs(denom) > 1e-12) | ((first >= lo) & (first <= hi)))
    d0 = tl.where(overlap, d0, -1.0)
    d1 = tl.where(overlap, d1, -1.0)
    d1 = tl.where(tl.sqrt((p1x - p0x) * (p1x - p0x) + (p1y - p0y) * (p1y - p0y)) > 1e-9, d1, -1.0)
    sign = tl.where(flip, -1.0, 1.0)
    return p0x, p0y, p1x, p1y, sign * nx, sign * ny, d0, d1, face * 8 + edge * 2


@triton.jit
def _wall_pair(w: tl.constexpr, x0, y0, x1, y1, x2, y2, x3, y3):
    """The two deepest vertices against wall ``w``, the lower index first on ties (``lax.top_k``)."""
    if w == 0:
        nx, ny, plane = 1.0, 0.0, 7.0
    elif w == 1:
        nx, ny, plane = -1.0, 0.0, -504.0
    elif w == 2:
        nx, ny, plane = 0.0, 1.0, 7.0
    else:
        nx, ny, plane = 0.0, -1.0, -504.0
    p0 = plane - (nx * x0 + ny * y0)
    p1 = plane - (nx * x1 + ny * y1)
    p2 = plane - (nx * x2 + ny * y2)
    p3 = plane - (nx * x3 + ny * y3)
    # First: the deepest. Second: the deepest of the rest.
    first = tl.where(p1 > p0, 1, 0)
    top = tl.maximum(p0, p1)
    second = tl.where(p1 > p0, 0, 1)
    low = tl.minimum(p0, p1)
    better = p2 > top
    second = tl.where(better, first, tl.where(p2 > low, 2, second))
    low = tl.where(better, top, tl.maximum(low, p2))
    first = tl.where(better, 2, first)
    top = tl.maximum(top, p2)
    better = p3 > top
    second = tl.where(better, first, tl.where(p3 > low, 3, second))
    first = tl.where(better, 3, first)
    return first, second, nx, ny, plane


@triton.jit
def _contacts(
    scratch,
    B,
    lane,
    mask,
    vertices,
    enabled,
    radius,
    inv_agent,
    inv_block,
    ax,
    ay,
    aa,
    bx,
    by,
    ba,
    S: tl.constexpr,
    CIRCLE: tl.constexpr,
):
    """Write every contact slot's point, normal, depth and id, in ``jax_physics.contacts`` order.

    The loops over shapes run at run time, so the kernel compiles one copy of each collision test.
    """
    ca, sa = tl.cos(aa), tl.sin(aa)
    cb, sb = tl.cos(ba), tl.sin(ba)
    disc = radius > 0
    if CIRCLE:
        for b in range(S):
            x0, y0 = _world(vertices, lane, 1, b, 0, bx, by, cb, sb, mask, S)
            x1, y1 = _world(vertices, lane, 1, b, 1, bx, by, cb, sb, mask, S)
            x2, y2 = _world(vertices, lane, 1, b, 2, bx, by, cb, sb, mask, S)
            x3, y3 = _world(vertices, lane, 1, b, 3, bx, by, cb, sb, mask, S)
            px, py, nx, ny, depth, ident = _circle_polygon(ax, ay, radius, x0, y0, x1, y1, x2, y2, x3, y3)
            on = (tl.load(enabled + lane * 2 * S, mask=mask, other=0) != 0) & (
                tl.load(enabled + lane * 2 * S + S + b, mask=mask, other=0) != 0
            )
            _put(scratch, b, B, lane, mask, 0, px)
            _put(scratch, b, B, lane, mask, 1, py)
            _put(scratch, b, B, lane, mask, 2, nx)
            _put(scratch, b, B, lane, mask, 3, ny)
            _put(scratch, b, B, lane, mask, 4, tl.where(on, depth, -1.0))
            _put(scratch, b, B, lane, mask, 16, ident.to(tl.float32))
    else:
        for pair in range(S * S):
            a = pair // S
            b = pair % S
            slot = 2 * pair
            p0, p1 = _world(vertices, lane, 0, a, 0, ax, ay, ca, sa, mask, S)
            p2, p3 = _world(vertices, lane, 0, a, 1, ax, ay, ca, sa, mask, S)
            p4, p5 = _world(vertices, lane, 0, a, 2, ax, ay, ca, sa, mask, S)
            p6, p7 = _world(vertices, lane, 0, a, 3, ax, ay, ca, sa, mask, S)
            q0, q1 = _world(vertices, lane, 1, b, 0, bx, by, cb, sb, mask, S)
            q2, q3 = _world(vertices, lane, 1, b, 1, bx, by, cb, sb, mask, S)
            q4, q5 = _world(vertices, lane, 1, b, 2, bx, by, cb, sb, mask, S)
            q6, q7 = _world(vertices, lane, 1, b, 3, bx, by, cb, sb, mask, S)
            u0x, u0y, u1x, u1y, nx, ny, d0, d1, ident = _polygon_polygon(
                p0, p1, p2, p3, p4, p5, p6, p7, q0, q1, q2, q3, q4, q5, q6, q7
            )
            cx_, cy_, cnx, cny, cdepth, cident = _circle_polygon(ax, ay, radius, q0, q1, q2, q3, q4, q5, q6, q7)
            # A round agent touches with its circle (first piece only); its polygon pieces never touch.
            use = disc & (a == 0)
            u0x, u0y = tl.where(use, cx_, u0x), tl.where(use, cy_, u0y)
            u1x, u1y = tl.where(use, cx_, u1x), tl.where(use, cy_, u1y)
            nx, ny = tl.where(use, cnx, nx), tl.where(use, cny, ny)
            d0, d1 = tl.where(use, cdepth, d0), tl.where(use, -1.0, d1)
            hide = disc & (a > 0)
            d0, d1 = tl.where(hide, -1.0, d0), tl.where(hide, -1.0, d1)
            first, second = tl.where(use, cident, ident), tl.where(use, -1, ident + 1)
            on = (tl.load(enabled + lane * 2 * S + a, mask=mask, other=0) != 0) & (
                tl.load(enabled + lane * 2 * S + S + b, mask=mask, other=0) != 0
            )
            _put(scratch, slot, B, lane, mask, 0, u0x)
            _put(scratch, slot, B, lane, mask, 1, u0y)
            _put(scratch, slot, B, lane, mask, 2, nx)
            _put(scratch, slot, B, lane, mask, 3, ny)
            _put(scratch, slot, B, lane, mask, 4, tl.where(on, d0, -1.0))
            _put(scratch, slot, B, lane, mask, 16, first.to(tl.float32))
            _put(scratch, slot + 1, B, lane, mask, 0, u1x)
            _put(scratch, slot + 1, B, lane, mask, 1, u1y)
            _put(scratch, slot + 1, B, lane, mask, 2, nx)
            _put(scratch, slot + 1, B, lane, mask, 3, ny)
            _put(scratch, slot + 1, B, lane, mask, 4, tl.where(on, d1, -1.0))
            _put(scratch, slot + 1, B, lane, mask, 16, second.to(tl.float32))
    if CIRCLE:
        start = 1
        base = S
    else:
        start = 0
        base = 2 * S * S
    for piece in range((2 - start) * S):
        index = start + piece // S
        shape = piece % S
        agent = index == 0
        px_, py_ = tl.where(agent, ax, bx), tl.where(agent, ay, by)
        c_, s_ = tl.where(agent, ca, cb), tl.where(agent, sa, sb)
        dynamic = tl.where(agent, inv_agent, inv_block) > 0
        x0, y0 = _world(vertices, lane, index, shape, 0, px_, py_, c_, s_, mask, S)
        x1, y1 = _world(vertices, lane, index, shape, 1, px_, py_, c_, s_, mask, S)
        x2, y2 = _world(vertices, lane, index, shape, 2, px_, py_, c_, s_, mask, S)
        x3, y3 = _world(vertices, lane, index, shape, 3, px_, py_, c_, s_, mask, S)
        live = (tl.load(enabled + lane * 2 * S + index * S + shape, mask=mask, other=0) != 0) & dynamic
        for w in tl.static_range(4):
            first, second, nx, ny, plane = _wall_pair(w, x0, y0, x1, y1, x2, y2, x3, y3)
            for pick in tl.static_range(2):
                if pick == 0:
                    k = first
                else:
                    k = second
                slot = base + piece * 8 + w * 2 + pick
                vx = _corner(k, x0, x1, x2, x3)
                vy = _corner(k, y0, y1, y2, y3)
                _put(scratch, slot, B, lane, mask, 0, vx)
                _put(scratch, slot, B, lane, mask, 1, vy)
                _put(scratch, slot, B, lane, mask, 2, tl.zeros_like(vx) + nx)
                _put(scratch, slot, B, lane, mask, 3, tl.zeros_like(vx) + ny)
                _put(scratch, slot, B, lane, mask, 4, tl.where(live, plane - (nx * vx + ny * vy), -1.0))
                _put(scratch, slot, B, lane, mask, 16, k.to(tl.float32))


@triton.jit
def _pair(i, S: tl.constexpr, CIRCLE: tl.constexpr):
    """Bodies of slot ``i``: whether body ``a`` is the agent (else the static walls) and body ``b``'s index."""
    if CIRCLE:
        a_agent = i < S
        b = 1
    else:
        a_agent = i < 2 * S * S
        b = tl.where(i < 2 * S * S + 8 * S, tl.where(a_agent, 1, 0), 1)
    return a_agent, b


@triton.jit
def advance_kernel(
    body,
    bias,
    cached,
    identifiers,
    vertices,
    cog,
    inverse,
    radius_ptr,
    enabled,
    target,
    active,
    contacts_out,
    segments_out,
    lines,
    scratch,
    B,
    S: tl.constexpr,
    C: tl.constexpr,
    CIRCLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = lane < B
    mask = mask & (tl.load(active + lane, mask=mask, other=0) != 0)
    row = body + lane * 12
    ax, ay = tl.load(row, mask=mask, other=0.0).to(tl.float32), tl.load(row + 1, mask=mask, other=0.0).to(tl.float32)
    avx, avy = (
        tl.load(row + 2, mask=mask, other=0.0).to(tl.float32),
        tl.load(row + 3, mask=mask, other=0.0).to(tl.float32),
    )
    aa, aw = (
        tl.load(row + 4, mask=mask, other=0.0).to(tl.float32),
        tl.load(row + 5, mask=mask, other=0.0).to(tl.float32),
    )
    bx, by = (
        tl.load(row + 6, mask=mask, other=0.0).to(tl.float32),
        tl.load(row + 7, mask=mask, other=0.0).to(tl.float32),
    )
    bvx, bvy = (
        tl.load(row + 8, mask=mask, other=0.0).to(tl.float32),
        tl.load(row + 9, mask=mask, other=0.0).to(tl.float32),
    )
    ba, bw = (
        tl.load(row + 10, mask=mask, other=0.0).to(tl.float32),
        tl.load(row + 11, mask=mask, other=0.0).to(tl.float32),
    )
    brow = bias + lane * 6
    pax, pay, paw = (
        tl.load(brow, mask=mask, other=0.0).to(tl.float32),
        tl.load(brow + 1, mask=mask, other=0.0).to(tl.float32),
        tl.load(brow + 2, mask=mask, other=0.0).to(tl.float32),
    )
    pbx, pby, pbw = (
        tl.load(brow + 3, mask=mask, other=0.0).to(tl.float32),
        tl.load(brow + 4, mask=mask, other=0.0).to(tl.float32),
        tl.load(brow + 5, mask=mask, other=0.0).to(tl.float32),
    )
    gax, gay = (
        tl.load(cog + lane * 4, mask=mask, other=0.0).to(tl.float32),
        tl.load(cog + lane * 4 + 1, mask=mask, other=0.0).to(tl.float32),
    )
    gbx, gby = (
        tl.load(cog + lane * 4 + 2, mask=mask, other=0.0).to(tl.float32),
        tl.load(cog + lane * 4 + 3, mask=mask, other=0.0).to(tl.float32),
    )
    ima, iia = (
        tl.load(inverse + lane * 4, mask=mask, other=0.0).to(tl.float32),
        tl.load(inverse + lane * 4 + 1, mask=mask, other=0.0).to(tl.float32),
    )
    imb, iib = (
        tl.load(inverse + lane * 4 + 2, mask=mask, other=0.0).to(tl.float32),
        tl.load(inverse + lane * 4 + 3, mask=mask, other=0.0).to(tl.float32),
    )
    radius = tl.load(radius_ptr + lane, mask=mask, other=0.0).to(tl.float32)
    tx, ty = (
        tl.load(target + lane * 2, mask=mask, other=0.0).to(tl.float32),
        tl.load(target + lane * 2 + 1, mask=mask, other=0.0).to(tl.float32),
    )
    total = tl.zeros([BLOCK], dtype=tl.int64)
    for _ in range(10):
        # Agent PD control, then integration with the bias velocities, as Pymunk does.
        vax = avx + (100 * (tx - ax) - 20 * avx) * 0.01
        vay = avy + (100 * (ty - ay) - 20 * avy) * 0.01
        c, s = tl.cos(aa), tl.sin(aa)
        cax = ax + (c * gax - s * gay) + (vax + pax) * 0.01
        cay = ay + (s * gax + c * gay) + (vay + pay) * 0.01
        aa = aa + (aw + paw) * 0.01
        c, s = tl.cos(aa), tl.sin(aa)
        ax, ay = cax - (c * gax - s * gay), cay - (s * gax + c * gay)
        c, s = tl.cos(ba), tl.sin(ba)
        cbx = bx + (c * gbx - s * gby) + (bvx + pbx) * 0.01
        cby = by + (s * gbx + c * gby) + (bvy + pby) * 0.01
        ba = ba + (bw + pbw) * 0.01
        c, s = tl.cos(ba), tl.sin(ba)
        bx, by = cbx - (c * gbx - s * gby), cby - (s * gbx + c * gby)
        # Space.damping zeroes dynamic bodies after integration.
        vaw = aw
        vax, vay, vaw = tl.where(ima > 0, 0.0, vax), tl.where(ima > 0, 0.0, vay), tl.where(ima > 0, 0.0, vaw)
        vbx, vby, vbw = tl.where(imb > 0, 0.0, bvx), tl.where(imb > 0, 0.0, bvy), tl.where(imb > 0, 0.0, bw)
        _contacts(scratch, B, lane, mask, vertices, enabled, radius, ima, imb, ax, ay, aa, bx, by, ba, S, CIRCLE)
        # Contact geometry, warm start with the impulses of contacts that persist.
        for i in range(C):
            a_agent, b = _pair(i, S, CIRCLE)
            px, py = _get(scratch, i, B, lane, mask, 0), _get(scratch, i, B, lane, mask, 1)
            nx, ny = _get(scratch, i, B, lane, mask, 2), _get(scratch, i, B, lane, mask, 3)
            depth = _get(scratch, i, B, lane, mask, 4)
            ident = _get(scratch, i, B, lane, mask, 16).to(tl.int64)
            rax = tl.where(a_agent, px - cax, px)
            ray = tl.where(a_agent, py - cay, py)
            rbx = tl.where(b == 0, px - cax, px - cbx)
            rby = tl.where(b == 0, py - cay, py - cby)
            cra = rax * ny - ray * nx
            crb = rbx * ny - rby * nx
            ia_m, ia_i = tl.where(a_agent, ima, 0.0), tl.where(a_agent, iia, 0.0)
            ib_m, ib_i = tl.where(b == 0, ima, imb), tl.where(b == 0, iia, iib)
            mass = tl.maximum(ia_m + ib_m + cra * cra * ia_i + crb * crb * ib_i, 1e-12)
            on = depth >= 0
            total += on.to(tl.int64)
            old = tl.load(identifiers + lane * C + i, mask=mask, other=-1)
            impulse = tl.load(cached + lane * C + i, mask=mask, other=0.0).to(tl.float32)
            impulse = tl.where(on & (ident == old), impulse, 0.0)
            tl.store(identifiers + lane * C + i, ident, mask=mask)
            _put(scratch, i, B, lane, mask, 5, rax)
            _put(scratch, i, B, lane, mask, 6, ray)
            _put(scratch, i, B, lane, mask, 7, rbx)
            _put(scratch, i, B, lane, mask, 8, rby)
            _put(scratch, i, B, lane, mask, 9, cra)
            _put(scratch, i, B, lane, mask, 10, crb)
            _put(scratch, i, B, lane, mask, 11, mass)
            _put(scratch, i, B, lane, mask, 12, _RATE * tl.maximum(depth - 0.1, 0.0))
            _put(scratch, i, B, lane, mask, 13, impulse)
            _put(scratch, i, B, lane, mask, 14, tl.zeros_like(impulse))
            _put(scratch, i, B, lane, mask, 15, on.to(tl.float32))
            jx, jy = impulse * nx, impulse * ny
            vax = vax - tl.where(a_agent, jx * ia_m, 0.0) + tl.where(b == 0, jx * ib_m, 0.0)
            vay = vay - tl.where(a_agent, jy * ia_m, 0.0) + tl.where(b == 0, jy * ib_m, 0.0)
            vaw = vaw - tl.where(a_agent, impulse * cra * ia_i, 0.0) + tl.where(b == 0, impulse * crb * ib_i, 0.0)
            vbx = vbx + tl.where(b == 1, jx * ib_m, 0.0)
            vby = vby + tl.where(b == 1, jy * ib_m, 0.0)
            vbw = vbw + tl.where(b == 1, impulse * crb * ib_i, 0.0)
        qax, qay, qaw = tl.zeros_like(vax), tl.zeros_like(vax), tl.zeros_like(vax)
        qbx, qby, qbw = tl.zeros_like(vax), tl.zeros_like(vax), tl.zeros_like(vax)
        for _ in range(10):
            for i in range(C):
                # Slots with no contact in this block of environments change nothing; skip them.
                on = _get(scratch, i, B, lane, mask, 15) != 0
                if tl.max(on.to(tl.int32), axis=0) > 0:
                    a_agent, b = _pair(i, S, CIRCLE)
                    nx, ny = _get(scratch, i, B, lane, mask, 2), _get(scratch, i, B, lane, mask, 3)
                    rax, ray = _get(scratch, i, B, lane, mask, 5), _get(scratch, i, B, lane, mask, 6)
                    rbx, rby = _get(scratch, i, B, lane, mask, 7), _get(scratch, i, B, lane, mask, 8)
                    cra, crb = _get(scratch, i, B, lane, mask, 9), _get(scratch, i, B, lane, mask, 10)
                    mass, goal = _get(scratch, i, B, lane, mask, 11), _get(scratch, i, B, lane, mask, 12)
                    old, old_bias = _get(scratch, i, B, lane, mask, 13), _get(scratch, i, B, lane, mask, 14)
                    ia_m, ia_i = tl.where(a_agent, ima, 0.0), tl.where(a_agent, iia, 0.0)
                    ib_m, ib_i = tl.where(b == 0, ima, imb), tl.where(b == 0, iia, iib)
                    # Relative normal speed of b against a, for the velocities and the bias velocities.
                    ux = tl.where(a_agent, vax, 0.0)
                    uy = tl.where(a_agent, vay, 0.0)
                    uw = tl.where(a_agent, vaw, 0.0)
                    wx, wy, ww = tl.where(b == 0, vax, vbx), tl.where(b == 0, vay, vby), tl.where(b == 0, vaw, vbw)
                    speed = (wx - rby * ww - ux + ray * uw) * nx + (wy + rbx * ww - uy - rax * uw) * ny
                    ux = tl.where(a_agent, qax, 0.0)
                    uy = tl.where(a_agent, qay, 0.0)
                    uw = tl.where(a_agent, qaw, 0.0)
                    wx, wy, ww = tl.where(b == 0, qax, qbx), tl.where(b == 0, qay, qby), tl.where(b == 0, qaw, qbw)
                    push = (wx - rby * ww - ux + ray * uw) * nx + (wy + rbx * ww - uy - rax * uw) * ny
                    impulse = tl.where(on, tl.maximum(old - speed / mass, 0.0), 0.0)
                    correction = tl.where(on, tl.maximum(old_bias + (goal - push) / mass, 0.0), 0.0)
                    d, e = impulse - old, correction - old_bias
                    jx, jy = d * nx, d * ny
                    vax = vax - tl.where(a_agent, jx * ia_m, 0.0) + tl.where(b == 0, jx * ib_m, 0.0)
                    vay = vay - tl.where(a_agent, jy * ia_m, 0.0) + tl.where(b == 0, jy * ib_m, 0.0)
                    vaw = vaw - tl.where(a_agent, d * cra * ia_i, 0.0) + tl.where(b == 0, d * crb * ib_i, 0.0)
                    vbx = vbx + tl.where(b == 1, jx * ib_m, 0.0)
                    vby = vby + tl.where(b == 1, jy * ib_m, 0.0)
                    vbw = vbw + tl.where(b == 1, d * crb * ib_i, 0.0)
                    jx, jy = e * nx, e * ny
                    qax = qax - tl.where(a_agent, jx * ia_m, 0.0) + tl.where(b == 0, jx * ib_m, 0.0)
                    qay = qay - tl.where(a_agent, jy * ia_m, 0.0) + tl.where(b == 0, jy * ib_m, 0.0)
                    qaw = qaw - tl.where(a_agent, e * cra * ia_i, 0.0) + tl.where(b == 0, e * crb * ib_i, 0.0)
                    qbx = qbx + tl.where(b == 1, jx * ib_m, 0.0)
                    qby = qby + tl.where(b == 1, jy * ib_m, 0.0)
                    qbw = qbw + tl.where(b == 1, e * crb * ib_i, 0.0)
                    _put(scratch, i, B, lane, mask, 13, impulse)
                    _put(scratch, i, B, lane, mask, 14, correction)
        avx, avy, aw = vax, vay, vaw
        bvx, bvy, bw = vbx, vby, vbw
        pax, pay, paw, pbx, pby, pbw = qax, qay, qaw, qbx, qby, qbw
        for i in range(C):
            tl.store(cached + lane * C + i, _get(scratch, i, B, lane, mask, 13), mask=mask)
    tl.store(row, ax, mask=mask)
    tl.store(row + 1, ay, mask=mask)
    tl.store(row + 2, avx, mask=mask)
    tl.store(row + 3, avy, mask=mask)
    tl.store(row + 4, aa, mask=mask)
    tl.store(row + 5, aw, mask=mask)
    tl.store(row + 6, bx, mask=mask)
    tl.store(row + 7, by, mask=mask)
    tl.store(row + 8, bvx, mask=mask)
    tl.store(row + 9, bvy, mask=mask)
    tl.store(row + 10, ba, mask=mask)
    tl.store(row + 11, bw, mask=mask)
    tl.store(brow, pax, mask=mask)
    tl.store(brow + 1, pay, mask=mask)
    tl.store(brow + 2, paw, mask=mask)
    tl.store(brow + 3, pbx, mask=mask)
    tl.store(brow + 4, pby, mask=mask)
    tl.store(brow + 5, pbw, mask=mask)
    # Render segments of the final contacts, active first.
    _contacts(scratch, B, lane, mask, vertices, enabled, radius, ima, imb, ax, ay, aa, bx, by, ba, S, CIRCLE)
    out = lines + lane * (90 * 8)
    # Rows from the new count on must be zero; only rows below the old count can hold segments.
    old_rows = tl.load(segments_out + lane, mask=mask, other=0)
    for r in range(tl.max(old_rows, axis=0)):
        for f in tl.static_range(8):
            tl.store(out + r * 8 + f, tl.zeros_like(ax), mask=mask & (r < old_rows))
    rows = tl.zeros([BLOCK], dtype=tl.int64)
    for i in range(C):
        px, py = _get(scratch, i, B, lane, mask, 0), _get(scratch, i, B, lane, mask, 1)
        nx, ny = _get(scratch, i, B, lane, mask, 2), _get(scratch, i, B, lane, mask, 3)
        depth = _get(scratch, i, B, lane, mask, 4)
        on = mask & (depth >= 0)
        at = out + rows * 8
        tl.store(at, px - nx * (depth + 2), mask=on)
        tl.store(at + 1, py - ny * (depth + 2), mask=on)
        tl.store(at + 2, px + nx * 2, mask=on)
        tl.store(at + 3, py + ny * 2, mask=on)
        tl.store(at + 4, tl.zeros_like(px) + 231.0, mask=on)
        tl.store(at + 5, tl.zeros_like(px) + 76.0, mask=on)
        tl.store(at + 6, tl.zeros_like(px) + 60.0, mask=on)
        tl.store(at + 7, tl.zeros_like(px) + 255.0, mask=on)
        rows += on.to(tl.int64)
    tl.store(contacts_out + lane, total, mask=mask)
    tl.store(segments_out + lane, rows, mask=mask)


def advance(body, bias, cached, identifiers, geometry, target, circle, active, contacts, segments, lines, scratch):
    """Step every ``active`` environment in place, as ``jax_physics.advance`` does.

    ``geometry`` is (vertices, centers, inverse, radius, enabled as uint8). ``contacts`` gets the contacts
    summed over substeps and ``segments`` the number of render segments written to ``lines``.
    ``scratch`` holds ``FIELDS * cached.shape[1] * B`` float32 values. The kernel computes in float32 and
    stores the State in its float64 fields.
    """
    vertices, centers, inverse, radius, enabled = geometry
    count = body.shape[0]
    block = 64
    advance_kernel[(triton.cdiv(count, block),)](
        body,
        bias,
        cached,
        identifiers,
        vertices,
        centers,
        inverse,
        radius,
        enabled,
        target,
        active,
        contacts,
        segments,
        lines,
        scratch,
        count,
        S=vertices.shape[2],
        C=cached.shape[1],
        CIRCLE=bool(circle),
        BLOCK=block,
    )


@triton.jit
def _remainder(a, b):
    """``torch.remainder`` of float64 values: ``fmod`` with the sign of the divisor."""
    mod = libdevice.fmod(a, b)
    return tl.where((mod != 0) & ((b < 0) != (mod < 0)), mod + b, mod)


@triton.jit
def record_kernel(
    body,
    actions,
    relative,
    active,
    lengths,
    history,
    codes,
    capacity,
    latest_value,
    latest_type,
    latest_present,
    observation,
    goal,
    symmetry,
    reward,
    success,
    B,
    CODE: tl.constexpr,
    TAG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """After a step: append the actions to the histories, set the latest action, the observation, reward and success."""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < B
    on = valid & (tl.load(active + lane, mask=valid, other=0) != 0)
    length = tl.load(lengths + lane, mask=valid, other=0)
    at = history + (lane * capacity + length) * 2
    tl.store(at, tl.where(on, tl.load(actions + lane * 2, mask=valid, other=0).to(tl.float64), 0.0), mask=valid)
    tl.store(
        at + 1, tl.where(on, tl.load(actions + lane * 2 + 1, mask=valid, other=0).to(tl.float64), 0.0), mask=valid
    )
    tl.store(codes + lane * capacity + length, tl.where(on, CODE, 0).to(tl.uint8), mask=valid)
    tl.store(lengths + lane, length + on.to(tl.int64), mask=valid)
    tl.store(latest_value + lane * 2, tl.load(relative + lane * 2, mask=on, other=0).to(tl.float64), mask=on)
    tl.store(latest_value + lane * 2 + 1, tl.load(relative + lane * 2 + 1, mask=on, other=0).to(tl.float64), mask=on)
    tl.store(latest_type + lane * 2, tl.full([BLOCK], 4, tl.uint8), mask=on)
    tl.store(latest_type + lane * 2 + 1, tl.full([BLOCK], TAG, tl.uint8), mask=on)
    tl.store(latest_present + lane, tl.full([BLOCK], 1, tl.int1), mask=on)
    row = body + lane * 12
    two_pi = tl.full([BLOCK], 6.283185307179586, tl.float64)
    obs0 = tl.load(row, mask=valid, other=0.0)
    obs1 = tl.load(row + 1, mask=valid, other=0.0)
    obs2 = tl.load(row + 6, mask=valid, other=0.0)
    obs3 = tl.load(row + 7, mask=valid, other=0.0)
    obs4 = _remainder(tl.load(row + 10, mask=valid, other=0.0), two_pi)
    obs5 = tl.load(row + 2, mask=valid, other=0.0)
    obs6 = tl.load(row + 3, mask=valid, other=0.0)
    out = observation + lane * 7
    tl.store(out, obs0, mask=valid)
    tl.store(out + 1, obs1, mask=valid)
    tl.store(out + 2, obs2, mask=valid)
    tl.store(out + 3, obs3, mask=valid)
    tl.store(out + 4, obs4, mask=valid)
    tl.store(out + 5, obs5, mask=valid)
    tl.store(out + 6, obs6, mask=valid)
    g = goal + lane * 7
    d0 = tl.load(g, mask=valid, other=0.0) - obs0
    d1 = tl.load(g + 1, mask=valid, other=0.0) - obs1
    d2 = tl.load(g + 2, mask=valid, other=0.0) - obs2
    d3 = tl.load(g + 3, mask=valid, other=0.0) - obs3
    d4 = tl.load(g + 4, mask=valid, other=0.0) - obs4
    d5 = tl.load(g + 5, mask=valid, other=0.0) - obs5
    d6 = tl.load(g + 6, mask=valid, other=0.0) - obs6
    near = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
    tl.store(reward + lane, -tl.sqrt(near + d4 * d4 + d5 * d5 + d6 * d6), mask=valid)
    period = tl.load(symmetry + lane, mask=valid, other=1.0)
    angle = _remainder(tl.abs(d4), period)
    hit = (tl.sqrt(near) < 20) & (tl.minimum(angle, period - angle) < 0.3490658503988659)
    tl.store(success + lane, hit, mask=valid)


def record(body, actions, relative, active, lengths, history, codes, latest, observation, goal, symmetry):
    """Launch ``record_kernel``; return the reward and success. ``latest`` is (value, type, present)."""
    count = body.shape[0]
    reward = torch.empty(count, dtype=torch.float64, device=body.device)
    success = torch.empty(count, dtype=torch.bool, device=body.device)
    if actions.dtype in (torch.int32, torch.int64):
        code = (torch.int32, torch.int64).index(actions.dtype) + 2
    else:
        code = (torch.float32, torch.float64).index(actions.dtype)
    block = 128
    record_kernel[(triton.cdiv(count, block),)](
        body,
        actions,
        relative,
        active,
        lengths,
        history,
        codes,
        history.shape[1],
        *latest,
        observation,
        goal,
        symmetry,
        reward,
        success,
        count,
        CODE=code,
        TAG=0 if relative.dtype == torch.float32 else 1,
        BLOCK=block,
    )
    return reward, success
