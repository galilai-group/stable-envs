"""Numba CPU port of the PushT contact solver in ``jax_physics``: the same math, in parallel over environments.

Ports the solver that ``jax_physics`` adapts from fracapuano/envX at d26c817.
Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.

The wall contacts (``SegmentToPoly``, ``ContactPoints``) and the arbiter contact cache port Munk2D at commit
ade7ed72849e. Copyright (c) 2025 Victor Blomqvist, Copyright (c) 2007-2024 Scott Lembcke and Howling Moon
Software. MIT license; see licenses/Munk2D-MIT.txt.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import numba_broadphase as broadphase

_DT = 0.01
_SUBSTEPS = 10
_ITERATIONS = 10
#: Munk2D's default ``collisionSlop`` and ``collisionBias`` come from float literals: ``0.1f`` and ``(1 - 0.1f)**60``.
_SLOP = float(np.float32(0.1))
_BIAS = 1 - (float(np.float32(1) - np.float32(0.1)) ** 60) ** _DT
#: The four wall segments (a x, a y, b x, b y) of the Pymunk scene, each of radius ``_WALL_RADIUS``.
_WALLS = np.array(
    ((5.0, 506.0, 5.0, 5.0), (5.0, 5.0, 506.0, 5.0), (506.0, 5.0, 506.0, 506.0), (5.0, 506.0, 506.0, 506.0))
)
_WALL_RADIUS = 2.0
#: ``CPFLOAT_MIN``, which Munk2D adds to the clipping denominators.
_TINY = 2.2250738585072014e-308
_SEGMENT_COLOR = np.array((231.0, 76.0, 60.0, 255.0))
#: Munk2D drops an arbiter that has had no contacts for ``collisionPersistence`` (3) steps; until then its
#: contact ids are stored as ``id + _STALE * steps``.
_STALE = 1 << 16


@njit(cache=True, inline="always")
def _outward(vertices, k):
    """Unit outward normal of edge ``k`` of a counterclockwise quad."""
    j = (k + 1) % 4
    ex, ey = vertices[j, 0] - vertices[k, 0], vertices[j, 1] - vertices[k, 1]
    length = math.sqrt(ex * ex + ey * ey)
    return ey / length, -ex / length


@njit(cache=True)
def _circle_polygon(cx, cy, radius, vertices, out, slot):
    face, best = 0, -np.inf
    for k in range(4):
        nx, ny = _outward(vertices, k)
        separation = (cx - vertices[k, 0]) * nx + (cy - vertices[k, 1]) * ny
        if separation > best:
            face, best = k, separation
    nx, ny = _outward(vertices, face)
    sx, sy = vertices[face, 0], vertices[face, 1]
    j = (face + 1) % 4
    ex, ey = vertices[j, 0] - sx, vertices[j, 1] - sy
    t = ((cx - sx) * ex + (cy - sy) * ey) / (ex * ex + ey * ey)
    t = min(max(t, 0.0), 1.0)
    px, py = sx + t * ex, sy + t * ey
    dx, dy = cx - px, cy - py
    distance = max(math.sqrt(dx * dx + dy * dy), 1e-12)
    if best <= 0:
        depth = radius - best
    else:
        nx, ny = dx / distance, dy / distance
        depth = radius - distance
    out[slot, 0], out[slot, 1], out[slot, 2], out[slot, 3], out[slot, 4] = px, py, -nx, -ny, depth
    # Munk2D hashes every circle contact as 0.
    return 0


@njit(cache=True, inline="always")
def _reference_face(vertices, other):
    face, best = 0, -np.inf
    for k in range(4):
        nx, ny = _outward(vertices, k)
        low = np.inf
        for m in range(4):
            low = min(low, other[m, 0] * nx + other[m, 1] * ny)
        depth = low - (vertices[k, 0] * nx + vertices[k, 1] * ny)
        if depth > best:
            face, best = k, depth
    return face, best


@njit(cache=True)
def _polygon_polygon(a, b, out, slot, ids):
    ai, ad = _reference_face(a, b)
    bi, bd = _reference_face(b, a)
    flip = bd > ad
    ref, inc = (b, a) if flip else (a, b)
    face = bi if flip else ai
    nx, ny = _outward(ref, face)
    edge, lowest = 0, np.inf
    for k in range(4):
        mx, my = _outward(inc, k)
        dot = mx * nx + my * ny
        if dot < lowest:
            edge, lowest = k, dot
    fx, fy = inc[edge, 0], inc[edge, 1]
    gx, gy = inc[(edge + 1) % 4, 0], inc[(edge + 1) % 4, 1]
    tx, ty = -ny, nx
    r0 = ref[face, 0] * tx + ref[face, 1] * ty
    r1 = ref[(face + 1) % 4, 0] * tx + ref[(face + 1) % 4, 1] * ty
    lo, hi = min(r0, r1), max(r0, r1)
    denom = (gx - fx) * tx + (gy - fy) * ty
    safe = denom if abs(denom) > 1e-12 else 1.0
    first = fx * tx + fy * ty
    e0, e1 = (lo - first) / safe, (hi - first) / safe
    e0, e1 = min(e0, e1), max(e0, e1)
    t0, t1 = min(max(e0, 0.0), 1.0), min(max(e1, 0.0), 1.0)
    p0x, p0y = fx + t0 * (gx - fx), fy + t0 * (gy - fy)
    p1x, p1y = fx + t1 * (gx - fx), fy + t1 * (gy - fy)
    rx, ry = ref[face, 0], ref[face, 1]
    d0 = (rx - p0x) * nx + (ry - p0y) * ny
    d1 = (rx - p1x) * nx + (ry - p1y) * ny
    overlap = max(ad, bd) <= 0 and (abs(denom) > 1e-12 or lo <= first <= hi)
    if not overlap:
        d0, d1 = -1.0, -1.0
    if math.sqrt((p1x - p0x) ** 2 + (p1y - p0y) ** 2) <= 1e-9:
        d1 = -1.0
    sign = -1.0 if flip else 1.0
    out[slot, 0], out[slot, 1], out[slot, 2], out[slot, 3], out[slot, 4] = p0x, p0y, sign * nx, sign * ny, d0
    out[slot + 1, 0], out[slot + 1, 1], out[slot + 1, 2], out[slot + 1, 3] = p1x, p1y, sign * nx, sign * ny
    out[slot + 1, 4] = d1
    base = face * 8 + edge * 2
    ids[slot], ids[slot + 1] = base, base + 1


@njit(cache=True)
def _segment_polygon(ax, ay, bx, by, radius, v, out, slot, ids):
    """Munk2D ``SegmentToPoly`` for a static segment ``a``-``b`` and a counterclockwise quad ``v``.

    Writes up to two contacts to ``slot`` and ``slot + 1``, in Munk2D's order: the point on the quad, the
    normal from the segment into the quad, the depth (-1 for no contact) and the id ``4 * end + vertex`` of
    the segment end and quad vertex that Munk2D hashes for the cached impulse.
    """
    dx, dy = bx - ax, by - ay
    length = math.sqrt(dx * dx + dy * dy)
    tnx, tny = dy / length, -dx / length
    for r in range(slot, slot + 2):
        out[r, 0], out[r, 1], out[r, 2], out[r, 3], out[r, 4] = v[0, 0], v[0, 1], tnx, tny, -1.0
        ids[r] = -1
    # The axis of least separation (EPA when the cores overlap); n points from the segment into the quad.
    low, high = np.inf, -np.inf
    for k in range(4):
        p = tnx * v[k, 0] + tny * v[k, 1]
        low, high = min(low, p), max(high, p)
    c = tnx * ax + tny * ay
    nx, ny, d = tnx, tny, low - c
    if c - high > d:
        nx, ny, d = -tnx, -tny, c - high
    # A separation along the segment normal bounds the distance from below, so a far quad cannot touch.
    if d > radius:
        return
    for k in range(4):
        mx, my = _outward(v, k)
        s = min(mx * ax + my * ay, mx * bx + my * by) - (mx * v[k, 0] + my * v[k, 1])
        if s > d:
            nx, ny, d = -mx, -my, s
    if d > 0:
        # Separated cores: GJK's closest points of the segment and the quad.
        best = np.inf
        for k in range(4):
            t = min(max(((v[k, 0] - ax) * dx + (v[k, 1] - ay) * dy) / (length * length), 0.0), 1.0)
            qx, qy = ax + t * dx, ay + t * dy
            gap = (v[k, 0] - qx) ** 2 + (v[k, 1] - qy) ** 2
            if gap < best:
                best, nx, ny = gap, v[k, 0] - qx, v[k, 1] - qy
            j = (k + 1) % 4
            ex, ey = v[j, 0] - v[k, 0], v[j, 1] - v[k, 1]
            for px, py in ((ax, ay), (bx, by)):
                t = min(max(((px - v[k, 0]) * ex + (py - v[k, 1]) * ey) / (ex * ex + ey * ey), 0.0), 1.0)
                qx, qy = v[k, 0] + t * ex, v[k, 1] + t * ey
                gap = (qx - px) ** 2 + (qy - py) ** 2
                if gap < best:
                    best, nx, ny = gap, qx - px, qy - py
        d = math.sqrt(best)
        nx, ny = nx / d, ny / d
    if d - radius > 0:
        return
    # Support edges: the segment's face toward the quad, and the quad's edge most against n.
    if tnx * nx + tny * ny > 0:
        e1ax, e1ay, e1bx, e1by, end_a, end_b = ax, ay, bx, by, 0, 1
    else:
        e1ax, e1ay, e1bx, e1by, end_a, end_b = bx, by, ax, ay, 1, 0
    i1, top = 0, -np.inf
    for k in range(4):
        p = -(nx * v[k, 0] + ny * v[k, 1])
        if p > top:
            i1, top = k, p
    i0, i2 = (i1 + 3) % 4, (i1 + 1) % 4
    # Munk2D's plane i holds the normal of the edge ending at vertex i.
    p1x, p1y = _outward(v, i0)
    p2x, p2y = _outward(v, i1)
    if -(nx * p1x + ny * p1y) > -(nx * p2x + ny * p2y):
        first, second = i0, i1
    else:
        first, second = i1, i2
    e2ax, e2ay, e2bx, e2by = v[first, 0], v[first, 1], v[second, 0], v[second, 1]
    d1a = e1ax * ny - e1ay * nx
    d1b = e1bx * ny - e1by * nx
    d2a = e2ax * ny - e2ay * nx
    d2b = e2bx * ny - e2by * nx
    inv1 = 1.0 / (d1b - d1a + _TINY)
    inv2 = 1.0 / (d2b - d2a + _TINY)
    row = slot
    for pick in range(2):
        if pick == 0:
            t1, t2, ident = (d2b - d1a) * inv1, (d1a - d2a) * inv2, end_a * 4 + second
        else:
            t1, t2, ident = (d2a - d1a) * inv1, (d1b - d2a) * inv2, end_b * 4 + first
        t1, t2 = min(max(t1, 0.0), 1.0), min(max(t2, 0.0), 1.0)
        q1x = nx * radius + (e1ax * (1.0 - t1) + e1bx * t1)
        q1y = ny * radius + (e1ay * (1.0 - t1) + e1by * t1)
        q2x = e2ax * (1.0 - t2) + e2bx * t2
        q2y = e2ay * (1.0 - t2) + e2by * t2
        dist = (q2x - q1x) * nx + (q2y - q1y) * ny
        if dist <= 0:
            out[row, 0], out[row, 1], out[row, 2], out[row, 3], out[row, 4] = q2x, q2y, nx, ny, -dist
            ids[row] = ident
            row += 1


@njit(cache=True)
def _contacts(body, vertices, inverse, radius, enabled, circle, world, out, ids, pairs, tight):
    """Fill ``out`` rows (point x, y, normal x, y, depth), ``ids`` and ``pairs`` in ``jax_physics.contacts`` order.

    Also sets ``tight`` to each shape's bounding box, as Munk2D caches it, in ``numba_broadphase`` leaf order.
    """
    shapes = vertices.shape[1]
    for index in range(2):
        c, s = math.cos(body[index, 4]), math.sin(body[index, 4])
        for shape in range(shapes):
            leaf = broadphase.SHAPES * index + shape
            tight[leaf, 0], tight[leaf, 1], tight[leaf, 2], tight[leaf, 3] = np.inf, np.inf, -np.inf, -np.inf
            for k in range(4):
                vx, vy = vertices[index, shape, k, 0], vertices[index, shape, k, 1]
                x, y = body[index, 0] + c * vx - s * vy, body[index, 1] + s * vx + c * vy
                world[index, shape, k, 0], world[index, shape, k, 1] = x, y
                tight[leaf, 0], tight[leaf, 1] = min(tight[leaf, 0], x), min(tight[leaf, 1], y)
                tight[leaf, 2], tight[leaf, 3] = max(tight[leaf, 2], x), max(tight[leaf, 3], y)
    if radius > 0:
        x, y = body[0, 0], body[0, 1]
        tight[0, 0], tight[0, 1], tight[0, 2], tight[0, 3] = x - radius, y - radius, x + radius, y + radius
    slot = 0
    for a in range(1 if circle else shapes):
        for b in range(shapes):
            on = enabled[0, a] and enabled[1, b]
            if circle:
                ids[slot] = _circle_polygon(body[0, 0], body[0, 1], radius, world[1, b], out, slot)
                pairs[slot, 0], pairs[slot, 1] = 0, 1
                if not on:
                    out[slot, 4] = -1.0
                slot += 1
                continue
            if radius > 0 and a == 0:
                ids[slot] = _circle_polygon(body[0, 0], body[0, 1], radius, world[1, b], out, slot)
                for f in range(4):
                    out[slot + 1, f] = out[slot, f]
                out[slot + 1, 4] = -1.0
                ids[slot + 1] = -1
            else:
                _polygon_polygon(world[0, a], world[1, b], out, slot, ids)
                if radius > 0:
                    out[slot, 4], out[slot + 1, 4] = -1.0, -1.0
            for p in range(2):
                pairs[slot + p, 0], pairs[slot + p, 1] = 0, 1
                if not on:
                    out[slot + p, 4] = -1.0
            slot += 2
    for index in range(1 if circle else 0, 2):
        for shape in range(shapes):
            live = enabled[index, shape] and inverse[index, 0] > 0
            box = tight[broadphase.SHAPES * index + shape]
            for w in range(4):
                # The wall's line is x (walls 0, 2) or y (walls 1, 3) = 5 or 506; a box farther than the wall
                # radius from it cannot touch.
                axis, line = w % 2, 5.0 if w < 2 else 506.0
                if live and box[axis] <= line + _WALL_RADIUS and box[axis + 2] >= line - _WALL_RADIUS:
                    _segment_polygon(
                        _WALLS[w, 0],
                        _WALLS[w, 1],
                        _WALLS[w, 2],
                        _WALLS[w, 3],
                        _WALL_RADIUS,
                        world[index, shape],
                        out,
                        slot,
                        ids,
                    )
                else:
                    out[slot, 4], out[slot + 1, 4] = -1.0, -1.0
                    ids[slot], ids[slot + 1] = -1, -1
                for p in range(2):
                    pairs[slot + p, 0], pairs[slot + p, 1] = 2, index
                slot += 2
    return slot


@njit(cache=True, inline="always")
def _apply(vel, a, b, impulse, nx, ny, ca, cb, inv):
    vel[a, 0] -= impulse * nx * inv[a, 0]
    vel[a, 1] -= impulse * ny * inv[a, 0]
    vel[a, 2] -= impulse * ca * inv[a, 1]
    vel[b, 0] += impulse * nx * inv[b, 0]
    vel[b, 1] += impulse * ny * inv[b, 0]
    vel[b, 2] += impulse * cb * inv[b, 1]


@njit(cache=True, inline="always")
def _normal_speed(vel, a, b, ra, rb, nx, ny):
    x = vel[b, 0] - rb[1] * vel[b, 2] - vel[a, 0] + ra[1] * vel[a, 2]
    y = vel[b, 1] + rb[0] * vel[b, 2] - vel[a, 1] - ra[0] * vel[a, 2]
    return x * nx + y * ny


@njit(cache=True, inline="always")
def _group(i, walls, circle):
    """First slot and slot count of the arbiter (shape pair) that contact slot ``i`` belongs to."""
    if i >= walls:
        return walls + (i - walls) // 2 * 2, 2
    if circle:
        return i, 1
    return i // 2 * 2, 2


@njit(cache=True)
def _warm(cached, identifiers, ids, active, walls, circle, impulses, fresh):
    """Set each active contact's starting impulse from the cached contact of its arbiter with the same id.

    As in Munk2D, an arbiter that had no contacts for one or two steps keeps its old contacts and ids
    (``identifiers`` then holds ``id + _STALE * steps``). When it touches again it takes their impulses,
    but its first step does not apply them to the bodies (``fresh`` is False).
    """
    for i in range(cached.shape[0]):
        impulses[i] = 0.0
        fresh[i] = True
        if not active[i]:
            continue
        start, size = _group(i, walls, circle)
        for k in range(start, start + size):
            if identifiers[k] >= 0 and identifiers[k] % _STALE == ids[i]:
                impulses[i] = cached[k]
            if identifiers[k] >= _STALE:
                fresh[i] = False


@njit(cache=True)
def _keep(cached, identifiers, ids, active, walls, circle, impulses):
    """Store the solved impulses and ids; an arbiter without contacts ages its old ones (see ``_warm``)."""
    i = 0
    while i < cached.shape[0]:
        start, size = _group(i, walls, circle)
        touching = False
        for k in range(start, start + size):
            touching = touching or active[k]
        for k in range(start, start + size):
            if touching:
                cached[k] = impulses[k] if active[k] else 0.0
                identifiers[k] = ids[k] if active[k] else -1
            elif identifiers[k] >= 0 and identifiers[k] < 2 * _STALE:
                identifiers[k] += _STALE
            else:
                cached[k], identifiers[k] = 0.0, -1
        i = start + size


@njit(cache=True)
def _visit_order(calls, reported, active, walls, circle, shapes, seen, sequence):
    """Write the active contact slots to ``sequence`` by arbiter, in broadphase report order; return their count."""
    for i in range(seen.shape[0]):
        seen[i] = False
    live = 0
    for c in range(reported):
        x, y = calls[c, 0], calls[c, 1]
        if x >= broadphase.LEAVES or (y < broadphase.LEAVES and y // broadphase.SHAPES < x // broadphase.SHAPES):
            x, y = y, x
        index, shape = x // broadphase.SHAPES, x % broadphase.SHAPES
        if y >= broadphase.LEAVES:
            if circle and index == 0:
                continue
            piece = shape if circle else index * shapes + shape
            start, size = walls + piece * 8 + (y - broadphase.LEAVES) * 2, 2
        elif y // broadphase.SHAPES == index:
            continue
        elif circle:
            start, size = y % broadphase.SHAPES, 1
        else:
            start, size = 2 * (shape * shapes + y % broadphase.SHAPES), 2
        for i in range(start, start + size):
            if active[i] and not seen[i]:
                seen[i] = True
                sequence[live] = i
                live += 1
    for i in range(seen.shape[0]):
        if active[i] and not seen[i]:
            sequence[live] = i
            live += 1
    return live


@njit(cache=True, inline="always")
def _control(velocity, position, target, single):
    """The agent velocity after one PD substep, in float32 when the actions are float32, as the source computes."""
    if single:
        step = np.float32(
            np.float32(np.float32(100.0) * np.float32(np.float32(target) - np.float32(position)))
            + np.float32(-20.0 * velocity)
        )
        return float(np.float32(velocity) + step * np.float32(_DT))
    return velocity + (100 * (target - position) - 20 * velocity) * _DT


@njit(cache=True)
def _lane(
    body,
    bias,
    cached,
    identifiers,
    tree,
    bounds,
    order,
    vertices,
    cog,
    inverse,
    radius,
    enabled,
    target,
    single,
    circle,
    lines,
):
    count = cached.shape[0]
    shapes = vertices.shape[1]
    world = np.empty((2, shapes, 4, 2))
    out = np.empty((count, 5))
    ra = np.empty((count, 2))
    rb = np.empty((count, 2))
    ca = np.empty(count)
    cb = np.empty(count)
    mass = np.empty(count)
    goal = np.empty(count)
    impulses = np.empty(count)
    corrections = np.empty(count)
    inv = np.zeros((3, 2))
    inv[:2] = inverse
    centers = np.zeros((3, 2))
    vel = np.zeros((3, 3))
    push = np.zeros((3, 3))
    tight = np.zeros((broadphase.LEAVES, 4))
    moving = np.zeros((broadphase.LEAVES, 2))
    ids = np.empty(count, np.int64)
    sequence = np.empty(count, np.int64)
    pairs = np.empty((count, 2), np.int64)
    calls = np.empty((broadphase.LEAVES * broadphase.PAIRS, 2), np.int64)
    stack = np.empty(2 * broadphase.NODES + 2, np.int64)
    active = np.empty(count, np.bool_)
    fresh = np.empty(count, np.bool_)
    seen = np.empty(count, np.bool_)
    refit = np.empty(broadphase.LEAVES, np.bool_)
    walls = shapes if circle else 2 * shapes * shapes
    total = 0
    reported, valid = 0, False
    for _ in range(_SUBSTEPS):
        vel[:] = 0.0
        vel[0, 0] = _control(body[0, 2], body[0, 0], target[0], single)
        vel[0, 1] = _control(body[0, 3], body[0, 1], target[1], single)
        vel[1, 0], vel[1, 1] = body[1, 2], body[1, 3]
        for i in range(2):
            c, s = math.cos(body[i, 4]), math.sin(body[i, 4])
            gx, gy = c * cog[i, 0] - s * cog[i, 1], s * cog[i, 0] + c * cog[i, 1]
            cx = body[i, 0] + gx + (vel[i, 0] + bias[i, 0]) * _DT
            cy = body[i, 1] + gy + (vel[i, 1] + bias[i, 1]) * _DT
            angle = body[i, 4] + (body[i, 5] + bias[i, 2]) * _DT
            c, s = math.cos(angle), math.sin(angle)
            body[i, 0] = cx - (c * cog[i, 0] - s * cog[i, 1])
            body[i, 1] = cy - (s * cog[i, 0] + c * cog[i, 1])
            body[i, 4] = angle
            centers[i, 0], centers[i, 1] = cx, cy
            for k in range(broadphase.SHAPES):
                moving[broadphase.SHAPES * i + k, 0] = vel[i, 0]
                moving[broadphase.SHAPES * i + k, 1] = vel[i, 1]
            # Pymunk applies Space.damping to dynamic bodies after integrating positions.
            if inverse[i, 0] > 0:
                vel[i, 0], vel[i, 1], vel[i, 2] = 0.0, 0.0, 0.0
            else:
                vel[i, 2] = body[i, 5]
        _contacts(body, vertices, inverse, radius, enabled, circle, world, out, ids, pairs, tight)
        for i in range(count):
            a, b = pairs[i, 0], pairs[i, 1]
            nx, ny = out[i, 2], out[i, 3]
            ra[i, 0], ra[i, 1] = out[i, 0] - centers[a, 0], out[i, 1] - centers[a, 1]
            rb[i, 0], rb[i, 1] = out[i, 0] - centers[b, 0], out[i, 1] - centers[b, 1]
            ca[i] = ra[i, 0] * ny - ra[i, 1] * nx
            cb[i] = rb[i, 0] * ny - rb[i, 1] * nx
            m = inv[a, 0] + inv[b, 0] + ca[i] ** 2 * inv[a, 1] + cb[i] ** 2 * inv[b, 1]
            mass[i] = max(m, 1e-12)
            depth = out[i, 4]
            active[i] = depth >= 0
            goal[i] = _BIAS * max(depth - _SLOP, 0.0) / _DT
            total += active[i]
        _warm(cached, identifiers, ids, active, walls, circle, impulses, fresh)
        push[:] = 0.0
        # With one touching shape pair the order is moot. An unmoved tree reports what it reported last.
        first, need = -1, False
        for i in range(count):
            if active[i]:
                start = _group(i, walls, circle)[0]
                need = need or (first >= 0 and start != first)
                first = start
        moved = broadphase.refit(tree, bounds, tight, moving, order, refit)
        if moved or (need and not valid):
            reported = broadphase.report(tree, bounds, refit, calls, stack)
            valid = not moved
        # The solver visits the touching shape pairs in the order the broadphase reports them.
        live = _visit_order(calls, reported if need else 0, active, walls, circle, shapes, seen, sequence)
        for i in range(count):
            corrections[i] = 0.0
        for j in range(live):
            i = sequence[j]
            if fresh[i]:
                _apply(vel, pairs[i, 0], pairs[i, 1], impulses[i], out[i, 2], out[i, 3], ca[i], cb[i], inv)
        for _ in range(_ITERATIONS):
            for j in range(live):
                i = sequence[j]
                a, b = pairs[i, 0], pairs[i, 1]
                nx, ny = out[i, 2], out[i, 3]
                impulse = max(impulses[i] - _normal_speed(vel, a, b, ra[i], rb[i], nx, ny) / mass[i], 0.0)
                speed = _normal_speed(push, a, b, ra[i], rb[i], nx, ny)
                correction = max(corrections[i] + (goal[i] - speed) / mass[i], 0.0)
                _apply(vel, a, b, impulse - impulses[i], nx, ny, ca[i], cb[i], inv)
                _apply(push, a, b, correction - corrections[i], nx, ny, ca[i], cb[i], inv)
                impulses[i], corrections[i] = impulse, correction
        for i in range(2):
            body[i, 2], body[i, 3], body[i, 5] = vel[i, 0], vel[i, 1], vel[i, 2]
            bias[i, 0], bias[i, 1], bias[i, 2] = push[i, 0], push[i, 1], push[i, 2]
        _keep(cached, identifiers, ids, active, walls, circle, impulses)
    _contacts(body, vertices, inverse, radius, enabled, circle, world, out, ids, pairs, tight)
    lines[:] = 0.0
    row = 0
    for i in range(count):
        if out[i, 4] >= 0:
            depth = out[i, 4]
            lines[row, 0] = out[i, 0] + out[i, 2] * (depth - 2)
            lines[row, 1] = out[i, 1] + out[i, 3] * (depth - 2)
            lines[row, 2] = out[i, 0] + out[i, 2] * 2
            lines[row, 3] = out[i, 1] + out[i, 3] * 2
            lines[row, 4:] = _SEGMENT_COLOR
            row += 1
    return total, row


@njit(cache=True)
def warm_start(body, vertices, inverse, radius, enabled, circle, found, impulse, identifiers):
    """Set each environment's cached impulses and contact ids from the Pymunk contacts ``found`` after a reset.

    ``found`` rows are a contact point on body ``b`` and its normal impulse (see ``numeric.solver_rows``); a
    contact slot takes the impulse of the row at its point.
    """
    count = impulse.shape[1]
    world = np.empty((2, vertices.shape[2], 4, 2))
    tight = np.empty((broadphase.LEAVES, 4))
    out = np.empty((count, 5))
    ids = np.empty(count, np.int64)
    pairs = np.empty((count, 2), np.int64)
    for lane in range(body.shape[0]):
        _contacts(
            body[lane],
            vertices[lane],
            inverse[lane],
            radius[lane],
            enabled[lane],
            circle,
            world,
            out,
            ids,
            pairs,
            tight,
        )
        for i in range(count):
            impulse[lane, i] = 0.0
            identifiers[lane, i] = ids[i] if out[i, 4] >= 0 else -1
            if out[i, 4] >= 0:
                for j in range(found.shape[1]):
                    if abs(found[lane, j, 0] - out[i, 0]) <= 1e-7 and abs(found[lane, j, 1] - out[i, 1]) <= 1e-7:
                        impulse[lane, i] = found[lane, j, 2]


@njit(parallel=True, cache=True)
def visit_order(tree, bounds, tight, velocity, order, active, walls, circle, shapes, sequence, live):
    """Reindex each environment's broadphase and write its active contact slots in solver order.

    For a port that runs the rest of the substep itself (``jax_physics``): ``tight`` and ``velocity`` are the
    leaves' boxes and body velocities after the position update, ``active`` the contact slots that touch.
    """
    count = active.shape[1]
    for lane in prange(tree.shape[0]):
        fresh = np.empty(broadphase.LEAVES, np.bool_)
        calls = np.empty((broadphase.LEAVES * broadphase.PAIRS, 2), np.int64)
        stack = np.empty(2 * broadphase.NODES + 2, np.int64)
        seen = np.empty(count, np.bool_)
        slots = np.empty(count, np.int64)
        reported = broadphase.reindex(
            tree[lane], bounds[lane], tight[lane], velocity[lane], order[lane], fresh, calls, stack
        )
        live[lane] = _visit_order(calls, reported, active[lane], walls, circle, shapes, seen, slots)
        for j in range(count):
            sequence[lane, j] = slots[j] if j < live[lane] else 0


@njit(parallel=True, cache=True)
def advance_batch(
    body,
    bias,
    cached,
    identifiers,
    tree,
    bounds,
    order,
    vertices,
    cog,
    inverse,
    radius,
    enabled,
    target,
    single,
    circle,
    active,
    counts,
    lines,
):
    """Step every ``active`` environment in place.

    ``counts[:, 0]`` gets the contacts summed over substeps and ``counts[:, 1]`` the render segments
    written to ``lines``.
    """
    for lane in prange(body.shape[0]):
        if active[lane]:
            total, rows = _lane(
                body[lane],
                bias[lane],
                cached[lane],
                identifiers[lane],
                tree[lane],
                bounds[lane],
                order[lane],
                vertices[lane],
                cog[lane],
                inverse[lane],
                radius[lane],
                enabled[lane],
                target[lane],
                single,
                circle,
                lines[lane],
            )
            counts[lane, 0], counts[lane, 1] = total, rows
