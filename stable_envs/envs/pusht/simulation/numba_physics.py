"""Numba CPU port of the PushT contact solver in ``jax_physics``: the same math, in parallel over environments.

Ports the solver that ``jax_physics`` adapts from fracapuano/envX at d26c817.
Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

_DT = 0.01
_SUBSTEPS = 10
_ITERATIONS = 10
_BIAS_RATE = (1 - 0.9**0.6) / _DT
_SLOP = 0.1
_WALL_NORMALS = np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))
_WALL_PLANES = np.array((7.0, -504.0, 7.0, -504.0))
_SEGMENT_COLOR = np.array((231.0, 76.0, 60.0, 255.0))


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
    return face * 3 + (1 if t == 0 else 2 if t == 1 else 0)


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
def _contacts(body, vertices, inverse, radius, enabled, circle, world, out, ids, pairs):
    """Fill ``out`` rows (point x, y, normal x, y, depth), ``ids`` and ``pairs`` in ``jax_physics.contacts`` order."""
    shapes = vertices.shape[1]
    for index in range(2):
        c, s = math.cos(body[index, 4]), math.sin(body[index, 4])
        for shape in range(shapes):
            for k in range(4):
                vx, vy = vertices[index, shape, k, 0], vertices[index, shape, k, 1]
                world[index, shape, k, 0] = body[index, 0] + c * vx - s * vy
                world[index, shape, k, 1] = body[index, 1] + s * vx + c * vy
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
            for w in range(4):
                nx, ny = _WALL_NORMALS[w, 0], _WALL_NORMALS[w, 1]
                # The two deepest vertices, the lower index first on ties (``lax.top_k``).
                first, second = -1, -1
                for k in range(4):
                    p = _WALL_PLANES[w] - (nx * world[index, shape, k, 0] + ny * world[index, shape, k, 1])
                    if first < 0 or p > _WALL_PLANES[w] - (
                        nx * world[index, shape, first, 0] + ny * world[index, shape, first, 1]
                    ):
                        first, second = k, first
                    elif second < 0 or p > _WALL_PLANES[w] - (
                        nx * world[index, shape, second, 0] + ny * world[index, shape, second, 1]
                    ):
                        second = k
                for k in (first, second):
                    out[slot, 0], out[slot, 1] = world[index, shape, k, 0], world[index, shape, k, 1]
                    out[slot, 2], out[slot, 3] = nx, ny
                    depth = _WALL_PLANES[w] - (nx * out[slot, 0] + ny * out[slot, 1])
                    out[slot, 4] = depth if live else -1.0
                    ids[slot] = k
                    pairs[slot, 0], pairs[slot, 1] = 2, index
                    slot += 1
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


@njit(cache=True)
def _lane(body, bias, cached, identifiers, vertices, cog, inverse, radius, enabled, target, circle, lines):
    count = cached.shape[0]
    shapes = vertices.shape[1]
    world = np.empty((2, shapes, 4, 2))
    out = np.empty((count, 5))
    ids = np.empty(count, np.int64)
    pairs = np.empty((count, 2), np.int64)
    inv = np.zeros((3, 2))
    inv[:2] = inverse
    centers = np.zeros((3, 2))
    vel = np.zeros((3, 3))
    push = np.zeros((3, 3))
    ra = np.empty((count, 2))
    rb = np.empty((count, 2))
    ca = np.empty(count)
    cb = np.empty(count)
    mass = np.empty(count)
    goal = np.empty(count)
    active = np.empty(count, np.bool_)
    order = np.empty(count, np.int64)
    impulses = np.empty(count)
    corrections = np.empty(count)
    total = 0
    for _ in range(_SUBSTEPS):
        vel[:] = 0.0
        vel[0, 0] = body[0, 2] + (100 * (target[0] - body[0, 0]) - 20 * body[0, 2]) * _DT
        vel[0, 1] = body[0, 3] + (100 * (target[1] - body[0, 1]) - 20 * body[0, 3]) * _DT
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
            # Pymunk applies Space.damping to dynamic bodies after integrating positions.
            if inverse[i, 0] > 0:
                vel[i, 0], vel[i, 1], vel[i, 2] = 0.0, 0.0, 0.0
            else:
                vel[i, 2] = body[i, 5]
        _contacts(body, vertices, inverse, radius, enabled, circle, world, out, ids, pairs)
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
            goal[i] = _BIAS_RATE * max(depth - _SLOP, 0.0)
            if not (active[i] and ids[i] == identifiers[i]):
                cached[i] = 0.0
            total += active[i]
        push[:] = 0.0
        # An inactive contact keeps zero impulses, so the solver visits the active ones only.
        live = 0
        for i in range(count):
            impulses[i] = cached[i]
            corrections[i] = 0.0
            if active[i]:
                order[live] = i
                live += 1
        for j in range(live):
            i = order[j]
            _apply(vel, pairs[i, 0], pairs[i, 1], cached[i], out[i, 2], out[i, 3], ca[i], cb[i], inv)
        for _ in range(_ITERATIONS):
            for j in range(live):
                i = order[j]
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
        for i in range(count):
            cached[i] = impulses[i]
            identifiers[i] = ids[i]
    _contacts(body, vertices, inverse, radius, enabled, circle, world, out, ids, pairs)
    lines[:] = 0.0
    row = 0
    for i in range(count):
        if out[i, 4] >= 0:
            depth = out[i, 4]
            lines[row, 0] = out[i, 0] - out[i, 2] * (depth + 2)
            lines[row, 1] = out[i, 1] - out[i, 3] * (depth + 2)
            lines[row, 2] = out[i, 0] + out[i, 2] * 2
            lines[row, 3] = out[i, 1] + out[i, 3] * 2
            lines[row, 4:] = _SEGMENT_COLOR
            row += 1
    return total, row


@njit(parallel=True, cache=True)
def advance_batch(
    body, bias, cached, identifiers, vertices, cog, inverse, radius, enabled, target, circle, active, counts, lines
):
    """Step every ``active`` environment in place, as ``jax_physics.advance`` does.

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
                vertices[lane],
                cog[lane],
                inverse[lane],
                radius[lane],
                enabled[lane],
                target[lane],
                circle,
                lines[lane],
            )
            counts[lane, 0], counts[lane, 1] = total, rows
