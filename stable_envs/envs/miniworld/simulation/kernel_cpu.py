"""Numba CPU step and reset of MovingBlockWorld environments, one environment per thread.

The step follows ``MiniWorldEnv.step`` and ``MovingBlockWorld.step`` operation by operation in float64, so it
gives the source's values. The reset samples the distribution of ``MovingBlockWorld.reset`` from the
counter-based generator of ``rng.py``; ``kernel_torch.py`` draws the same numbers.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from . import rng as R

# Indices of the float parameter vector ``P`` (see ``world.py``).
from .world import (
    P_AGENT_CENTER,
    P_AGENT_OVERLAP,
    P_BALL_RADIUS,
    P_BLOCK_H,
    P_BLOCK_XY,
    P_BOX_AND_BALL,
    P_BOX_OVERLAP,
    P_BOX_TEX,
    P_BOX_TEX_RANDOM,
    P_CARDINAL,
    P_CEIL_TEX,
    P_ENSURE_BASE,
    P_FLOOR_TEX,
    P_FLOOR_TEX_RANDOM,
    P_FWD,
    P_GRID,
    P_HEADING_ZERO,
    P_MAX_FWD,
    P_MAX_STEPS,
    P_RANDOM_ORIENT,
    P_REPEAT,
    P_ROOM,
    P_SPAWN_BUFFER,
    P_SPEED_SCALE,
    P_STATIC,
    P_TORUS,
    P_TURN,
    P_VMAX,
    P_VMIN,
    P_WALL_TEX,
    P_WALL_TEX_RANDOM,
    P_WRAP_HEADING,
)

AGENT_RADIUS = 0.4
CAM_HEIGHT = 1.5
#: Rejection-sampling candidates per placement; the last one is kept when none fits.
CANDIDATES = 16
GRID_TRIES = 8


@njit(cache=True, inline="always")
def _wall_hit(x, z, r, S):
    """``intersect_circle_segs`` against the four walls of the ``S`` x ``S`` room, in the source's order."""
    for k in range(4):
        # Room outline (S, S), (S, 0), (0, 0), (0, S); each segment runs from the next corner to this one.
        if k == 0:
            ax, az, bx, bz = S, 0.0, S, S
        elif k == 1:
            ax, az, bx, bz = 0.0, 0.0, S, 0.0
        elif k == 2:
            ax, az, bx, bz = 0.0, S, 0.0, 0.0
        else:
            ax, az, bx, bz = S, S, 0.0, S
        abx, abz = bx - ax, bz - az
        apx, apz = x - ax, z - az
        proj = (apx * abx + apz * abz) / (abx * abx + abz * abz)
        proj = min(max(proj, 0.0), 1.0)
        dx = ax + proj * abx - x
        dz = az + proj * abz - z
        if math.sqrt(dx * dx + dz * dz) < r:
            return True
    return False


@njit(cache=True, inline="always")
def _near(x, z, r, ox, oz, orad):
    dx, dz = ox - x, oz - z
    return math.sqrt(dx * dx + dz * dz) < r + orad


@njit(cache=True)
def _hit(i, who, x, z, r, walls, apos, bpos, rad, count, P):
    """``MovingBlockWorld.intersect`` of entity ``who`` (-1: the agent, else a block) at ``(x, z)``.

    Returns -2 for a wall, -1 for the agent, the index of the first block hit, or -3 for nothing.
    """
    if walls and _wall_hit(x, z, r, P[P_ROOM]):
        return -2
    if who >= 0 and not (P[P_AGENT_OVERLAP] > 0) and _near(x, z, r, apos[i, 0], apos[i, 2], AGENT_RADIUS):
        return -1
    if who >= 0 and P[P_BOX_OVERLAP] > 0:
        return -3
    if who < 0 and P[P_AGENT_OVERLAP] > 0:
        return -3
    for m in range(count[i]):
        if m != who and _near(x, z, r, bpos[i, m, 0], bpos[i, m, 2], rad[i, m]):
            return m
    return -3


@njit(cache=True)
def _carry_pos(i, c, ax, az, d, rad, height, P):
    dist = AGENT_RADIUS + rad[i, c] + P[P_MAX_FWD]
    x = ax + (math.cos(d) * 1.05) * dist
    z = az + (-math.sin(d) * 1.05) * dist
    y = max(CAM_HEIGHT - height[i, c] - 0.3, 0.0)
    return x, y, z


@njit(cache=True)
def _snap(i, who, pos3, walls, apos, bpos, rad, count, P):
    """``_snap_entity_to_grid``: round x and z, else the first free neighbour, else stay."""
    r = AGENT_RADIUS if who < 0 else rad[i, who]
    sx, sz = float(np.round(pos3[0])), float(np.round(pos3[2]))
    if _hit(i, who, sx, sz, r, walls, apos, bpos, rad, count, P) == -3:
        pos3[0], pos3[2] = sx, sz
        return
    for dx in range(-1, 2):
        for dz in range(-1, 2):
            if _hit(i, who, sx + dx, sz + dz, r, walls, apos, bpos, rad, count, P) == -3:
                pos3[0], pos3[2] = sx + dx, sz + dz
                return


@njit(cache=True)
def _wrap_across(v, r, S):
    if v + r > S:
        return max(v - S, r)
    if v - r < 0.0:
        return min(v + S, S - r)
    return v


@njit(cache=True)
def _move_block(i, m, apos, bpos, vel, rad, count, P):
    S = P[P_ROOM]
    r = rad[i, m]
    torus = P[P_TORUS] > 0
    for axis in (0, 2):
        v = vel[i, m, axis]
        if v != 0.0:
            cur = bpos[i, m, axis]
            cand = cur + v
            if torus and (cand + r > S or cand - r < 0.0):
                bpos[i, m, axis] = _wrap_across(cand, r, S)
            else:
                cx = cand if axis == 0 else bpos[i, m, 0]
                cz = cand if axis == 2 else bpos[i, m, 2]
                if _hit(i, m, cx, cz, r, not torus, apos, bpos, rad, count, P) == -3:
                    bpos[i, m, axis] = cand
                else:
                    vel[i, m, axis] = -v
                    cand = cur - v
                    cx = cand if axis == 0 else bpos[i, m, 0]
                    cz = cand if axis == 2 else bpos[i, m, 2]
                    if _hit(i, m, cx, cz, r, not torus, apos, bpos, rad, count, P) == -3:
                        bpos[i, m, axis] = cand
        if P[P_GRID] > 0:
            snapped = float(np.round(bpos[i, m, axis]))
            if torus:
                snapped = min(max(snapped, 1.0), S - 1.0)
            bpos[i, m, axis] = snapped


@njit(cache=True)
def _step_env(i, a, apos, adir, carry, count, rad, height, bpos, bdir, vel, steps, truncated, P):
    steps[i] += 1
    d = adir[i]
    c = carry[i]
    if a == 2 or a == 3:
        f = P[P_FWD] if a == 2 else -P[P_FWD]
        nx = apos[i, 0] + math.cos(d) * f + math.sin(d) * 0.0
        nz = apos[i, 2] + (-math.sin(d)) * f + math.cos(d) * 0.0
        if _hit(i, -1, nx, nz, AGENT_RADIUS, True, apos, bpos, rad, count, P) == -3:
            moved = True
            if c >= 0:
                x, y, z = _carry_pos(i, c, nx, nz, d, rad, height, P)
                if _hit(i, c, x, z, rad[i, c], True, apos, bpos, rad, count, P) == -3:
                    bpos[i, c, 0], bpos[i, c, 1], bpos[i, c, 2] = x, apos[i, 1] + y, z
                else:
                    moved = False
            if moved:
                apos[i, 0], apos[i, 2] = nx, nz
    elif a == 0 or a == 1:
        turn = P[P_TURN] if a == 0 else -P[P_TURN]
        nd = d + turn * (math.pi / 180)
        adir[i] = nd
        if c >= 0:
            x, y, z = _carry_pos(i, c, apos[i, 0], apos[i, 2], nd, rad, height, P)
            if _hit(i, c, x, z, rad[i, c], True, apos, bpos, rad, count, P) != -3:
                adir[i] = d
            else:
                bpos[i, c, 0], bpos[i, c, 1], bpos[i, c, 2] = x, apos[i, 1] + y, z
                bdir[i, c] = nd
    elif a == 5:
        tx = apos[i, 0] + (math.cos(d) * 1.5) * AGENT_RADIUS
        tz = apos[i, 2] + (-math.sin(d) * 1.5) * AGENT_RADIUS
        hit = _hit(i, -1, tx, tz, 1.2 * AGENT_RADIUS, True, apos, bpos, rad, count, P)
        if c < 0 and hit >= 0:
            carry[i] = hit
    elif a == 6:
        if c >= 0:
            bpos[i, c, 1] = 0.0
            carry[i] = -1
    c = carry[i]
    if c >= 0:
        x, y, z = _carry_pos(i, c, apos[i, 0], apos[i, 2], adir[i], rad, height, P)
        bpos[i, c, 0], bpos[i, c, 1], bpos[i, c, 2] = x, apos[i, 1] + y, z
        bdir[i, c] = adir[i]
    if steps[i] >= P[P_MAX_STEPS]:
        truncated[i] = True
    else:
        if P[P_GRID] > 0:
            _snap(i, -1, apos[i], True, apos, bpos, rad, count, P)
            q = math.pi / 2
            adir[i] = float(np.round(adir[i] / q)) * q
            if c >= 0:
                bdir[i, c] = adir[i]
                _snap(i, c, bpos[i, c], True, apos, bpos, rad, count, P)
        if not (P[P_STATIC] > 0):
            for m in range(count[i]):
                if m != c:
                    _move_block(i, m, apos, bpos, vel, rad, count, P)
    if P[P_WRAP_HEADING] > 0:
        adir[i] = adir[i] % (2.0 * math.pi)


@njit(cache=True, parallel=True)
def step_parallel(actions, active, apos, adir, carry, count, rad, height, bpos, bdir, vel, steps, truncated, P):
    for i in prange(actions.shape[0]):
        truncated[i] = False
        if active[i]:
            _step_env(i, actions[i], apos, adir, carry, count, rad, height, bpos, bdir, vel, steps, truncated, P)


@njit(cache=True)
def step_serial(actions, active, apos, adir, carry, count, rad, height, bpos, bdir, vel, steps, truncated, P):
    for i in range(actions.shape[0]):
        truncated[i] = False
        if active[i]:
            _step_env(i, actions[i], apos, adir, carry, count, rad, height, bpos, bdir, vel, steps, truncated, P)


@njit(cache=True)
def _extents(r, P):
    """``_spawn_extents``: the buffered placement box, or the whole room (flag False)."""
    S = P[P_ROOM]
    if P[P_SPAWN_BUFFER] < 0:
        return False, 0.0, S, 0.0, S
    buf = P[P_SPAWN_BUFFER] + r
    lo, hi = buf, S + buf - 1.0
    if hi <= lo:
        return False, 0.0, S, 0.0, S
    return True, lo, hi, lo, hi


@njit(cache=True)
def _place(i, who, r, base, slot, apos, bpos, rad, count, P):
    """``place_entity`` by rejection sampling over ``CANDIDATES`` draws; returns (x, z)."""
    S = P[P_ROOM]
    _, lx, hx, lz, hz = _extents(r, P)
    x = z = 0.0
    for k in range(CANDIDATES):
        x = (lx - r) + ((hx + r) - (lx - r)) * R.uniform(base, slot + 2 * k)
        z = (lz - r) + ((hz + r) - (lz - r)) * R.uniform(base, slot + 2 * k + 1)
        if 0.0 < x < S and 0.0 < z < S and _hit(i, who, x, z, r, True, apos, bpos, rad, count, P) == -3:
            return x, z
    return min(max(x, r), S - r), min(max(z, r), S - r)


@njit(cache=True)
def _reset_env(
    i,
    seed,
    episode,
    room,
    kind,
    color,
    texture,
    size,
    apos,
    adir,
    carry,
    count,
    rad,
    height,
    bpos,
    bdir,
    vel,
    steps,
    P,
    lo,
    hi,
    pool,
    base_palette,
    floor_pool,
    wall_pool,
    box_pool,
):
    base = R.stream(seed[i], episode[i])
    M = kind.shape[1]
    S = P[P_ROOM]
    n = lo + int(R.uniform(base, R.COUNT) * (hi - lo + 1))
    count[i] = n
    room[i, 0] = floor_pool[int(R.uniform(base, R.FLOOR) * 3)] if P[P_FLOOR_TEX_RANDOM] > 0 else int(P[P_FLOOR_TEX])
    room[i, 1] = wall_pool[int(R.uniform(base, R.WALL) * 3)] if P[P_WALL_TEX_RANDOM] > 0 else int(P[P_WALL_TEX])
    room[i, 2] = int(P[P_CEIL_TEX])
    # colours
    npool = pool.shape[0]
    if P[P_ENSURE_BASE] > 0:
        nb = base_palette.shape[0]
        keys = np.empty(nb)
        for p in range(nb):
            keys[p] = R.uniform(base, R.PERM + p)
        order = np.argsort(keys)
        for j in range(n):
            if j < nb:
                color[i, j] = base_palette[order[j]]
            else:
                color[i, j] = pool[int(R.uniform(base, R.COLOR + j) * npool)]
    elif not (P[P_REPEAT] > 0) and n <= npool:
        keys = np.empty(npool)
        for p in range(npool):
            keys[p] = R.uniform(base, R.PERM + p)
        order = np.argsort(keys)
        for j in range(n):
            color[i, j] = pool[order[j]]
    else:
        for j in range(n):
            color[i, j] = pool[int(R.uniform(base, R.COLOR + j) * npool)]
    for m in range(n, M):
        kind[i, m], color[i, m], texture[i, m] = 0, 0, -1
        size[i, m, 0] = size[i, m, 1] = size[i, m, 2] = 1.0
        rad[i, m] = math.sqrt(2.0) / 2
        height[i, m] = 1.0
        bpos[i, m, 0] = bpos[i, m, 1] = bpos[i, m, 2] = 0.0
        vel[i, m, 0] = vel[i, m, 1] = vel[i, m, 2] = 0.0
        bdir[i, m] = 0.0
    # the agent, placed before the blocks (no block counts yet)
    count[i] = 0
    apos[i, 1] = 0.0
    if P[P_AGENT_CENTER] > 0:
        ok, mnx, mxx, mnz, mxz = _extents(AGENT_RADIUS, P)
        if not ok:
            mnx, mxx, mnz, mxz = 0.0, S, 0.0, S
        cx, cz = (mnx + mxx) / 2.0, (mnz + mxz) / 2.0
        if P[P_GRID] > 0:
            cx = np.round(cx) - 1.0 if abs(cx - np.round(cx)) < 1e-6 else np.floor(cx)
            cz = np.round(cz) - 1.0 if abs(cz - np.round(cz)) < 1e-6 else np.floor(cz)
        else:
            cx = cx - 0.5 if abs(cx - np.round(cx)) < 1e-6 else cx
            cz = cz - 0.5 if abs(cz - np.round(cz)) < 1e-6 else cz
        apos[i, 0], apos[i, 2] = min(max(cx, mnx), mxx), min(max(cz, mnz), mxz)
    else:
        apos[i, 0], apos[i, 2] = _place(i, -1, AGENT_RADIUS, base, R.AGENT, apos, bpos, rad, count, P)
    adir[i] = -math.pi + 2 * math.pi * R.uniform(base, R.AGENT_DIR)
    # the blocks, each checked against the agent and the blocks placed before it
    for m in range(n):
        b = R.BLOCK + R.BLOCK_STRIDE * m
        texture[i, m] = (
            box_pool[int(R.uniform(base, b + R.B_TEX) * 3)] if P[P_BOX_TEX_RANDOM] > 0 else int(P[P_BOX_TEX])
        )
        if P[P_BOX_AND_BALL] > 0 and int(R.uniform(base, b + R.B_BALL) * 3) == 1:
            kind[i, m], texture[i, m] = 1, -1
            size[i, m, 0] = size[i, m, 1] = size[i, m, 2] = 0.85
            rad[i, m], height[i, m] = P[P_BALL_RADIUS] * 0.85, 0.85
        else:
            kind[i, m] = 0
            if P[P_BLOCK_XY] >= 0 or P[P_BLOCK_H] >= 0:
                sx = P[P_BLOCK_XY] if P[P_BLOCK_XY] >= 0 else 0.75
                sy = P[P_BLOCK_H] if P[P_BLOCK_H] >= 0 else sx
            else:
                sx = sy = 0.6 + 0.25 * R.uniform(base, b + R.B_SIZE)
            size[i, m, 0], size[i, m, 1], size[i, m, 2] = sx, sy, sx
            rad[i, m], height[i, m] = math.sqrt(sx * sx + sx * sx) / 2, sy
        x, z = _place(i, m, rad[i, m], base, b + R.B_PLACE, apos, bpos, rad, count, P)
        bpos[i, m, 0], bpos[i, m, 1], bpos[i, m, 2] = x, 0.0, z
        bdir[i, m] = -math.pi + 2 * math.pi * R.uniform(base, b + R.B_DIR) if P[P_RANDOM_ORIENT] > 0 else 0.0
        count[i] = m + 1
    if P[P_GRID] > 0:
        _snap(i, -1, apos[i], True, apos, bpos, rad, count, P)
        for m in range(n):
            _snap(i, m, bpos[i, m], True, apos, bpos, rad, count, P)
    for m in range(n):
        vx = vz = 0.0
        if P[P_STATIC] > 0:
            pass
        elif P[P_GRID] > 0:
            if P[P_CARDINAL] > 0:
                speed = max(1.0, max(abs(P[P_VMIN]), abs(P[P_VMAX])))
                choice = int(R.uniform(base, R.BLOCK + R.BLOCK_STRIDE * m + R.B_VEL) * 4)
                if choice == 0:
                    vx = speed
                elif choice == 1:
                    vx = -speed
                elif choice == 2:
                    vz = speed
                else:
                    vz = -speed
            else:
                span = P[P_VMAX] - P[P_VMIN] + 1.0
                vx = P[P_VMAX]
                for t in range(GRID_TRIES):
                    b = R.BLOCK + R.BLOCK_STRIDE * m + R.B_GRID + 2 * t
                    tx = P[P_VMIN] + math.floor(R.uniform(base, b) * span)
                    tz = P[P_VMIN] + math.floor(R.uniform(base, b + 1) * span)
                    if tx != 0.0 or tz != 0.0:
                        vx, vz = tx, tz
                        break
        else:
            theta = -math.pi + 2 * math.pi * R.uniform(base, R.BLOCK + R.BLOCK_STRIDE * m + R.B_THETA)
            step = P[P_FWD] * P[P_SPEED_SCALE]
            vx, vz = step * math.cos(theta), step * math.sin(theta)
        vel[i, m, 0], vel[i, m, 1], vel[i, m, 2] = vx, 0.0, vz
    if P[P_HEADING_ZERO] > 0:
        adir[i] = 0.0
    if P[P_WRAP_HEADING] > 0:
        adir[i] = adir[i] % (2.0 * math.pi)
    carry[i] = -1
    steps[i] = 0


@njit(cache=True, parallel=True)
def reset_all(
    seed,
    episode,
    room,
    kind,
    color,
    texture,
    size,
    apos,
    adir,
    carry,
    count,
    rad,
    height,
    bpos,
    bdir,
    vel,
    steps,
    P,
    lo,
    hi,
    pool,
    base_palette,
    floor_pool,
    wall_pool,
    box_pool,
):
    for i in prange(seed.shape[0]):
        _reset_env(
            i,
            seed,
            episode,
            room,
            kind,
            color,
            texture,
            size,
            apos,
            adir,
            carry,
            count,
            rad,
            height,
            bpos,
            bdir,
            vel,
            steps,
            P,
            lo,
            hi,
            pool,
            base_palette,
            floor_pool,
            wall_pool,
            box_pool,
        )
