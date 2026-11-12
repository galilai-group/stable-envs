"""Triton CUDA step of MovingBlockWorld environments: one thread per environment, in place, in one launch.

The per-environment rules are those of ``kernel_cpu.py``, in float64 without fused multiply-adds; float
constants come from a float64 vector so that none is rounded to float32. Distances are compared squared, as
float64 division and square root are slow on most GPUs, so a distance equal to its bound to the last bit can
compare differently.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .kernel_cpu import AGENT_RADIUS, CAM_HEIGHT
from .world import P_FWD, P_MAX_FWD, P_MAX_STEPS, P_ROOM, P_TURN

BLOCK = 32
#: Indices of the constant vector ``C``.
C_ROOM = tl.constexpr(0)
C_FWD = tl.constexpr(1)
C_TURN = tl.constexpr(2)
C_MAX_FWD = tl.constexpr(3)
C_MAX_STEPS = tl.constexpr(4)
C_AGENT_R = tl.constexpr(5)
C_105 = tl.constexpr(6)
C_03 = tl.constexpr(7)
C_CAM = tl.constexpr(8)
C_15 = tl.constexpr(9)
C_12 = tl.constexpr(10)
C_DEG = tl.constexpr(11)
C_QUARTER = tl.constexpr(12)
C_TURN_FULL = tl.constexpr(13)
C_ZERO = tl.constexpr(14)
C_ONE = tl.constexpr(15)


def constants(P, device) -> torch.Tensor:
    values = [P[P_ROOM], P[P_FWD], P[P_TURN], P[P_MAX_FWD], P[P_MAX_STEPS]]
    values += [AGENT_RADIUS, 1.05, 0.3, CAM_HEIGHT, 1.5, 1.2, math.pi / 180, math.pi / 2, 2.0 * math.pi, 0.0, 1.0]
    return torch.tensor(values, dtype=torch.float64, device=device)


@triton.jit
def _wall(x, z, r, S, zero):
    """Whether a circle touches a wall: its squared distance to each wall segment against ``r**2``."""
    r2 = r * r
    dx = tl.minimum(tl.maximum(x, zero), S) - x
    dz = tl.minimum(tl.maximum(z, zero), S) - z
    e = S - x
    n = S - z
    return (e * e + dz * dz < r2) | (x * x + dz * dz < r2) | (dx * dx + z * z < r2) | (dx * dx + n * n < r2)


@triton.jit
def _near(x, z, r, ox, oz, orad):
    dx = ox - x
    dz = oz - z
    bound = r + orad
    return dx * dx + dz * dz < bound * bound


@triton.jit
def _blocked(
    x, z, r, who, offs, mask, count, apos, bpos, rad, S, AR, zero,
    WALLS: tl.constexpr, AGENT_OV: tl.constexpr, BOX_OV: tl.constexpr, M: tl.constexpr,
):  # fmt: skip
    """``MovingBlockWorld.intersect`` of entity ``who`` (-1: the agent) at ``(x, z)``, as a hit flag."""
    hit = x != x
    if WALLS:
        hit = hit | _wall(x, z, r, S, zero)
    if not AGENT_OV:
        ax = tl.load(apos + offs * 3, mask=mask, other=0.0)
        az = tl.load(apos + offs * 3 + 2, mask=mask, other=0.0)
        hit = hit | ((who >= 0) & _near(x, z, r, ax, az, AR))
    if not (AGENT_OV and BOX_OV):
        for j in range(M):
            sel = (j < count) & (j != who)
            if AGENT_OV:
                sel = sel & (who >= 0)
            if BOX_OV:
                sel = sel & (who < 0)
            bx = tl.load(bpos + (offs * M + j) * 3, mask=mask, other=0.0)
            bz = tl.load(bpos + (offs * M + j) * 3 + 2, mask=mask, other=0.0)
            br = tl.load(rad + offs * M + j, mask=mask, other=0.0)
            hit = hit | (sel & _near(x, z, r, bx, bz, br))
    return hit


@triton.jit
def _snap(
    x, z, r, who, offs, mask, count, apos, bpos, rad, S, AR, zero,
    AGENT_OV: tl.constexpr, BOX_OV: tl.constexpr, M: tl.constexpr,
):  # fmt: skip
    """``_snap_entity_to_grid``: the rounded point, else its first free neighbour, else stay."""
    sx = libdevice.rint(x)
    sz = libdevice.rint(z)
    free = ~_blocked(sx, sz, r, who, offs, mask, count, apos, bpos, rad, S, AR, zero, True, AGENT_OV, BOX_OV, M)
    nx = tl.where(free, sx, x)
    nz = tl.where(free, sz, z)
    done = free
    for k in range(9):
        cx = sx + (k // 3 - 1).to(tl.float64)
        cz = sz + (k % 3 - 1).to(tl.float64)
        ok = ~done & ~_blocked(
            cx, cz, r, who, offs, mask, count, apos, bpos, rad, S, AR, zero, True, AGENT_OV, BOX_OV, M
        )
        nx = tl.where(ok, cx, nx)
        nz = tl.where(ok, cz, nz)
        done = done | ok
    return nx, nz


@triton.jit
def _carry_pos(ax, az, cosd, sind, rc, hc, C):
    dist = tl.load(C + C_AGENT_R) + rc + tl.load(C + C_MAX_FWD)
    k = tl.load(C + C_105)
    x = ax + (cosd * k) * dist
    z = az + (-sind * k) * dist
    y = tl.maximum(tl.load(C + C_CAM) - hc - tl.load(C + C_03), tl.load(C + C_ZERO))
    return x, y, z


@triton.jit
def _free(x, z, r, ax, az, S, AR, zero, WALLS: tl.constexpr, AGENT_OV: tl.constexpr):
    """No wall (when ``WALLS``) and, without agent overlap, not the agent: a block's free test without other blocks."""
    hit = x != x
    if WALLS:
        hit = hit | _wall(x, z, r, S, zero)
    if not AGENT_OV:
        hit = hit | _near(x, z, r, ax, az, AR)
    return ~hit


@triton.jit
def _axis(cur, v, cand, back, free, free2, r, mv, S, zero, TORUS: tl.constexpr, GRID: tl.constexpr):
    """One axis of a block's move: on, else bounce back, else stay; torus wrap; grid rounding."""
    new = tl.where(free, cand, tl.where(free2, back, cur))
    nv = tl.where(free, v, -v)
    if TORUS:
        over = cand + r > S
        wrap = over | (cand - r < zero)
        wrapped = tl.where(over, tl.maximum(cand - S, r), tl.minimum(cand + S, S - r))
        new = tl.where(wrap, wrapped, new)
        nv = tl.where(wrap, v, nv)
    moving = mv & (v != zero)
    new = tl.where(moving, new, cur)
    if GRID:
        rounded = libdevice.rint(new)
        if TORUS:
            rounded = tl.minimum(tl.maximum(rounded, zero + 1.0), S - 1.0)
        new = tl.where(mv, rounded, new)
    return new, nv, moving


@triton.jit
def step_kernel(
    actions, active, apos, adir, carry, count_ptr, rad, height, bpos, bdir, vel, steps_ptr, truncated, C, B,
    M: tl.constexpr, BLOCK: tl.constexpr, ALL: tl.constexpr, GRID: tl.constexpr, TORUS: tl.constexpr,
    STATIC: tl.constexpr, AGENT_OV: tl.constexpr, BOX_OV: tl.constexpr, WRAP: tl.constexpr,
):  # fmt: skip
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < B
    if ALL:
        live = mask
    else:
        live = mask & (tl.load(active + offs, mask=mask, other=0) != 0)
    S = tl.load(C + C_ROOM)
    AR = tl.load(C + C_AGENT_R)
    zero = tl.load(C + C_ZERO)
    a = tl.load(actions + offs, mask=mask, other=4)
    count = tl.load(count_ptr + offs, mask=mask, other=0)
    steps = tl.load(steps_ptr + offs, mask=mask, other=0) + 1
    d = tl.load(adir + offs, mask=mask, other=0.0)
    c = tl.load(carry + offs, mask=mask, other=-1)
    ax = tl.load(apos + offs * 3, mask=mask, other=0.0)
    ay = tl.load(apos + offs * 3 + 1, mask=mask, other=0.0)
    az = tl.load(apos + offs * 3 + 2, mask=mask, other=0.0)
    has = c >= 0
    ci = tl.maximum(c, 0)
    rc = tl.load(rad + offs * M + ci, mask=mask, other=1.0)
    hc = tl.load(height + offs * M + ci, mask=mask, other=1.0)
    cptr = bpos + (offs * M + ci) * 3
    # move forward or back
    fwd = tl.load(C + C_FWD)
    f = tl.where(a == 2, fwd, -fwd)
    cosd = libdevice.cos(d)
    sind = libdevice.sin(d)
    nx = ax + cosd * f + sind * zero
    nz = az + (-sind) * f + cosd * zero
    free = ~_blocked(
        nx, nz, zero + AR, ci * 0 - 1, offs, mask, count, apos, bpos, rad, S, AR, zero, True, AGENT_OV, BOX_OV, M
    )
    cx, cy, cz = _carry_pos(nx, nz, cosd, sind, rc, hc, C)
    cfree = ~_blocked(cx, cz, rc, ci, offs, mask, count, apos, bpos, rad, S, AR, zero, True, AGENT_OV, BOX_OV, M)
    moved = live & ((a == 2) | (a == 3)) & free & (~has | cfree)
    tl.store(cptr, cx, mask=moved & has)
    tl.store(cptr + 1, ay + cy, mask=moved & has)
    tl.store(cptr + 2, cz, mask=moved & has)
    ax = tl.where(moved, nx, ax)
    az = tl.where(moved, nz, az)
    tl.store(apos + offs * 3, ax, mask=moved)
    tl.store(apos + offs * 3 + 2, az, mask=moved)
    # turn
    turn = tl.load(C + C_TURN)
    nd = d + tl.where(a == 0, turn, -turn) * tl.load(C + C_DEG)
    cosn = libdevice.cos(nd)
    sinn = libdevice.sin(nd)
    tx, ty, tz = _carry_pos(ax, az, cosn, sinn, rc, hc, C)
    tfree = ~_blocked(tx, tz, rc, ci, offs, mask, count, apos, bpos, rad, S, AR, zero, True, AGENT_OV, BOX_OV, M)
    turned = live & ((a == 0) | (a == 1)) & (~has | tfree)
    tl.store(cptr, tx, mask=turned & has)
    tl.store(cptr + 1, ay + ty, mask=turned & has)
    tl.store(cptr + 2, tz, mask=turned & has)
    tl.store(bdir + offs * M + ci, nd, mask=turned & has)
    d = tl.where(turned, nd, d)
    cosd = tl.where(turned, cosn, cosd)
    sind = tl.where(turned, sinn, sind)
    # pick up the first block hit, unless a wall is hit
    k15 = tl.load(C + C_15)
    px = ax + (cosd * k15) * AR
    pz = az + (-sind * k15) * AR
    pr = tl.load(C + C_12) * AR
    first = ci * 0 - 1
    if not AGENT_OV:
        for j in range(M):
            bx = tl.load(bpos + (offs * M + j) * 3, mask=mask, other=0.0)
            bz = tl.load(bpos + (offs * M + j) * 3 + 2, mask=mask, other=0.0)
            br = tl.load(rad + offs * M + j, mask=mask, other=0.0)
            hit = (j < count) & _near(px, pz, pr, bx, bz, br)
            first = tl.where(hit & (first < 0), j, first)
    pick = live & (a == 5) & ~has & ~_wall(px, pz, pr, S, zero) & (first >= 0)
    drop = live & (a == 6) & has
    tl.store(cptr + 1, zero, mask=drop)
    c = tl.where(pick, first, tl.where(drop, -1, c))
    # the carried block follows the agent
    has = c >= 0
    ci = tl.maximum(c, 0)
    rc = tl.load(rad + offs * M + ci, mask=mask, other=1.0)
    hc = tl.load(height + offs * M + ci, mask=mask, other=1.0)
    cptr = bpos + (offs * M + ci) * 3
    x, y, z = _carry_pos(ax, az, cosd, sind, rc, hc, C)
    follow = live & has
    tl.store(cptr, x, mask=follow)
    tl.store(cptr + 1, ay + y, mask=follow)
    tl.store(cptr + 2, z, mask=follow)
    tl.store(bdir + offs * M + ci, d, mask=follow)
    trunc = steps.to(tl.float64) >= tl.load(C + C_MAX_STEPS)
    go = live & ~trunc
    if GRID:
        sx, sz = _snap(
            ax, az, zero + AR, ci * 0 - 1, offs, mask, count, apos, bpos, rad, S, AR, zero, AGENT_OV, BOX_OV, M
        )
        ax = tl.where(go, sx, ax)
        az = tl.where(go, sz, az)
        tl.store(apos + offs * 3, ax, mask=go)
        tl.store(apos + offs * 3 + 2, az, mask=go)
        q = tl.load(C + C_QUARTER)
        d = tl.where(go, libdevice.rint(d / q) * q, d)
        tl.store(bdir + offs * M + ci, d, mask=go & has)
        ox = tl.load(cptr, mask=mask & has, other=0.0)
        oz = tl.load(cptr + 2, mask=mask & has, other=0.0)
        sx, sz = _snap(ox, oz, rc, ci, offs, mask, count, apos, bpos, rad, S, AR, zero, AGENT_OV, BOX_OV, M)
        tl.store(cptr, sx, mask=go & has)
        tl.store(cptr + 2, sz, mask=go & has)
    if not STATIC:
        if BOX_OV:
            # Blocks pass through each other: all of an environment's blocks move at once.
            MP: tl.constexpr = triton.next_power_of_2(M)
            j = tl.arange(0, MP)
            row = offs[:, None] * M + j[None, :]
            on = mask[:, None] & (j[None, :] < M)
            mv = go[:, None] & (j[None, :] < count[:, None]) & ~(has[:, None] & (ci[:, None] == j[None, :]))
            r = tl.load(rad + row, mask=on, other=1.0)
            for axis in tl.static_range(2):
                k = axis * 2
                v = tl.load(vel + row * 3 + k, mask=on, other=0.0)
                cur = tl.load(bpos + row * 3 + k, mask=on, other=0.0)
                other = tl.load(bpos + row * 3 + 2 - k, mask=on, other=0.0)
                cand = cur + v
                back = cur - v
                if axis == 0:
                    free = _free(cand, other, r, ax[:, None], az[:, None], S, AR, zero, not TORUS, AGENT_OV)
                    free2 = _free(back, other, r, ax[:, None], az[:, None], S, AR, zero, not TORUS, AGENT_OV)
                else:
                    free = _free(other, cand, r, ax[:, None], az[:, None], S, AR, zero, not TORUS, AGENT_OV)
                    free2 = _free(other, back, r, ax[:, None], az[:, None], S, AR, zero, not TORUS, AGENT_OV)
                new, nv, moving = _axis(cur, v, cand, back, free, free2, r, mv, S, zero, TORUS, GRID)
                tl.store(bpos + row * 3 + k, new, mask=on & mv)
                tl.store(vel + row * 3 + k, nv, mask=on & moving)
        else:
            for m in range(M):
                mv = go & (m < count) & ~(has & (ci == m))
                r = tl.load(rad + offs * M + m, mask=mask, other=1.0)
                who = ci * 0 + m
                p = bpos + (offs * M + m) * 3
                for axis in tl.static_range(2):
                    k = axis * 2
                    v = tl.load(vel + (offs * M + m) * 3 + k, mask=mask, other=0.0)
                    cur = tl.load(p + k, mask=mask, other=0.0)
                    other = tl.load(p + 2 - k, mask=mask, other=0.0)
                    cand = cur + v
                    back = cur - v
                    if axis == 0:
                        x0, z0, x1, z1 = cand, other, back, other
                    else:
                        x0, z0, x1, z1 = other, cand, other, back
                    free = ~_blocked(
                        x0, z0, r, who, offs, mask, count, apos, bpos, rad, S, AR, zero, not TORUS, AGENT_OV, BOX_OV, M
                    )  # noqa: E501
                    free2 = ~_blocked(
                        x1, z1, r, who, offs, mask, count, apos, bpos, rad, S, AR, zero, not TORUS, AGENT_OV, BOX_OV, M
                    )  # noqa: E501
                    new, nv, moving = _axis(cur, v, cand, back, free, free2, r, mv, S, zero, TORUS, GRID)
                    tl.store(p + k, new, mask=mv)
                    tl.store(vel + (offs * M + m) * 3 + k, nv, mask=moving)
    if WRAP:
        full = tl.load(C + C_TURN_FULL)
        rem = libdevice.fmod(d, full)
        d = tl.where(live, tl.where((rem != zero) & ((rem < zero) != (full < zero)), rem + full, rem), d)
    tl.store(adir + offs, d, mask=live)
    tl.store(carry + offs, c, mask=live)
    tl.store(steps_ptr + offs, steps, mask=live)
    tl.store(truncated + offs, (live & trunc).to(tl.int8), mask=mask)


def step(s: dict, actions: torch.Tensor, active: torch.Tensor | None, C: torch.Tensor, flags: dict, truncated):
    """Step every environment (or the ``active`` ones) of State dict ``s`` in place; fills int8 ``truncated``."""
    B, M = s["bdir"].shape
    grid = (triton.cdiv(B, BLOCK),)
    step_kernel[grid](
        actions, actions if active is None else active, s["apos"], s["adir"], s["carry"], s["count"], s["rad"],
        s["height"], s["bpos"], s["bdir"], s["vel"], s["steps"], truncated, C, B,
        M=M, BLOCK=BLOCK, ALL=active is None, enable_fp_fusion=False, num_warps=1, **flags,
    )  # fmt: skip
