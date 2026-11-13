"""Numba CPU step of the TwoRoom and Piecewise classes: physics, state and task outputs in one pass.

Each float32 operation of the Torch path is computed in float64 and rounded to float32. A float64
result of one float32 addition, subtraction, multiplication or square root rounds to the correctly
rounded float32 value, so the bytes match the Torch path. The explicit rounding also stops the
compiler from fusing a multiply and an add.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .envs import BORDER, DOOR_MARGIN, GRID_N, IMG, MAX_DOOR, WALL_CENTER, WALL_PUSHBACK

TWO_ROOM = 0
PIECEWISE = 1

_F32 = np.float32
_F64 = np.float64
_LOW = float(BORDER)
_HIGH = float(IMG - BORDER)
_PLAY = float(IMG - 2 * BORDER)
_CENTER = float(WALL_CENTER)


@njit(cache=True, inline="always")
def _clamp_action(x):
    if x != x:
        return x
    if x < _F32(-1.0):
        return _F32(-1.0)
    if x > _F32(1.0):
        return _F32(1.0)
    return x


@njit(cache=True, inline="always")
def _clamp(x, lo, hi):
    """``torch.maximum(torch.minimum(x, hi), lo)`` for one float32 element."""
    if x != x or hi != hi or lo != lo:
        return _F32(np.nan)
    v = hi if hi < x else x
    return lo if v < lo else v


@njit(cache=True, inline="always")
def _move(px, py, ax, ay, speed):
    """Return ``pos + clamp(a) * speed`` with float32 rounding after the product and the sum."""
    dx = _F32(_F64(_clamp_action(ax)) * _F64(speed))
    dy = _F32(_F64(_clamp_action(ay)) * _F64(speed))
    return _F32(_F64(px) + _F64(dx)), _F32(_F64(py) + _F64(dy))


@njit(cache=True, inline="always")
def _border(x, y, radius):
    lo = _F32(_LOW + _F64(radius))
    hi = _F32(_HIGH - _F64(radius))
    return _clamp(x, lo, hi), _clamp(y, lo, hi)


@njit(cache=True)
def _two_room(px, py, ax, ay, row, cols):
    radius = row[cols[1]]
    nx, ny = _move(px, py, ax, ay, row[cols[0]])
    nx, ny = _border(nx, ny, radius)
    half = np.int64(np.trunc(row[cols[2]])) // 2
    axis1 = row[cols[3]] > _F32(0.5)
    eff_lo = _F32((_CENTER - _F64(half)) - _F64(radius))
    eff_hi = _F32((_CENTER + _F64(half)) + _F64(radius))
    moving = nx if axis1 else ny
    started = (px if axis1 else py) < _F32(_CENTER)
    other = ny if axis1 else nx
    number = _F32(np.trunc(row[cols[4]]))
    door = False
    for i in range(MAX_DOOR):
        p = _F64(row[cols[5] + i])
        s = _F64(row[cols[6] + i])
        lo = _F32(_F64(_F32(p - s)) - DOOR_MARGIN)
        hi = _F32(_F64(_F32(p + s)) + DOOR_MARGIN)
        if other >= lo and other <= hi and number > _F32(i):
            door = True
    moved = moving
    if started and moving > eff_lo and not door:
        moved = _F32(_F64(eff_lo) - WALL_PUSHBACK)
    elif not started and moving < eff_hi and not door:
        moved = _F32(_F64(eff_hi) + WALL_PUSHBACK)
    if axis1:
        return moved, ny
    return nx, moved


@njit(cache=True, inline="always")
def _zone_index(p):
    v = np.trunc((_F64(p) - _LOW) / _PLAY * GRID_N)
    # Torch converts NaN and out-of-range values to the lowest int64, which the clamp maps to zone 0.
    if not (v > -9.2e18 and v < 9.2e18) or v < 0:
        return 0
    if v > GRID_N - 1:
        return GRID_N - 1
    return np.int64(v)


@njit(cache=True)
def _piecewise(px, py, ax, ay, row, cols):
    zone = _zone_index(py) * GRID_N + _zone_index(px)
    nx, ny = _move(px, py, ax, ay, row[cols[0]])
    bias = cols[2 + zone]
    nx = _F32(_F64(nx) + _F64(row[bias]))
    ny = _F32(_F64(ny) + _F64(row[bias + 1]))
    return _border(nx, ny, row[cols[1]])


@njit(cache=True)
def advance(kind, rows, action, style, cols, active, masked):
    """Move columns 0 and 1 of ``rows`` in place; lanes with a false ``active`` entry hold when ``masked``."""
    for i in range(rows.shape[0]):
        if masked and not active[i]:
            continue
        if kind == TWO_ROOM:
            x, y = _two_room(rows[i, 0], rows[i, 1], action[i, 0], action[i, 1], style[i], cols)
        else:
            x, y = _piecewise(rows[i, 0], rows[i, 1], action[i, 0], action[i, 1], style[i], cols)
        rows[i, 0] = x
        rows[i, 1] = y


@njit(cache=True)
def state_step(kind, rows, action, style, cols, active, masked):
    """Advance ``rows`` and return copies of the agent and the other columns, and whether all are finite."""
    advance(kind, rows, action, style, cols, active, masked)
    n, width = rows.shape
    agent = np.empty((n, 2), np.float32)
    rest = np.empty((n, width - 2), np.float32)
    finite = True
    for i in range(n):
        for j in range(width):
            v = rows[i, j]
            if not np.isfinite(v):
                finite = False
            if j < 2:
                agent[i, j] = v
            else:
                rest[i, j - 2] = v
    return agent, rest, finite


@njit(cache=True)
def task_step(kind, rows, action, style, cols, active, masked, target, doors):
    """Advance ``rows`` and return state and task outputs of the classes with a hidden target.

    The tuple holds agent, target, observation, proprio, info state, goal state, distance and
    terminated. ``doors`` appends the TwoRoom door centers to the observation.
    """
    advance(kind, rows, action, style, cols, active, masked)
    n = rows.shape[0]
    width = 4 + 2 * MAX_DOOR if doors else 4
    agent = np.empty((n, 2), np.float32)
    goal = np.empty((n, 2), np.float32)
    observation = np.zeros((n, width), np.float32)
    proprio = np.empty((n, 2), np.float32)
    state = np.empty((n, 2), np.float32)
    goal_state = np.empty((n, 2), np.float32)
    distance = np.empty(n, np.float64)
    terminated = np.empty(n, np.bool_)
    for i in range(n):
        x = rows[i, 0]
        y = rows[i, 1]
        tx = target[i, 0]
        ty = target[i, 1]
        for out in (agent, proprio, state):
            out[i, 0] = x
            out[i, 1] = y
        for out in (goal, goal_state):
            out[i, 0] = tx
            out[i, 1] = ty
        observation[i, 0] = x
        observation[i, 1] = y
        observation[i, 2] = tx
        observation[i, 3] = ty
        # Round x*x, add y*y with one float64 rounding, round to float32, then round the square root.
        dx = _F64(_F32(_F64(x) - _F64(tx)))
        dy = _F64(_F32(_F64(y) - _F64(ty)))
        first = _F32(dx * dx)
        squared = _F32(_F64(first) + dy * dy)
        d = _F64(_F32(math.sqrt(_F64(squared))))
        distance[i] = d
        terminated[i] = d < 16.0
        if doors:
            row = style[i]
            axis1 = row[cols[3]] > _F32(0.5)
            number = _F32(np.trunc(row[cols[4]]))
            for k in range(MAX_DOOR):
                if _F32(k) < number:
                    p = row[cols[5] + k]
                    observation[i, 4 + 2 * k] = _F32(_CENTER) if axis1 else p
                    observation[i, 5 + 2 * k] = p if axis1 else _F32(_CENTER)
    return agent, goal, observation, proprio, state, goal_state, distance, terminated
