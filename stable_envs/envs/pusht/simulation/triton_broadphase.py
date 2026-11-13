"""Triton port of ``numba_broadphase``: Munk2D's bounding-box tree reindex for each environment, in float32.

Ported from Munk2D ``src/cpBBTree.c`` at commit ade7ed72849e. Copyright (c) 2025 Victor Blomqvist, Copyright
(c) 2007-2024 Scott Lembcke and Howling Moon Software. MIT license; see licenses/Munk2D-MIT.txt.

The tree layout is ``numba_broadphase``'s. ``work`` is an int32 ``[WORK, B]`` scratch per launch: the reported
pairs, two stacks and the refit flags.
"""

from __future__ import annotations

import triton
import triton.language as tl

from . import numba_broadphase as layout

LEAVES = tl.constexpr(layout.LEAVES)
LIST = tl.constexpr(layout.LIST)
NODES = tl.constexpr(layout.NODES)
PAIRS = tl.constexpr(layout.PAIRS)
WIDTH = tl.constexpr(layout.WIDTH)
# ``work`` fields: the reported pairs, the query stack, the tree walk stack and the refit flags.
_CALLS = layout.LEAVES * layout.PAIRS
_STACK = tl.constexpr(2 * _CALLS)
_PATH = tl.constexpr(2 * _CALLS + layout.NODES + 1)
_FRESH = tl.constexpr(2 * _CALLS + 2 * layout.NODES + 2)
#: The first ``work`` field after the broadphase's own.
WORK = tl.constexpr(2 * _CALLS + 2 * layout.NODES + 2 + layout.LEAVES)


@triton.jit
def _get(tree, lane, index, mask):
    # Volatile, and the loops below start with a barrier: each thread reads the tree right after it rewrites
    # it, and a cached or reordered read misses the write.
    return tl.load(tree + lane * WIDTH + index, mask=mask, other=-1, volatile=True)


@triton.jit
def _set(tree, lane, index, value, mask):
    tl.store(tree + lane * WIDTH + index, value, mask=mask)


@triton.jit
def _bb(bounds, lane, node, k: tl.constexpr, mask):
    return tl.load(bounds + (lane * NODES + node) * 4 + k, mask=mask, other=0.0, volatile=True).to(tl.float32)


@triton.jit
def _set_bb(bounds, lane, node, k: tl.constexpr, value, mask):
    tl.store(bounds + (lane * NODES + node) * 4 + k, value.to(tl.float64), mask=mask)


@triton.jit
def _work(work, field, B, lane, mask):
    return tl.load(work + field * B + lane, mask=mask, other=0, volatile=True)


@triton.jit
def _put(work, field, B, lane, value, mask):
    tl.store(work + field * B + lane, value.to(tl.int32), mask=mask)


@triton.jit
def _area(bounds, lane, n, mask):
    return (_bb(bounds, lane, n, 2, mask) - _bb(bounds, lane, n, 0, mask)) * (
        _bb(bounds, lane, n, 3, mask) - _bb(bounds, lane, n, 1, mask)
    )


@triton.jit
def _merged_area(bounds, lane, a, b, mask):
    width = tl.maximum(_bb(bounds, lane, a, 2, mask), _bb(bounds, lane, b, 2, mask)) - tl.minimum(
        _bb(bounds, lane, a, 0, mask), _bb(bounds, lane, b, 0, mask)
    )
    height = tl.maximum(_bb(bounds, lane, a, 3, mask), _bb(bounds, lane, b, 3, mask)) - tl.minimum(
        _bb(bounds, lane, a, 1, mask), _bb(bounds, lane, b, 1, mask)
    )
    return width * height


@triton.jit
def _proximity(bounds, lane, a, b, mask):
    return tl.abs(
        _bb(bounds, lane, a, 0, mask)
        + _bb(bounds, lane, a, 2, mask)
        - _bb(bounds, lane, b, 0, mask)
        - _bb(bounds, lane, b, 2, mask)
    ) + tl.abs(
        _bb(bounds, lane, a, 1, mask)
        + _bb(bounds, lane, a, 3, mask)
        - _bb(bounds, lane, b, 1, mask)
        - _bb(bounds, lane, b, 3, mask)
    )


@triton.jit
def _overlap(bounds, lane, a, l, b, r, t, mask):
    """Whether node ``a``'s box intersects the box (l, b, r, t)."""
    return (
        (_bb(bounds, lane, a, 0, mask) <= r)
        & (l <= _bb(bounds, lane, a, 2, mask))
        & (_bb(bounds, lane, a, 1, mask) <= t)
        & (b <= _bb(bounds, lane, a, 3, mask))
    )


@triton.jit
def _insert(tree, bounds, lane, leaf, go):
    """``SubtreeInsert`` of ``leaf`` from the root, in the lanes ``go``."""
    node = _get(tree, lane, 0, go)
    empty = go & (node < 0)
    _set(tree, lane, 0, leaf, empty)
    _set(tree, lane, 1 + 3 * leaf, tl.full(leaf.shape, -1, tl.int64), empty)
    go = go & (node >= 0)
    above = tl.full(leaf.shape, -1, tl.int64)
    side = tl.zeros(leaf.shape, tl.int64)
    while tl.max(go.to(tl.int32), axis=0) > 0:
        tl.debug_barrier()
        make = go & (node < LEAVES)
        inner = tl.full(leaf.shape, -1, tl.int64)
        for n in tl.static_range(LEAVES, NODES):
            free = _get(tree, lane, 1 + 3 * n, make) == -2
            inner = tl.where((inner < 0) & free, n, inner)
        for k in tl.static_range(4):
            if k < 2:
                value = tl.minimum(_bb(bounds, lane, leaf, k, make), _bb(bounds, lane, node, k, make))
            else:
                value = tl.maximum(_bb(bounds, lane, leaf, k, make), _bb(bounds, lane, node, k, make))
            _set_bb(bounds, lane, inner, k, value, make)
        _set(tree, lane, 2 + 3 * inner, leaf, make)
        _set(tree, lane, 3 + 3 * inner, node, make)
        _set(tree, lane, 1 + 3 * leaf, inner, make)
        _set(tree, lane, 1 + 3 * node, inner, make)
        _set(tree, lane, 1 + 3 * inner, above, make)
        _set(tree, lane, 0, inner, make & (above < 0))
        _set(tree, lane, 2 + side + 3 * above, inner, make & (above >= 0))
        down = go & (node >= LEAVES)
        a = _get(tree, lane, 2 + 3 * node, down)
        b = _get(tree, lane, 3 + 3 * node, down)
        cost_a = _area(bounds, lane, b, down) + _merged_area(bounds, lane, a, leaf, down)
        cost_b = _area(bounds, lane, a, down) + _merged_area(bounds, lane, b, leaf, down)
        tie = cost_a == cost_b
        cost_a = tl.where(tie, _proximity(bounds, lane, a, leaf, down), cost_a)
        cost_b = tl.where(tie, _proximity(bounds, lane, b, leaf, down), cost_b)
        turn = tl.where(cost_b < cost_a, 1, 0)
        for k in tl.static_range(4):
            if k < 2:
                value = tl.minimum(_bb(bounds, lane, node, k, down), _bb(bounds, lane, leaf, k, down))
            else:
                value = tl.maximum(_bb(bounds, lane, node, k, down), _bb(bounds, lane, leaf, k, down))
            _set_bb(bounds, lane, node, k, value, down)
        above = tl.where(down, node, above)
        side = tl.where(down, turn, side)
        node = tl.where(down, tl.where(turn == 1, b, a), node)
        go = down


@triton.jit
def _remove(tree, bounds, lane, leaf, go):
    """``SubtreeRemove`` of ``leaf`` from the root, in the lanes ``go``."""
    gone = tl.full(leaf.shape, -2, tl.int64)
    root = _get(tree, lane, 0, go)
    alone = go & (root == leaf)
    _set(tree, lane, 0, tl.full(leaf.shape, -1, tl.int64), alone)
    _set(tree, lane, 1 + 3 * leaf, gone, alone)
    go = go & (root != leaf)
    parent = _get(tree, lane, 1 + 3 * leaf, go)
    a = _get(tree, lane, 2 + 3 * parent, go)
    b = _get(tree, lane, 3 + 3 * parent, go)
    other = tl.where(a == leaf, b, a)
    grand = _get(tree, lane, 1 + 3 * parent, go)
    _set(tree, lane, 1 + 3 * leaf, gone, go)
    _set(tree, lane, 1 + 3 * parent, gone, go)
    _set(tree, lane, 1 + 3 * other, grand, go)
    _set(tree, lane, 0, other, go & (grand < 0))
    go = go & (grand >= 0)
    first = _get(tree, lane, 2 + 3 * grand, go)
    _set(tree, lane, tl.where(first == parent, 2, 3) + 3 * grand, other, go)
    node = grand
    while tl.max(go.to(tl.int32), axis=0) > 0:
        tl.debug_barrier()
        a = _get(tree, lane, 2 + 3 * node, go)
        b = _get(tree, lane, 3 + 3 * node, go)
        for k in tl.static_range(4):
            if k < 2:
                value = tl.minimum(_bb(bounds, lane, a, k, go), _bb(bounds, lane, b, k, go))
            else:
                value = tl.maximum(_bb(bounds, lane, a, k, go), _bb(bounds, lane, b, k, go))
            _set_bb(bounds, lane, node, k, value, go)
        node = _get(tree, lane, 1 + 3 * node, go)
        go = go & (node >= 0)


@triton.jit
def _push(tree, lane, leaf, entry, go):
    base = LIST + leaf * PAIRS
    for k in range(PAIRS - 1):
        j = PAIRS - 1 - k
        _set(tree, lane, base + j, _get(tree, lane, base + j - 1, go), go)
    _set(tree, lane, base, entry, go)


@triton.jit
def _drop(tree, lane, leaf, other, go):
    """Remove the entry for ``other`` from ``leaf``'s pair list."""
    base = LIST + leaf * PAIRS
    at = tl.zeros(leaf.shape, tl.int64)
    for j in range(PAIRS):
        entry = _get(tree, lane, base + j, go)
        keep = go & (entry >= 0) & (entry // 2 != other)
        _set(tree, lane, base + at, entry, keep)
        at += keep.to(tl.int64)
    for j in range(PAIRS):
        _set(tree, lane, base + j, tl.full(leaf.shape, -1, tl.int64), go & (at <= j))


@triton.jit
def _pair(tree, lane, a, b, go):
    """``PairInsert``: ``a`` first, ``b`` second; walls keep no list."""
    _push(tree, lane, a, 2 * b, go & (a < LEAVES))
    _push(tree, lane, b, 2 * a + 1, go & (b < LEAVES))


@triton.jit
def _clear(tree, lane, leaf, go):
    """``PairsClear``."""
    base = LIST + leaf * PAIRS
    for j in range(PAIRS):
        entry = _get(tree, lane, base + j, go)
        _drop(tree, lane, entry // 2, leaf, go & (entry >= 0) & (entry // 2 < LEAVES))
        _set(tree, lane, base + j, tl.full(leaf.shape, -1, tl.int64), go)


@triton.jit
def _call(work, B, lane, count, a, b, go):
    _put(work, 2 * count, B, lane, a, go)
    _put(work, 2 * count + 1, B, lane, b, go)
    return count + go.to(tl.int32)


@triton.jit
def _walls(tree, bounds, work, B, lane, leaf, count, go):
    """``MarkLeafQuery`` against the walls: their static tree visits walls 3, 2, 1, 0."""
    l, b = _bb(bounds, lane, leaf, 0, go), _bb(bounds, lane, leaf, 1, go)
    r, t = _bb(bounds, lane, leaf, 2, go), _bb(bounds, lane, leaf, 3, go)
    for k in range(4):
        w = 3 - k
        # Wall w's box: (3, 3, 7, 508), (3, 3, 508, 7), (504, 3, 508, 508), (3, 504, 508, 508).
        left = tl.where(w == 2, 504.0, 3.0)
        bottom = tl.where(w == 3, 504.0, 3.0)
        right = tl.where(w == 0, 7.0, 508.0)
        top = tl.where(w == 1, 7.0, 508.0)
        hit = go & (left <= r) & (l <= right) & (bottom <= t) & (b <= top)
        wall = tl.zeros(leaf.shape, tl.int64) + LEAVES + w
        _push(tree, lane, leaf, 2 * wall + 1, hit)
        count = _call(work, B, lane, count, leaf, wall, hit)
    return count


@triton.jit
def _query(tree, bounds, work, B, lane, root, leaf, count, go, left):
    """``MarkLeafQuery`` of ``leaf`` over the dynamic subtree at ``root``; ``left`` per lane."""
    l, b = _bb(bounds, lane, leaf, 0, go), _bb(bounds, lane, leaf, 1, go)
    r, t = _bb(bounds, lane, leaf, 2, go), _bb(bounds, lane, leaf, 3, go)
    top = go.to(tl.int32)
    _put(work, _STACK, B, lane, root, go)
    while tl.max(top, axis=0) > 0:
        tl.debug_barrier()
        on = top > 0
        top -= on.to(tl.int32)
        node = _work(work, _STACK + top, B, lane, on).to(tl.int64)
        hit = on & _overlap(bounds, lane, node, l, b, r, t, on)
        found = hit & (node < LEAVES)
        fresh = _work(work, _FRESH + node, B, lane, found & ~left) != 0
        # Left: pair (leaf, node). Right: pair (node, leaf) unless node also refit, and report it.
        insert = found & (left | ~fresh)
        _pair(tree, lane, tl.where(left, leaf, node), tl.where(left, node, leaf), insert)
        count = _call(work, B, lane, count, leaf, node, found & ~left)
        inner = hit & (node >= LEAVES)
        _put(work, _STACK + top, B, lane, _get(tree, lane, 3 + 3 * node, inner), inner)
        _put(work, _STACK + top + 1, B, lane, _get(tree, lane, 2 + 3 * node, inner), inner)
        top += 2 * inner.to(tl.int32)
    return count


@triton.jit
def reindex(tree, bounds, work, tight, order, B, lane, mask, need):
    """``numba_broadphase.refit`` and ``report`` for the lanes ``mask``; returns the reported pair counts.

    ``tight`` is a float32 ``[6 * LEAVES, B]`` scratch: the leaves' boxes, then their body velocities
    (fields ``4 * LEAVES + 2 * leaf``). A lane reports only if a leaf moved or it ``need``s the order.
    """
    tl.debug_barrier()
    for k in range(LEAVES):
        _put(work, _FRESH + k, B, lane, tl.zeros(lane.shape, tl.int32), mask)
    for j in range(LEAVES):
        leaf = tl.load(order + lane * LEAVES + j, mask=mask, other=-1)
        go = mask & (leaf >= 0)
        tl_ = tl.load(tight + (4 * leaf) * B + lane, mask=go, other=0.0)
        tb = tl.load(tight + (4 * leaf + 1) * B + lane, mask=go, other=0.0)
        tr = tl.load(tight + (4 * leaf + 2) * B + lane, mask=go, other=0.0)
        tt = tl.load(tight + (4 * leaf + 3) * B + lane, mask=go, other=0.0)
        inside = (
            (_bb(bounds, lane, leaf, 0, go) <= tl_)
            & (_bb(bounds, lane, leaf, 2, go) >= tr)
            & (_bb(bounds, lane, leaf, 1, go) <= tb)
            & (_bb(bounds, lane, leaf, 3, go) >= tt)
        )
        go = go & ~inside
        # Most steps move no leaf out of its box in a whole block of environments.
        if tl.max(go.to(tl.int32), axis=0) > 0:
            # ``GetBB``: the tight box grown by a tenth of its size and by a tenth of the velocity.
            x = (tr - tl_) * 0.1
            y = (tt - tb) * 0.1
            vx = tl.load(tight + (4 * LEAVES + 2 * leaf) * B + lane, mask=go, other=0.0) * 0.1
            vy = tl.load(tight + (4 * LEAVES + 2 * leaf + 1) * B + lane, mask=go, other=0.0) * 0.1
            _set_bb(bounds, lane, leaf, 0, tl_ + tl.minimum(-x, vx), go)
            _set_bb(bounds, lane, leaf, 1, tb + tl.minimum(-y, vy), go)
            _set_bb(bounds, lane, leaf, 2, tr + tl.maximum(x, vx), go)
            _set_bb(bounds, lane, leaf, 3, tt + tl.maximum(y, vy), go)
            _remove(tree, bounds, lane, leaf, go)
            _insert(tree, bounds, lane, leaf, go)
            _clear(tree, lane, leaf, go)
            _put(work, _FRESH + leaf, B, lane, tl.full(lane.shape, 1, tl.int32), go)
            tl.debug_barrier()
    count = tl.zeros(lane.shape, tl.int32)
    moved = tl.zeros(lane.shape, tl.int32)
    for k in range(LEAVES):
        moved = tl.maximum(moved, _work(work, _FRESH + k, B, lane, mask))
    run = mask & ((moved != 0) | need)
    # ``MarkSubtree``: leaves depth first, child A before child B.
    root = _get(tree, lane, 0, run)
    top = (run & (root >= 0)).to(tl.int32)
    _put(work, _PATH, B, lane, root, top > 0)
    while tl.max(top, axis=0) > 0:
        tl.debug_barrier()
        on = top > 0
        top -= on.to(tl.int32)
        node = _work(work, _PATH + top, B, lane, on).to(tl.int64)
        inner = on & (node >= LEAVES)
        _put(work, _PATH + top, B, lane, _get(tree, lane, 3 + 3 * node, inner), inner)
        _put(work, _PATH + top + 1, B, lane, _get(tree, lane, 2 + 3 * node, inner), inner)
        top += 2 * inner.to(tl.int32)
        found = on & (node < LEAVES)
        fresh = found & (_work(work, _FRESH + node, B, lane, found) != 0)
        count = _walls(tree, bounds, work, B, lane, node, count, fresh)
        child = node
        parent = _get(tree, lane, 1 + 3 * child, fresh)
        up = fresh & (parent >= 0)
        while tl.max(up.to(tl.int32), axis=0) > 0:
            tl.debug_barrier()
            first = _get(tree, lane, 2 + 3 * parent, up)
            second = _get(tree, lane, 3 + 3 * parent, up)
            is_a = first == child
            count = _query(tree, bounds, work, B, lane, tl.where(is_a, second, first), node, count, up, is_a)
            child = tl.where(up, parent, child)
            parent = _get(tree, lane, 1 + 3 * child, up)
            up = up & (parent >= 0)
        listed = found & ~fresh
        for j in range(PAIRS):
            entry = _get(tree, lane, LIST + node * PAIRS + j, listed)
            listed = listed & (entry >= 0)
            count = _call(work, B, lane, count, entry // 2, node, listed & (entry % 2 == 1))
    return count


@triton.jit
def pair_at(work, B, lane, c, mask):
    """The ``c``-th reported pair."""
    return _work(work, 2 * c, B, lane, mask), _work(work, 2 * c + 1, B, lane, mask)
