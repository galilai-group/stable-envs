"""Munk2D's bounding-box tree broadphase for one PushT scene, which sets the order the solver visits shape pairs.

Munk2D (the Chipmunk2D fork in pymunk 7.2.0) solves arbiters in the order its broadphase reports shape pairs.
With 10 solver iterations that order changes the result, so the ports replay ``cpBBTree``'s reindex query:
``LeafUpdate`` in hash-set order, then ``MarkSubtree``. Ported from Munk2D ``src/cpBBTree.c`` at commit
ade7ed72849e. Copyright (c) 2025 Victor Blomqvist, Copyright (c) 2007-2024 Scott Lembcke and Howling Moon
Software. MIT license; see licenses/Munk2D-MIT.txt.

A scene's tree is one int64 row (``tree``) and one float64 ``[NODES, 4]`` block of bounds (left, bottom, right,
top). Leaves ``0..LEAVES - 1`` are the shapes, ``3 * body + shape`` (agent 0, block 1); nodes from ``LEAVES``
are inner nodes. ``tree[0]`` is the root, ``tree[1 + 3 * n : 4 + 3 * n]`` node ``n``'s parent and children
(-1 for none; a parent of -2 marks a free or absent node), and from ``LIST`` each leaf's pair list, newest
first, as ``2 * other + (1 if the leaf is the pair's second leaf else 0)``; walls are others ``LEAVES..``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

#: Most convex pieces of one body (``numeric.MAX_SHAPES``).
SHAPES = 3
LEAVES = 2 * SHAPES
NODES = 2 * LEAVES - 1
#: Entries of one leaf's pair list: every other leaf and the four walls.
PAIRS = LEAVES - 1 + 4
LIST = 1 + 3 * NODES
WIDTH = LIST + LEAVES * PAIRS
_WALL_RADIUS = 2.0
#: ``GetBB``'s ``0.1f``.
_GROW = float(np.float32(0.1))
_WALL_ENDS = ((5.0, 506.0, 5.0, 5.0), (5.0, 5.0, 506.0, 5.0), (506.0, 5.0, 506.0, 506.0), (5.0, 506.0, 506.0, 506.0))


def _static_tree():
    """Munk2D's static tree after the four walls are inserted in order: bounds, children and the root."""
    bounds = np.zeros((7, 4))
    children = np.full((7, 2), -1, np.int64)
    parent = np.full(7, -1, np.int64)
    for w, (ax, ay, bx, by) in enumerate(_WALL_ENDS):
        r = _WALL_RADIUS
        bounds[w] = (min(ax, bx) - r, min(ay, by) - r, max(ax, bx) + r, max(ay, by) + r)
    root, free = -1, 4
    for leaf in range(4):
        if root < 0:
            root = leaf
            continue
        node, above, side = root, -1, 0
        while True:
            if node < 4:
                inner, free = free, free + 1
                bounds[inner] = _merge(bounds[leaf], bounds[node])
                children[inner] = (leaf, node)
                parent[leaf] = parent[node] = inner
                parent[inner] = above
                if above < 0:
                    root = inner
                else:
                    children[above, side] = inner
                break
            side = _descend(bounds, children[node, 0], children[node, 1], leaf)
            bounds[node] = _merge(bounds[node], bounds[leaf])
            above, node = node, children[node, side]
    return bounds, children, root


def _merge(a, b):
    return np.array((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))


def _descend(bounds, a, b, leaf):
    """``SubtreeInsert``'s choice of child: 1 for ``b``, 0 for ``a``."""
    cost_a = _area(bounds[b]) + _merged_area(bounds[a], bounds[leaf])
    cost_b = _area(bounds[a]) + _merged_area(bounds[b], bounds[leaf])
    if cost_a == cost_b:
        cost_a = _proximity(bounds[a], bounds[leaf])
        cost_b = _proximity(bounds[b], bounds[leaf])
    return 1 if cost_b < cost_a else 0


@njit(cache=True, inline="always")
def _area(bb):
    return (bb[2] - bb[0]) * (bb[3] - bb[1])


@njit(cache=True, inline="always")
def _merged_area(a, b):
    return (max(a[2], b[2]) - min(a[0], b[0])) * (max(a[3], b[3]) - min(a[1], b[1]))


@njit(cache=True, inline="always")
def _proximity(a, b):
    return abs(a[0] + a[2] - b[0] - b[2]) + abs(a[1] + a[3] - b[1] - b[3])


@njit(cache=True, inline="always")
def _intersects(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


@njit(cache=True, inline="always")
def _contains(a, b):
    return a[0] <= b[0] and a[2] >= b[2] and a[1] <= b[1] and a[3] >= b[3]


_STATIC_BOUNDS, _STATIC_CHILDREN, _STATIC_ROOT = _static_tree()


@njit(cache=True)
def _insert(tree, bounds, leaf):
    """``SubtreeInsert`` of ``leaf`` from the root."""
    node, above, side = tree[0], -1, 0
    if node < 0:
        tree[0] = leaf
        tree[1 + 3 * leaf] = -1
        return
    while True:
        if node < LEAVES:
            inner = LEAVES
            while tree[1 + 3 * inner] != -2:
                inner += 1
            for k in range(4):
                bounds[inner, k] = (
                    min(bounds[leaf, k], bounds[node, k]) if k < 2 else max(bounds[leaf, k], bounds[node, k])
                )
            tree[2 + 3 * inner], tree[3 + 3 * inner] = leaf, node
            tree[1 + 3 * leaf] = inner
            tree[1 + 3 * node] = inner
            tree[1 + 3 * inner] = above
            if above < 0:
                tree[0] = inner
            else:
                tree[2 + side + 3 * above] = inner
            return
        a, b = tree[2 + 3 * node], tree[3 + 3 * node]
        cost_a = _area(bounds[b]) + _merged_area(bounds[a], bounds[leaf])
        cost_b = _area(bounds[a]) + _merged_area(bounds[b], bounds[leaf])
        if cost_a == cost_b:
            cost_a = _proximity(bounds[a], bounds[leaf])
            cost_b = _proximity(bounds[b], bounds[leaf])
        side = 1 if cost_b < cost_a else 0
        for k in range(4):
            bounds[node, k] = min(bounds[node, k], bounds[leaf, k]) if k < 2 else max(bounds[node, k], bounds[leaf, k])
        above, node = node, tree[2 + side + 3 * node]


@njit(cache=True)
def _remove(tree, bounds, leaf):
    """``SubtreeRemove`` of ``leaf`` from the root."""
    if tree[0] == leaf:
        tree[0] = -1
        tree[1 + 3 * leaf] = -2
        return
    parent = tree[1 + 3 * leaf]
    other = tree[3 + 3 * parent] if tree[2 + 3 * parent] == leaf else tree[2 + 3 * parent]
    grand = tree[1 + 3 * parent]
    tree[1 + 3 * leaf] = -2
    tree[1 + 3 * parent] = -2
    tree[1 + 3 * other] = grand
    if grand < 0:
        tree[0] = other
        return
    tree[(2 if tree[2 + 3 * grand] == parent else 3) + 3 * grand] = other
    node = grand
    while node >= 0:
        a, b = tree[2 + 3 * node], tree[3 + 3 * node]
        for k in range(4):
            bounds[node, k] = min(bounds[a, k], bounds[b, k]) if k < 2 else max(bounds[a, k], bounds[b, k])
        node = tree[1 + 3 * node]


@njit(cache=True)
def _push(tree, leaf, entry):
    base = LIST + leaf * PAIRS
    for j in range(PAIRS - 1, 0, -1):
        tree[base + j] = tree[base + j - 1]
    tree[base] = entry


@njit(cache=True)
def _drop(tree, leaf, other):
    """Remove the entry for ``other`` from ``leaf``'s pair list."""
    base = LIST + leaf * PAIRS
    at = 0
    for j in range(PAIRS):
        entry = tree[base + j]
        if entry >= 0 and entry // 2 != other:
            tree[base + at] = entry
            at += 1
    for j in range(at, PAIRS):
        tree[base + j] = -1


@njit(cache=True)
def _pair(tree, a, b):
    """``PairInsert``: ``a`` first, ``b`` second; walls keep no list."""
    if a < LEAVES:
        _push(tree, a, 2 * b)
    if b < LEAVES:
        _push(tree, b, 2 * a + 1)


@njit(cache=True)
def _clear(tree, leaf):
    """``PairsClear``."""
    base = LIST + leaf * PAIRS
    for j in range(PAIRS):
        entry = tree[base + j]
        if entry >= 0 and entry // 2 < LEAVES:
            _drop(tree, entry // 2, leaf)
        tree[base + j] = -1


@njit(cache=True)
def _call(calls, count, a, b):
    calls[count, 0], calls[count, 1] = a, b
    return count + 1


@njit(cache=True)
def _query(tree, bounds, root, leaf, left, fresh, calls, count, stack):
    """``MarkLeafQuery`` over the dynamic subtree at ``root`` (or the walls when ``root`` is -1)."""
    walls = root < 0
    top = 1
    stack[0] = _STATIC_ROOT if walls else root
    while top:
        top -= 1
        node = stack[top]
        box = _STATIC_BOUNDS[node] if walls else bounds[node]
        if not _intersects(bounds[leaf], box):
            continue
        if (node < 4) if walls else (node < LEAVES):
            other = LEAVES + node if walls else node
            if left:
                _pair(tree, leaf, other)
            else:
                if walls or not fresh[other]:
                    _pair(tree, other, leaf)
                count = _call(calls, count, leaf, other)
        else:
            a, b = (
                (_STATIC_CHILDREN[node, 0], _STATIC_CHILDREN[node, 1])
                if walls
                else (tree[2 + 3 * node], tree[3 + 3 * node])
            )
            stack[top], stack[top + 1] = b, a
            top += 2
    return count


@njit(cache=True)
def refit(tree, bounds, tight, velocity, order, fresh):
    """``cpBBTreeReindexQuery``'s ``LeafUpdate`` pass: move each leaf whose box left its grown box; True if any.

    ``tight`` holds each leaf's bounding box after the step's position update and ``velocity`` its body's
    velocity. ``order`` lists the present leaves in hash-set order; ``fresh`` marks the moved leaves.
    """
    moved = False
    for j in range(LEAVES):
        fresh[j] = False
    for j in range(LEAVES):
        leaf = order[j]
        if leaf < 0:
            break
        if _contains(bounds[leaf], tight[leaf]):
            continue
        # ``GetBB``: the tight box grown by a tenth of its size and by a tenth of the velocity.
        x = (tight[leaf, 2] - tight[leaf, 0]) * _GROW
        y = (tight[leaf, 3] - tight[leaf, 1]) * _GROW
        vx, vy = velocity[leaf, 0] * _GROW, velocity[leaf, 1] * _GROW
        bounds[leaf, 0] = tight[leaf, 0] + min(-x, vx)
        bounds[leaf, 1] = tight[leaf, 1] + min(-y, vy)
        bounds[leaf, 2] = tight[leaf, 2] + max(x, vx)
        bounds[leaf, 3] = tight[leaf, 3] + max(y, vy)
        _remove(tree, bounds, leaf)
        _insert(tree, bounds, leaf)
        _clear(tree, leaf)
        fresh[leaf] = True
        moved = True
    return moved


@njit(cache=True)
def report(tree, bounds, fresh, calls, stack):
    """``cpBBTreeReindexQuery``'s ``MarkSubtree`` pass: write the shape pairs in report order; return their count.

    Each pair in ``calls`` is two leaves, or a leaf and ``LEAVES + wall``. A moved leaf (``fresh``) queries
    the tree and pairs anew; the others report their pair lists.
    """
    count = 0
    if tree[0] < 0:
        return count
    # Leaves depth first, child A before child B.
    nodes = stack[NODES:]
    top = 1
    nodes[0] = tree[0]
    while top:
        top -= 1
        node = nodes[top]
        if node >= LEAVES:
            nodes[top], nodes[top + 1] = tree[3 + 3 * node], tree[2 + 3 * node]
            top += 2
            continue
        if fresh[node]:
            count = _query(tree, bounds, -1, node, False, fresh, calls, count, stack)
            child = node
            while tree[1 + 3 * child] >= 0:
                parent = tree[1 + 3 * child]
                if tree[2 + 3 * parent] == child:
                    count = _query(tree, bounds, tree[3 + 3 * parent], node, True, fresh, calls, count, stack)
                else:
                    count = _query(tree, bounds, tree[2 + 3 * parent], node, False, fresh, calls, count, stack)
                child = parent
        else:
            base = LIST + node * PAIRS
            for j in range(PAIRS):
                entry = tree[base + j]
                if entry < 0:
                    break
                if entry % 2:
                    count = _call(calls, count, entry // 2, node)
    return count


@njit(cache=True)
def reindex(tree, bounds, tight, velocity, order, fresh, calls, stack):
    """One step's ``cpBBTreeReindexQuery``: ``refit``, then ``report``."""
    refit(tree, bounds, tight, velocity, order, fresh)
    return report(tree, bounds, fresh, calls, stack)
