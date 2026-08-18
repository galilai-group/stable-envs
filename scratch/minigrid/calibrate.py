"""Extract MiniGrid's tile appearance set and view geometry from the official renderer.

Nothing here is assumed. Everything a GPU renderer needs is *measured* against
``minigrid.core.grid.Grid``:

  * that ``render_tile`` output is a pure function of ``(obj.encode(), agent_dir,
    highlight, tile_size)`` -- checked by rendering objects that share an encoding
    but were built by different code paths, and diffing pixels;
  * the finite set of distinct tile images, and how many of the dense
    (type, color, state) x agent_dir x highlight product is actually distinct;
  * the agent-view geometry: the composite of ``Grid.slice`` + ``rotate_left**(d+1)``
    is traced with tagged objects rather than re-derived, and the resulting offset
    table is cross-checked against ``get_full_render``'s own highlight formula.

Writes atlas.npz + meta.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from minigrid.core.constants import (
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
)
from minigrid.core.grid import Grid
from minigrid.core.world_object import (
    Ball,
    Box,
    Door,
    Floor,
    Goal,
    Key,
    Lava,
    Wall,
    WorldObj,
)

OUT = os.path.dirname(os.path.abspath(__file__))

N_TYPE = len(OBJECT_TO_IDX)   # 11
N_COLOR = len(COLOR_TO_IDX)   # 6
N_STATE = 3
N_DIR = 5                     # 0..3 = agent facing, 4 = no agent
N_HL = 2


def make_obj(t: int, c: int, s: int) -> WorldObj | None:
    """Build the object that ``Grid.decode`` *would* build, except honouring colour.

    ``WorldObj.decode`` throws colour away for Goal and Lava. The renderer does not
    (``Goal.render`` reads ``self.color``), so an atlas built through ``decode``
    would silently be wrong for a non-green goal. Build faithfully instead.
    """
    name = IDX_TO_OBJECT[t]
    col = IDX_TO_COLOR[c]
    if name in ("unseen", "empty", "agent"):
        return None
    if name == "wall":
        return Wall(col)
    if name == "floor":
        return Floor(col)
    if name == "door":
        return Door(col, is_open=(s == 0), is_locked=(s == 2))
    if name == "key":
        return Key(col)
    if name == "ball":
        return Ball(col)
    if name == "box":
        return Box(col)
    if name == "goal":
        return Goal(col)
    if name == "lava":
        return Lava()
    raise AssertionError(name)


def atlas_index(hl: int, d: int, t: int, c: int, s: int) -> int:
    return ((((hl * N_DIR + d) * N_TYPE + t) * N_COLOR + c) * N_STATE + s)


def build_atlas(tile_size: int) -> np.ndarray:
    n = N_HL * N_DIR * N_TYPE * N_COLOR * N_STATE
    atlas = np.zeros((n, tile_size, tile_size, 3), dtype=np.uint8)
    for hl in range(N_HL):
        for d in range(N_DIR):
            for t in range(N_TYPE):
                for c in range(N_COLOR):
                    for s in range(N_STATE):
                        obj = make_obj(t, c, s)
                        img = Grid.render_tile(
                            obj,
                            agent_dir=(d if d < 4 else None),
                            highlight=bool(hl),
                            tile_size=tile_size,
                        )
                        # Grid.render assigns this float array into a uint8 buffer;
                        # reproduce that cast exactly.
                        atlas[atlas_index(hl, d, t, c, s)] = img.astype(np.uint8)
    return atlas


def check_purity(tile_size: int) -> dict:
    """render_tile must depend on the object only through obj.encode()."""
    Grid.tile_cache.clear()
    cases = []
    # Same encoding, different construction paths / hidden fields.
    cases.append(("box-contains", Box("red"), Box("red", contains=Key("blue"))))
    cases.append(("box-contains2", Box("red"), Box("red", contains=Ball("yellow"))))
    cases.append(("door-open-locked", Door("yellow", is_open=True, is_locked=False),
                  Door("yellow", is_open=True, is_locked=True)))
    cases.append(("decode-vs-ctor", Key("purple"), WorldObj.decode(*Key("purple").encode())))
    cases.append(("decode-vs-ctor-door", Door("grey", is_open=False, is_locked=True),
                  WorldObj.decode(*Door("grey", is_open=False, is_locked=True).encode())))
    o = Ball("green")
    o.init_pos, o.cur_pos = (3, 4), (7, 1)
    cases.append(("posfields", Ball("green"), o))

    res = {}
    for name, a, b in cases:
        assert a.encode() == b.encode(), (name, a.encode(), b.encode())
        bad = 0
        for d in (None, 0, 1, 2, 3):
            for hl in (False, True):
                # bypass the cache, which is keyed on encode() and would trivially
                # make this test pass
                Grid.tile_cache.clear()
                ia = Grid.render_tile(a, agent_dir=d, highlight=hl, tile_size=tile_size)
                Grid.tile_cache.clear()
                ib = Grid.render_tile(b, agent_dir=d, highlight=hl, tile_size=tile_size)
                bad += int(not np.array_equal(ia, ib))
        res[name] = bad
    Grid.tile_cache.clear()
    return res


def enumerate_distinct(atlas: np.ndarray) -> dict:
    h2i: dict[str, int] = {}
    labels: dict[str, list] = {}
    for hl in range(N_HL):
        for d in range(N_DIR):
            for t in range(N_TYPE):
                for c in range(N_COLOR):
                    for s in range(N_STATE):
                        i = atlas_index(hl, d, t, c, s)
                        h = hashlib.sha1(atlas[i].tobytes()).hexdigest()
                        h2i.setdefault(h, i)
                        labels.setdefault(h, []).append(
                            (hl, d, IDX_TO_OBJECT[t], IDX_TO_COLOR[c], s)
                        )
    return {"n_slots": atlas.shape[0], "n_distinct": len(h2i)}


def trace_view_geometry(view_size: int = 7) -> dict:
    """Trace Grid.slice + rotate_left**(d+1) with tagged objects.

    Returns offs[d][vj][vi] = (dx, dy) world offset from agent_pos, or (127,127)
    for the out-of-bounds cells that Grid.slice fills with Wall().
    """

    class Tag(WorldObj):
        def __init__(self, i, j):
            super().__init__("ball", "red")
            self.tag = (i, j)

    # Probe positions are kept >= view_size from every edge so that no view cell
    # falls outside the grid: the offsets are then pure geometry, and whether a
    # cell is OOB is a bounds test the renderer does itself.
    W = H = 4 * view_size + 3
    OOB = 127
    offs = np.full((4, view_size, view_size, 2), OOB, dtype=np.int8)
    m = view_size
    agent_positions = [(m, m), (m + 1, 2 * m), (2 * m, m + 2), (3 * m, 2 * m + 1)]
    consistent = True
    ref = None
    for ap in agent_positions:
        cur = np.full((4, view_size, view_size, 2), OOB, dtype=np.int8)
        for d in range(4):
            g = Grid(W, H)
            for i in range(W):
                for j in range(H):
                    g.set(i, j, Tag(i, j))

            # replicate MiniGridEnv.get_view_exts, verbatim call surface
            ax, ay = ap
            if d == 0:
                topX, topY = ax, ay - view_size // 2
            elif d == 1:
                topX, topY = ax - view_size // 2, ay
            elif d == 2:
                topX, topY = ax - view_size + 1, ay - view_size // 2
            else:
                topX, topY = ax - view_size // 2, ay - view_size + 1

            sub = g.slice(topX, topY, view_size, view_size)
            for _ in range(d + 1):
                sub = sub.rotate_left()
            for vi in range(view_size):
                for vj in range(view_size):
                    cell = sub.get(vi, vj)
                    if isinstance(cell, Tag):
                        cur[d, vj, vi] = (cell.tag[0] - ax, cell.tag[1] - ay)
        if ref is None:
            ref = cur
        else:
            consistent &= bool(np.array_equal(ref, cur))
    offs = ref
    return {"offs": offs, "translation_invariant": consistent, "oob": OOB}


def check_fullrender_highlight_formula(offs: np.ndarray, view_size: int = 7) -> bool:
    """get_full_render maps vis_mask[vi,vj] -> world cell with its own formula.

    Verify it agrees with the traced slice/rotate offsets, i.e. that highlighting
    a world cell is the same index map as reading the view cell.
    """
    from minigrid.core.constants import DIR_TO_VEC

    ok = True
    for d in range(4):
        f_vec = DIR_TO_VEC[d]
        dx, dy = f_vec
        r_vec = np.array((-dy, dx))
        top_left = f_vec * (view_size - 1) - r_vec * (view_size // 2)  # relative to agent
        for vj in range(view_size):
            for vi in range(view_size):
                abs_ij = top_left - (f_vec * vj) + (r_vec * vi)
                traced = offs[d, vj, vi]
                if traced[0] == 127:
                    continue
                ok &= bool(np.array_equal(abs_ij, traced))
    return ok


def main():
    tile_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    meta: dict = {"tile_size": tile_size}
    meta["purity_mismatches"] = check_purity(tile_size)

    atlas = build_atlas(tile_size)
    meta.update(enumerate_distinct(atlas))
    meta["atlas_shape"] = list(atlas.shape)
    meta["atlas_bytes"] = int(atlas.nbytes)
    meta["layout"] = "index = ((((hl*5 + dir)*11 + type)*6 + color)*3 + state)"

    # The offset table is what lets the GPU read the agent view as a gather. Trace it
    # for every view size MiniGrid allows in practice and check the traced table
    # against get_full_render's own (differently written) formula each time, so the
    # closed form used at runtime is validated, not assumed.
    meta["view_sizes"] = {}
    for V in (3, 5, 7, 9, 11):
        g = trace_view_geometry(V)
        meta["view_sizes"][str(V)] = {
            "translation_invariant": g["translation_invariant"],
            "formula_agrees": check_fullrender_highlight_formula(g["offs"], V),
        }
        if V == 7:
            geo = g
    meta["offs_dir0"] = geo["offs"][0].tolist()

    # which (type,color,state) triples can actually appear from a real encode()
    reachable = set()
    for name, t in OBJECT_TO_IDX.items():
        if name in ("unseen", "empty", "agent"):
            reachable.add((t, 0, 0))
            continue
        for c in range(N_COLOR):
            if name == "door":
                for s in range(3):
                    reachable.add((t, c, s))
            elif name == "lava":
                reachable.add((t, COLOR_TO_IDX["red"], 0))
            else:
                reachable.add((t, c, 0))
    meta["n_reachable_encodings"] = len(reachable)
    meta["n_reachable_tiles"] = len(reachable) * N_DIR * N_HL

    np.savez_compressed(
        os.path.join(OUT, f"atlas_{tile_size}.npz"),
        atlas=atlas,
        offs=geo["offs"],
    )
    with open(os.path.join(OUT, f"meta_{tile_size}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "offs_dir0"}, indent=2))


if __name__ == "__main__":
    main()
