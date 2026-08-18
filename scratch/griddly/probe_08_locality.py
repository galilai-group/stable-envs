"""Probe 8: is the frame a per-cell gather, and from what index function?

For each game, build a map  local_key(cell) -> md5 of that cell's T x T block, where

    local_key(x,y) = (sorted objects at (x,y) with name/orientation/playerId/renderTileId,
                      + the occupancy of the 8 neighbours by each object type)

The 8-neighbour term is there because griddly's `TilingMode: WALL_16` picks the wall image from
the neighbours rather than from state (RenderTileId stays 0 for walls -- see probe_05).

Reports:
  * collisions with the cell-only key      -> how much a pure per-cell atlas would miss
  * collisions with the cell+neighbour key -> 0 means the frame is a gather indexed by a
    2-cell-wide stencil, i.e. exactly the tile-atlas model
  * the number of distinct (key -> tile) entries, i.e. the atlas size the implementation needs

Also: reset cost after the first (setup amortisation), and two games with sprite Scale != 1
and with RotateWithAvatar player observers.
"""
import json, hashlib, collections
import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES = [
    ("sokoban", "Single-Player/GVGAI/sokoban.yaml"),
    ("zelda", "Single-Player/GVGAI/zelda.yaml"),
    ("labyrinth", "Single-Player/GVGAI/labyrinth.yaml"),
    ("butterflies", "Single-Player/GVGAI/butterflies.yaml"),
    ("drunkdwarf", "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml"),
    ("foragers", "Multi-Agent/foragers.yaml"),          # sprite Scale 0.5 / 0.8
    ("clusters", "Single-Player/GVGAI/clusters.yaml"),
    ("GriddlyRTS", "RTS/GriddlyRTS.yaml"),              # 2 players -> HighlightPlayers default true
]


def gframe(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def cellmap(st):
    m = collections.defaultdict(list)
    for o in st["Objects"]:
        m[tuple(o["Location"])].append(
            (o["Name"], o["Orientation"], o["PlayerId"], o["RenderTileId"]))
    return {k: tuple(sorted(v)) for k, v in m.items()}


def run(name, yf, episodes=10, ep_len=50):
    env = GymWrapper(yaml_file=yf, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    T = env.get_tile_size(0)[0]
    cell_only, cell_nb = {}, {}
    coll_cell = coll_nb = 0
    n_cells = 0
    ex_nb = []
    for ep in range(episodes):
        env.seed(300 + ep)
        env.reset(level_id=ep % env.level_count)
        rng = np.random.default_rng(ep * 13 + 1)
        for t in range(ep_len):
            st = env.get_state()
            fr = gframe(env)
            cm = cellmap(st)
            W, H = st["Grid"]["Width"], st["Grid"]["Height"]
            for y in range(H):
                for x in range(W):
                    blk = fr[y * T:(y + 1) * T, x * T:(x + 1) * T]
                    h = hashlib.md5(np.ascontiguousarray(blk)).digest()
                    k1 = cm.get((x, y), ())
                    nb = tuple(cm.get((x + dx, y + dy), ("__OUT__",))
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
                    k2 = (k1, nb)
                    n_cells += 1
                    if cell_only.setdefault(k1, h) != h:
                        coll_cell += 1
                    if cell_nb.setdefault(k2, h) != h:
                        coll_nb += 1
                        if len(ex_nb) < 2:
                            ex_nb.append(str(k2)[:200])
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done):
                break
    # reset cost
    import time
    env.reset(level_id=0)
    t0 = time.perf_counter()
    for i in range(50):
        env.reset(level_id=i % env.level_count)
    reset_ms = (time.perf_counter() - t0) / 50 * 1e3
    out = dict(game=name, tile=T, cells=n_cells,
               atlas_cell_only=len(cell_only), collisions_cell_only=coll_cell,
               atlas_cell_plus_neighbours=len(cell_nb), collisions_cell_plus_neighbours=coll_nb,
               reset_ms=reset_ms, examples=ex_nb)
    del env
    return out


if __name__ == "__main__":
    rows = []
    for n, y in GAMES:
        try:
            r = run(n, y)
        except Exception as e:
            import traceback; traceback.print_exc(); r = dict(game=n, error=str(e))
        rows.append(r); print(json.dumps(r), flush=True)
    print("\n=== SUMMARY (decoder index function) ===")
    for r in rows:
        if "error" in r:
            print(f"{r['game']:12s} ERROR {r['error'][:70]}"); continue
        print(f"{r['game']:12s} T={r['tile']:2d} cells={r['cells']:7d} | "
              f"cell-only atlas={r['atlas_cell_only']:4d} collisions={r['collisions_cell_only']:6d} | "
              f"cell+8nb atlas={r['atlas_cell_plus_neighbours']:5d} COLLISIONS={r['collisions_cell_plus_neighbours']} | "
              f"reset={r['reset_ms']:.2f} ms")
