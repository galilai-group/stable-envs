"""Probe 2: G1 (purity) and G2 (compression) for griddly.

G1 protocol, per game:
  * roll a seeded episode, snapshot (get_state(), global frame) at N checkpoints
  * run a DIFFERENT history (different seed, different level, many steps) on the same env
  * restore each snapshot with game.load_state(), re-render, pixel-diff against the snapshot frame
  * separately: render the same state twice in a row (does render() mutate?)
  * separately: collect every (render-key -> frame md5) pair seen and report collisions,
    i.e. two states whose *render slice* is identical but whose frames differ. That is the
    test for "the observer draws something that is not in get_state()".

G2: bytes of the render slice vs bytes of the frame, plus the size of griddly's own get_state().
"""
import os, sys, json, pickle, hashlib, argparse
import numpy as np

import griddly
from griddly import GymWrapper, gd

GAMES = [
    ("sokoban", "Single-Player/GVGAI/sokoban.yaml"),
    ("clusters", "Single-Player/GVGAI/clusters.yaml"),
    ("zelda", "Single-Player/GVGAI/zelda.yaml"),
    ("butterflies", "Single-Player/GVGAI/butterflies.yaml"),
    ("labyrinth", "Single-Player/GVGAI/labyrinth.yaml"),
    ("GriddlyRTS", "RTS/GriddlyRTS.yaml"),
]


def gframe(env):
    """Global SPRITE_2D frame as (H, W, 3) uint8."""
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def render_key(state):
    """The slice of get_state() a tile renderer would read."""
    objs = sorted(
        (o["Name"], tuple(o["Location"]), o["Orientation"], o["PlayerId"], o["RenderTileId"])
        for o in state["Objects"]
    )
    return json.dumps([state["Grid"], objs], sort_keys=True)


def full_key(state):
    s = dict(state)
    s.pop("Hash", None)
    return json.dumps(s, sort_keys=True, default=str)


def sample_action(env, rng):
    sp = env.action_space
    try:
        n = sp.n
        return int(rng.integers(n))
    except AttributeError:
        return [int(rng.integers(k)) for k in sp.nvec]


def run(game_name, yaml_file, n_ckpt=8, steps_between=6, verbose=True):
    out = {"game": game_name}
    env = GymWrapper(yaml_file=yaml_file, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    env.seed(1234)
    env.reset(level_id=0)
    rng = np.random.default_rng(0)

    snaps = []          # (state, frame)
    seen = {}           # render_key -> (frame_md5, full_key)
    collisions = []

    def record(st, fr):
        k = render_key(st)
        m = hashlib.md5(np.ascontiguousarray(fr)).hexdigest()
        if k in seen and seen[k][0] != m:
            collisions.append((k[:120], seen[k][1][:200], full_key(st)[:200]))
        seen.setdefault(k, (m, full_key(st)))

    # --- rollout, snapshot ---
    double_render_maxdiff = 0
    for i in range(n_ckpt):
        for _ in range(steps_between):
            env.step(sample_action(env, rng))
        st = env.get_state()
        f1 = gframe(env)
        f2 = gframe(env)   # render twice, no step in between
        double_render_maxdiff = max(double_render_maxdiff,
                                    int(np.abs(f1.astype(int) - f2.astype(int)).max()))
        snaps.append((st, f1))
        record(st, f1)

    # --- churn: a completely different history on the same env object ---
    env.seed(9999)
    env.reset(level_id=min(1, env.level_count - 1))
    rng2 = np.random.default_rng(7)
    for _ in range(120):
        env.step(sample_action(env, rng2))
        record(env.get_state(), gframe(env))
    env.reset(level_id=0)
    for _ in range(40):
        env.step(sample_action(env, rng2))
        record(env.get_state(), gframe(env))

    # --- restore and diff ---
    maxdiff = 0
    ndiff_px_total = 0
    npx_total = 0
    per_ckpt = []
    for st, f0 in snaps:
        env2 = env.load_state(st)
        fr = gframe(env2)
        d = np.abs(fr.astype(int) - f0.astype(int))
        maxdiff = max(maxdiff, int(d.max()))
        ndiff = int((d.max(axis=2) > 0).sum())
        ndiff_px_total += ndiff
        npx_total += f0.shape[0] * f0.shape[1]
        per_ckpt.append((int(d.max()), ndiff))
        # also record the restored one, same render key must give the same frame
        record(env2.get_state(), fr)
        del env2

    out["g1_restore_max_abs_diff"] = maxdiff
    out["g1_restore_diff_px"] = ndiff_px_total
    out["g1_restore_total_px"] = npx_total
    out["g1_double_render_max_abs_diff"] = double_render_maxdiff
    out["g1_per_ckpt"] = per_ckpt
    out["g1_distinct_render_keys"] = len(seen)
    out["g1_render_key_collisions"] = len(collisions)
    out["g1_collision_examples"] = collisions[:2]

    # --- G2 accounting on the final snapshot ---
    st, f0 = snaps[-1]
    frame_bytes = int(f0.size)
    W, H = st["Grid"]["Width"], st["Grid"]["Height"]
    nobj = len(st["Objects"])
    names = sorted({o["Name"] for o in st["Objects"]})
    orients = sorted({o["Orientation"] for o in st["Objects"]})
    tileids = sorted({o["RenderTileId"] for o in st["Objects"]})
    pids = sorted({o["PlayerId"] for o in st["Objects"]})
    # object-list packing: type(1) + x(1) + y(1) + orient/pid/tile packed(1..2)
    obj_pack = nobj * 5
    # dense per-cell packing: one code per (cell, z-layer); count z layers as distinct
    # (name -> Z) is a constant of the game, so layers = max objects sharing a cell
    from collections import Counter
    cellcount = Counter(tuple(o["Location"]) for o in st["Objects"])
    nz = max(cellcount.values())
    dense = W * H * nz * 2   # 1 byte code + 1 byte (orientation|tileid|playerid)
    out["g2"] = dict(
        frame_hw=[int(f0.shape[0]), int(f0.shape[1])],
        frame_bytes=frame_bytes,
        n_objects=nobj, grid=[W, H], n_object_types=len(names),
        orientations=orients, render_tile_ids=tileids, player_ids=pids,
        max_objects_per_cell=nz,
        objlist_bytes=obj_pack, objlist_ratio=obj_pack / frame_bytes,
        dense_bytes=dense, dense_ratio=dense / frame_bytes,
        griddly_get_state_pickle_bytes=len(pickle.dumps(st)),
        griddly_get_state_json_bytes=len(json.dumps(st, default=str)),
        griddly_get_state_ratio=len(pickle.dumps(st)) / frame_bytes,
    )
    del env
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="all")
    a = ap.parse_args()
    sel = GAMES if a.games == "all" else [g for g in GAMES if g[0] in a.games.split(",")]
    results = []
    for name, yf in sel:
        try:
            r = run(name, yf)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"game": name, "error": str(e)}
        results.append(r)
        print(json.dumps(r, indent=1, default=str), flush=True)
    print("=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"{r['game']:14s} ERROR {r['error']}")
            continue
        g2 = r["g2"]
        print(f"{r['game']:14s} restore_maxdiff={r['g1_restore_max_abs_diff']} "
              f"diffpx={r['g1_restore_diff_px']}/{r['g1_restore_total_px']} "
              f"rerender_maxdiff={r['g1_double_render_max_abs_diff']} "
              f"keys={r['g1_distinct_render_keys']} collisions={r['g1_render_key_collisions']} "
              f"| frame={g2['frame_bytes']}B objlist={g2['objlist_bytes']}B "
              f"({g2['objlist_ratio']:.5f}x) dense={g2['dense_bytes']}B ({g2['dense_ratio']:.5f}x) "
              f"griddly_state={g2['griddly_get_state_pickle_bytes']}B ({g2['griddly_get_state_ratio']:.3f}x)")
