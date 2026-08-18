"""Probe 4: the decisive G1 test.

Instead of relying on griddly's load_state (which does NOT round-trip Orientation -- see
probe_03), this accumulates (render_key -> frame md5) over many independent rollouts, levels
and seeds, and asks two questions:

  A. COLLISIONS. Two states that agree on the render slice (object name / location /
     orientation / playerId / renderTileId, plus grid size) but whose frames differ.
     Threshold: 0.  Any collision means the observer reads something not in get_state().

  B. WITNESSES. Of the render keys seen more than once, how many were reached with a
     DIFFERENT full state (different GameTicks / GlobalVariables / DelayedActions / object
     Variables / different episode / different history)?  Those are the pairs that make the
     0-collision number mean something rather than being vacuous.

Also reports, per game, whether object Variables (e.g. zelda's has_key) need to be in the key:
we key WITHOUT Variables and count collisions, then WITH RenderTileId only.
"""
import json, hashlib, argparse, collections
import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES = [
    ("sokoban", "Single-Player/GVGAI/sokoban.yaml"),
    ("clusters", "Single-Player/GVGAI/clusters.yaml"),
    ("zelda", "Single-Player/GVGAI/zelda.yaml"),
    ("butterflies", "Single-Player/GVGAI/butterflies.yaml"),
    ("labyrinth", "Single-Player/GVGAI/labyrinth.yaml"),
    ("bait", "Single-Player/GVGAI/bait.yaml"),
    ("zenpuzzle", "Single-Player/GVGAI/zenpuzzle.yaml"),
    ("cookmepasta", "Single-Player/GVGAI/cookmepasta.yaml"),
    ("spider-nest", "Single-Player/GVGAI/spider-nest.yaml"),
    ("drunkdwarf", "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml"),
    ("foragers", "Multi-Agent/foragers.yaml"),
    ("GriddlyRTS", "RTS/GriddlyRTS.yaml"),
    ("kill-the-king", "RTS/Stratega/kill-the-king.yaml"),
]


def gframe(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    # MultiAgentActionSpace
    out = []
    for s in sp:
        if hasattr(s, "nvec"):
            out.append([int(rng.integers(k)) for k in s.nvec])
        else:
            out.append(int(rng.integers(s.n)))
    return out


def key_render(st):
    objs = sorted(
        (o["Name"], tuple(o["Location"]), o["Orientation"], o["PlayerId"], o["RenderTileId"])
        for o in st["Objects"]
    )
    return json.dumps([st["Grid"]["Width"], st["Grid"]["Height"], objs])


def key_novars(st):
    """same but WITHOUT RenderTileId -- to show RenderTileId is load-bearing"""
    objs = sorted(
        (o["Name"], tuple(o["Location"]), o["Orientation"], o["PlayerId"])
        for o in st["Objects"]
    )
    return json.dumps([st["Grid"]["Width"], st["Grid"]["Height"], objs])


def key_full(st):
    s = {k: v for k, v in st.items() if k != "Hash"}
    return json.dumps(s, sort_keys=True, default=str)


def run(name, yaml_file, episodes=12, ep_len=60):
    env = GymWrapper(yaml_file=yaml_file, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    nlev = env.level_count
    seen = {}            # key_render -> (md5, key_full, frame)
    seen_nv = {}         # key_novars -> md5
    collisions, coll_nv = [], []
    repeats = 0
    witnesses = 0        # repeat with a DIFFERENT full state
    n_frames = 0
    maxdiff_on_repeat = 0

    for ep in range(episodes):
        lev = ep % nlev
        env.seed(1000 + ep)
        env.reset(level_id=lev)
        rng = np.random.default_rng(ep * 31 + 5)
        for t in range(ep_len):
            st = env.get_state()
            fr = gframe(env)
            n_frames += 1
            k, kf = key_render(st), key_full(st)
            m = hashlib.md5(np.ascontiguousarray(fr)).hexdigest()
            if k in seen:
                repeats += 1
                m0, kf0, fr0 = seen[k]
                if kf0 != kf:
                    witnesses += 1
                if m0 != m:
                    d = int(np.abs(fr.astype(int) - fr0.astype(int)).max())
                    maxdiff_on_repeat = max(maxdiff_on_repeat, d)
                    if len(collisions) < 3:
                        collisions.append((d, kf0[:300], kf[:300]))
            else:
                seen[k] = (m, kf, fr)
            knv = key_novars(st)
            if knv in seen_nv:
                if seen_nv[knv] != m and len(coll_nv) < 3:
                    coll_nv.append(knv[:120])
            else:
                seen_nv[knv] = m
            try:
                obs, rew, done, info = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done):
                break
    res = dict(game=name, frames=n_frames, distinct_render_keys=len(seen),
               repeats=repeats, witnesses_diff_full_state=witnesses,
               collisions=len(collisions), max_abs_diff_on_repeat=maxdiff_on_repeat,
               collisions_without_RenderTileId=len(coll_nv))
    del env
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--games", default="all")
    a = ap.parse_args()
    sel = GAMES if a.games == "all" else [g for g in GAMES if g[0] in a.games.split(",")]
    rows = []
    for n, y in sel:
        try:
            r = run(n, y)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = dict(game=n, error=str(e))
        rows.append(r); print(json.dumps(r), flush=True)
    print("=== SUMMARY (G1) ===")
    tot_f = tot_rep = tot_wit = tot_coll = 0
    for r in rows:
        if "error" in r:
            print(f"{r['game']:14s} ERROR {r['error'][:80]}"); continue
        print(f"{r['game']:14s} frames={r['frames']:5d} keys={r['distinct_render_keys']:5d} "
              f"repeats={r['repeats']:5d} witnesses={r['witnesses_diff_full_state']:5d} "
              f"COLLISIONS={r['collisions']} maxdiff={r['max_abs_diff_on_repeat']} "
              f"| coll_without_RenderTileId={r['collisions_without_RenderTileId']}")
        tot_f += r["frames"]; tot_rep += r["repeats"]
        tot_wit += r["witnesses_diff_full_state"]; tot_coll += r["collisions"]
    print(f"TOTAL frames={tot_f} repeats={tot_rep} witnesses={tot_wit} collisions={tot_coll}")
