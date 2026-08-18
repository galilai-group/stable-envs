"""P1: the packaging decisions Griddly needs, all measured in one run.

docs/research/griddly.md settled the RESEARCH questions (the atlas is finite, the index is a function
of a 3x3 stencil, ``tilemap.blit`` reproduces the frame bit-exactly). What a shipped ``StableEnv``
additionally needs is a decoder that is COMPLETE rather than harvested: the conformance suite proves a
state dimension is live by substituting a donor cell code into a real state, which manufactures
neighbourhoods no rollout ever produced. A table built from rollouts would miss them.

So this asks whether the 3x3 stencil FACTORS: is the tile a function of (own cell code, a 1-bit
membership mask over the 8 neighbours)? If it is, the table is n_codes x 2^k and can be ENUMERATED by
asking griddly to draw synthetic grids, which is what an env can ship.

Per game:
  0. TileSize is patchable in the yaml and the frame shape scales with it.
  1. rollout harvest: grid size per level, the cell-code alphabet, whether any orientation is non-NONE.
  2. ``load_state`` contract: can arbitrary objects be placed, and does the state round-trip?
  3. which of the 8 neighbour positions matter, and which codes matter AS a neighbour.
  4. enumerate the table and RECONSTRUCT the harvested frames from it. max_abs must be 0.
"""
import collections
import json
import os
import sys

import numpy as np
import yaml as pyyaml

import griddly
from griddly import GymWrapper, gd

OUT = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")

GAMES = {
    "sokoban": "Single-Player/GVGAI/sokoban.yaml",
    "sokoban2": "Single-Player/GVGAI/sokoban2.yaml",
    "labyrinth": "Single-Player/GVGAI/labyrinth.yaml",
    "butterflies": "Single-Player/GVGAI/butterflies.yaml",
    "zelda": "Single-Player/GVGAI/zelda.yaml",
    "bait": "Single-Player/GVGAI/bait.yaml",
    "clusters": "Single-Player/GVGAI/clusters.yaml",
    "cookmepasta": "Single-Player/GVGAI/cookmepasta.yaml",
    "zenpuzzle": "Single-Player/GVGAI/zenpuzzle.yaml",
    "spider-nest": "Single-Player/GVGAI/spider-nest.yaml",
    "drunkdwarf": "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml",
    "doggo": "Single-Player/Mini-Grid/minigrid-doggo.yaml",
    "eyeball": "Single-Player/Mini-Grid/minigrid-eyeball.yaml",
    "spiders": "Single-Player/Mini-Grid/minigrid-spiders.yaml",
    "foragers": "Multi-Agent/foragers.yaml",
}
N_FRAMES = 240
#: the 8 neighbour offsets, in a fixed order; bit k of the mask is offset k
OFFSETS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def patched(rel, tile_size=None):
    with open(os.path.join(GAMES_DIR, rel)) as f:
        src = f.read()
    g = pyyaml.load(src, Loader=pyyaml.FullLoader)
    sp = (g["Environment"].get("Observers") or {}).get("Sprite2D") or {}
    T = sp.get("TileSize", 24)
    T = 24 if isinstance(T, list) else int(T)
    if tile_size is None or int(tile_size) == T:
        return src, T
    g["Environment"].setdefault("Observers", {}).setdefault("Sprite2D", {})["TileSize"] = int(tile_size)
    return pyyaml.dump(g, sort_keys=False), int(tile_size)


def make(rel, tile_size=None):
    src, T = patched(rel, tile_size)
    env = GymWrapper(yaml_string=src, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    return env, T


def frame(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def cell_codes(state):
    """(x, y) -> the sorted tuple of (Name, Orientation, PlayerId, RenderTileId) standing there."""
    m = collections.defaultdict(list)
    for o in state["Objects"]:
        m[tuple(int(v) for v in o["Location"])].append(
            (o["Name"], o["Orientation"], int(o["PlayerId"]), int(o["RenderTileId"])))
    return {k: tuple(sorted(v)) for k, v in m.items()}


def tiles_of(fr, T):
    h, w = fr.shape[0] // T, fr.shape[1] // T
    return np.array([[fr[j * T:(j + 1) * T, i * T:(i + 1) * T].tobytes() for i in range(w)]
                     for j in range(h)], dtype=object)


def build_state(template, objects):
    """A griddly state holding exactly ``objects`` = [(name, (x,y), orientation, player, tile_id)]."""
    proto = {o["Name"]: o for o in template["Objects"]}
    out = []
    for name, (x, y), ori, pid, rt in objects:
        base = proto.get(name)
        o = {"Name": name, "Location": [int(x), int(y)], "Orientation": ori,
             "PlayerId": int(pid), "RenderTileId": int(rt),
             "Variables": dict(base["Variables"]) if base and "Variables" in base else {}}
        out.append(o)
    st = dict(template)
    st["Objects"] = out
    return st


def run(name, rel):
    res = {"game": name, "yaml": rel}
    env, T = make(rel)
    res["tile_size"] = T
    res["levels"] = int(env.level_count)
    res["players"] = int(env.player_count)

    # ---- 0. TileSize patch scales the frame
    env.seed(0)
    env.reset(level_id=0)
    fr0 = frame(env)
    res["level0_shape"] = list(fr0.shape)
    res["grid"] = [fr0.shape[1] // T, fr0.shape[0] // T]
    shapes = {}
    for t in (8, 16, 32):
        e2, _ = make(rel, t)
        e2.seed(0)
        e2.reset(level_id=0)
        shapes[t] = list(frame(e2).shape)
        del e2
    res["shapes_by_tile_size"] = shapes
    res["tile_size_scales"] = all(
        shapes[t] == [res["grid"][1] * t, res["grid"][0] * t, 3] for t in shapes)

    # ---- 1. rollout harvest at level 0 only (a class pins one level: state_dim is fixed)
    recs, ep = [], 0
    while len(recs) < N_FRAMES and ep < 200:
        env.seed(ep)
        env.reset(level_id=0)
        rng = np.random.default_rng(4000 + ep)
        for _ in range(60):
            st = env.get_state()
            fr = frame(env)
            assert len(np.unique(fr)) > 1, f"{name}: uniform frame"
            recs.append((cell_codes(st), fr))
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done) or len(recs) >= N_FRAMES:
                break
        ep += 1
    GW, GH = res["grid"]
    alphabet = sorted({() } | {c for cm, _ in recs for c in cm.values()}, key=str)
    res["n_frames"] = len(recs)
    res["alphabet"] = [str(c) for c in alphabet]
    res["n_codes"] = len(alphabet)
    res["orientations"] = sorted({o[1] for c in alphabet for o in c})
    res["render_tile_ids"] = sorted({o[3] for c in alphabet for o in c})
    res["stacked_codes"] = [str(c) for c in alphabet if len(c) > 1]

    # ---- 2. load_state
    template = env.get_state()
    env.seed(0)
    env.reset(level_id=0)
    want = cell_codes(env.get_state())
    objs = [(o[0], loc, o[1], o[2], o[3]) for loc, code in want.items() for o in code]
    try:
        env.load_state(build_state(template, objs))
        got = cell_codes(env.get_state())
        res["load_state"] = dict(
            ok=True, codes_match=(got == want),
            n_differ=sum(1 for k in set(got) | set(want) if got.get(k) != want.get(k)))
    except Exception as e:                                          # noqa: BLE001
        res["load_state"] = dict(ok=False, error=f"{type(e).__name__}: {e}")
        del env
        return res

    # placing an ARBITRARY object set (not one the game produced) — the enumeration needs this
    probe = [(alphabet[i][0][0], (1 + 3 * i, 1), "NONE", alphabet[i][0][2], alphabet[i][0][3])
             for i in range(min(len(alphabet), 3)) if alphabet[i]]
    try:
        env.load_state(build_state(template, probe))
        got = cell_codes(env.get_state())
        res["load_state_arbitrary"] = dict(ok=True, n_cells=len(got),
                                           placed=len(probe), matches=len(got) == len(probe))
    except Exception as e:                                          # noqa: BLE001
        res["load_state_arbitrary"] = dict(ok=False, error=f"{type(e).__name__}: {e}")

    # ---- 3/4 need a bigger canvas than level 0; try a synthetic level string of the right size
    del env
    return res


if __name__ == "__main__":
    out = {}
    only = sys.argv[1:] or list(GAMES)
    for name in only:
        try:
            out[name] = run(name, GAMES[name])
        except Exception as e:                                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            out[name] = {"game": name, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(out[name], default=str)[:2000], flush=True)
    with open(os.path.join(OUT, "p1_design.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "p1_design.json"), flush=True)
    sys.stdout.flush()
    os._exit(0)          # griddly segfaults in GymWrapper.__del__ at interpreter exit
