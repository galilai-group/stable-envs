"""P2: build the COMPLETE decoder by asking griddly, and reconstruct real frames from it.

The hypothesis this settles: griddly's tile is a function of (own cell code, a small class label for
each of the 8 neighbours), so the decoder is a table of size ``n_codes x m^8`` that can be ENUMERATED
rather than harvested. If it is, an env can ship it and the conformance suite's donor-substitution
probe cannot fall off the end of it.

Procedure, per game, entirely through griddly's own renderer:

  1. rollout at level 0 -> the cell-code alphabet (``_``-prefixed pseudo objects filtered out) and a
     set of real (codes, frame) pairs to check against.
  2. ``load_state`` an arbitrary object set and read it back: does griddly honour Name / Orientation /
     PlayerId / RenderTileId at an arbitrary location?
  3. probe every (centre code, neighbour code) pair on a 3-spaced lattice: which of the 8 neighbour
     positions can change a tile at all, and which codes are distinguishable AS a neighbour.
  4. enumerate the table over (centre, class^live_positions), both conventions for out-of-grid.
  5. RECONSTRUCT every harvested frame from the table. max_abs must be 0.
"""
import collections
import copy
import json
import os
import sys
import time

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
N_FRAMES = 200
OFFSETS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def yaml_src(rel, tile_size=None):
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
    src, T = yaml_src(rel, tile_size)
    return GymWrapper(yaml_string=src, level=0,
                      global_observer_type=gd.ObserverType.SPRITE_2D,
                      player_observer_type=gd.ObserverType.SPRITE_2D), T


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


def codes_of(state):
    """(x, y) -> sorted tuple of (Name, Orientation, PlayerId, RenderTileId); ``_`` names dropped."""
    m = collections.defaultdict(list)
    for o in state["Objects"]:
        if o["Name"].startswith("_"):
            continue
        m[tuple(int(v) for v in o["Location"])].append(
            (o["Name"], o["Orientation"], int(o["PlayerId"]), int(o["RenderTileId"])))
    return {k: tuple(sorted(v)) for k, v in m.items()}


def state_with(template, placed):
    """``placed`` = {(x, y): code}; a griddly state holding exactly those objects.

    The ``_empty`` / ``_boundary`` pseudo objects griddly keeps in its own state are carried over
    verbatim: measured in p3, a state built without them raises ``IndexError: unordered_map::at``
    inside ``game.load_state``. Every object also needs its ``Variables`` dict — a rebuilt object
    without one raises ``KeyError: 'Variables'``.
    """
    proto = {o["Name"]: o for o in template["Objects"]}
    objs = [copy.deepcopy(o) for o in template["Objects"] if o["Name"].startswith("_")]
    for (x, y), code in placed.items():
        for name, ori, pid, rt in code:
            base = proto.get(name)
            objs.append({"Name": name, "Location": [int(x), int(y)], "Orientation": ori,
                         "PlayerId": int(pid), "RenderTileId": int(rt),
                         "Variables": copy.deepcopy(base["Variables"]) if base else {}})
    st = copy.deepcopy({k: v for k, v in template.items() if k != "Objects"})
    st["Objects"] = objs
    return st


def tile_at(fr, x, y, T):
    return fr[y * T:(y + 1) * T, x * T:(x + 1) * T].tobytes()


class Canvas:
    """Renders many (centre, neighbourhood) configurations per griddly frame.

    Configurations are laid out on a lattice of 3x3 blocks, so each centre's whole 8-neighbourhood sits
    inside its own block and no two configurations can see each other.
    """

    #: every wrapper griddly.load_state hands back is kept alive. Dropping one runs
    #: ``GymWrapper.__del__ -> game.release()``, which is the call that segfaults.
    KEEP = []

    def __init__(self, env, template, GW, GH, T):
        self.env, self.template, self.GW, self.GH, self.T = env, template, GW, GH, T
        self.slots = [(3 * bi + 1, 3 * bj + 1)
                      for bj in range(GH // 3) for bi in range(GW // 3)]
        if not self.slots:
            raise RuntimeError(f"grid {GW}x{GH} is too small to hold one 3x3 probe block")

    def render(self, configs):
        """``configs`` = [(centre_code, {offset_index: neighbour_code})] -> list of tile bytes."""
        out = []
        for i in range(0, len(configs), len(self.slots)):
            chunk = configs[i:i + len(self.slots)]
            placed = {}
            for (cx, cy), (centre, nbs) in zip(self.slots, chunk):
                if centre:
                    placed[(cx, cy)] = centre
                for k, code in nbs.items():
                    dx, dy = OFFSETS[k]
                    if code:
                        placed[(cx + dx, cy + dy)] = code
            # load_state RETURNS A NEW WRAPPER and leaves this one untouched
            e = self.env.load_state(state_with(self.template, placed))
            Canvas.KEEP.append(e)
            fr = frame(e)
            for (cx, cy), _ in zip(self.slots, chunk):
                out.append(tile_at(fr, cx, cy, self.T))
        return out


def run(name, rel):
    t0 = time.time()
    res = {"game": name, "yaml": rel}
    env, T = make(rel)
    env.seed(0)
    env.reset(level_id=0)
    fr0 = frame(env)
    GW, GH = fr0.shape[1] // T, fr0.shape[0] // T
    res.update(tile_size=T, grid=[GW, GH])

    # ---- 1. harvest
    recs, ep = [], 0
    while len(recs) < N_FRAMES and ep < 200:
        env.seed(ep)
        env.reset(level_id=0)
        rng = np.random.default_rng(4000 + ep)
        for _ in range(60):
            recs.append((codes_of(env.get_state()), frame(env)))
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done) or len(recs) >= N_FRAMES:
                break
        ep += 1
    alphabet = sorted({()} | {c for cm, _ in recs for c in cm.values()}, key=str)
    cid = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet)
    res.update(n_frames=len(recs), n_codes=n, alphabet=[str(c) for c in alphabet])

    template = env.get_state()

    # ---- 2. does load_state honour an arbitrary placement?
    want = {(1 + 3 * (i % max(GW // 3, 1)), 1 + 3 * (i // max(GW // 3, 1))): c
            for i, c in enumerate(alphabet) if c}
    want = {k: v for k, v in want.items() if k[0] < GW - 1 and k[1] < GH - 1}
    e2 = env.load_state(state_with(template, want))
    Canvas.KEEP.append(e2)
    got = codes_of(e2.get_state())
    res["load_state_arbitrary"] = dict(
        placed=len(want), read_back=len(got),
        exact=all(got.get(k) == v for k, v in want.items()),
        extra=sorted(str(k) for k in set(got) - set(want))[:6])

    canvas = Canvas(env, template, GW, GH, T)

    # ---- 3. which positions matter, and which codes are distinguishable as a neighbour
    empty_base = canvas.render([(c, {}) for c in alphabet])
    sig_cfg, sig_key = [], []
    for d in alphabet:
        for c in alphabet:
            sig_cfg.append((c, {k: d for k in range(8)}))
            sig_key.append((d, c))
    sig_tiles = canvas.render(sig_cfg)
    sig = collections.defaultdict(list)
    for (d, c), t in zip(sig_key, sig_tiles):
        sig[d].append(t)
    groups = collections.defaultdict(list)
    for d in alphabet:
        groups[tuple(sig[d])].append(d)
    classes = sorted(groups.values(), key=lambda g: str(g[0]))
    # the class holding the empty code is class 0, so an all-empty neighbourhood keys to mask 0
    classes.sort(key=lambda g: 0 if () in g else 1)
    cls_of = {d: i for i, g in enumerate(classes) for d in g}
    m = len(classes)
    res["neighbour_classes"] = [[str(d) for d in g] for g in classes]
    res["m"] = m

    pos_cfg, pos_key = [], []
    rep = [g[0] for g in classes]
    for k in range(8):
        for r in rep[1:]:
            for c in alphabet:
                pos_cfg.append((c, {k: r}))
                pos_key.append((k, c))
    pos_tiles = canvas.render(pos_cfg)
    base_of = dict(zip(alphabet, empty_base))
    live = sorted({k for (k, c), t in zip(pos_key, pos_tiles) if t != base_of[c]})
    res["live_positions"] = [list(OFFSETS[k]) for k in live]
    res["n_live"] = len(live)

    # ---- 4. enumerate the table
    K = len(live)
    if m ** K * n > 60000:
        res["too_big"] = dict(m=m, K=K, n=n, entries=m ** K * n)
        res["seconds"] = round(time.time() - t0, 1)
        return res
    cfgs, keys = [], []
    for c in alphabet:
        for mask in range(m ** K):
            nbs, v = {}, mask
            for k in live:
                nbs[k] = rep[v % m]
                v //= m
            cfgs.append((c, nbs))
            keys.append((cid[c], mask))
    tiles = canvas.render(cfgs)
    uniq = sorted(set(tiles))
    tidx = {t: i for i, t in enumerate(uniq)}
    table = np.zeros((n, m ** K), np.int32)
    for (ci, mask), t in zip(keys, tiles):
        table[ci, mask] = tidx[t]
    atlas = np.stack([np.frombuffer(t, np.uint8).reshape(T, T, 3) for t in uniq])
    res.update(table_shape=list(table.shape), atlas_tiles=len(uniq),
               atlas_bytes=int(atlas.nbytes), n_renders_enumeration=len(cfgs))

    # ---- 5. reconstruct the harvested frames
    def rebuild(oob_cls):
        grids = np.zeros((len(recs), GH, GW), np.int32)
        for i, (cm, _) in enumerate(recs):
            for y in range(GH):
                for x in range(GW):
                    grids[i, y, x] = cid.get(cm.get((x, y), ()), -1)
        if (grids < 0).any():
            return None, "a harvested cell code is outside the alphabet"
        cl = np.array([cls_of[c] for c in alphabet], np.int32)
        pad = np.full((len(recs), GH + 2, GW + 2), oob_cls, np.int32)
        pad[:, 1:-1, 1:-1] = cl[grids]
        mask = np.zeros((len(recs), GH, GW), np.int64)
        for bit, k in enumerate(live):
            dx, dy = OFFSETS[k]
            mask += pad[:, 1 + dy:1 + dy + GH, 1 + dx:1 + dx + GW].astype(np.int64) * (m ** bit)
        idx = table[grids, mask]
        got = np.stack([np.concatenate(
            [np.concatenate([atlas[idx[i, y, x]] for x in range(GW)], 1) for y in range(GH)], 0)
            for i in range(len(recs))])
        ref = np.stack([r[1] for r in recs])
        d = np.abs(got.astype(np.int32) - ref.astype(np.int32))
        return dict(frames=len(recs), max_abs=int(d.max()),
                    exact_frames=int((d.reshape(len(recs), -1).max(1) == 0).sum()),
                    pixel_exact_frac=float((d == 0).mean())), None

    best = None
    for oob in range(m):
        r, err = rebuild(oob)
        if err:
            res["recon_error"] = err
            break
        res[f"recon_oob_class_{oob}"] = r
        if best is None or r["max_abs"] < best[1]["max_abs"]:
            best = (oob, r)
    if best:
        res["oob_class"] = best[0]
        res["recon"] = best[1]
    res["seconds"] = round(time.time() - t0, 1)
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
        print(json.dumps({k: v for k, v in out[name].items() if k != "alphabet"},
                         default=str)[:1400], flush=True)
    with open(os.path.join(OUT, "p2_table.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "p2_table.json"), flush=True)
    sys.stdout.flush()
    os._exit(0)
