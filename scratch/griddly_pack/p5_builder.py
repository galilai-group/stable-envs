"""P5: the decoder an env can actually ship, built through ``reset(level_string=...)``.

p3/p4 settled the mechanism. ``GymWrapper.load_state`` returns a NEW wrapper, each owning a Vulkan
device, and holding them exhausts the driver (``vkGetDeviceQueue: Invalid device`` after ~260);
``reset(level_string=...)`` mutates the wrapper IN PLACE, accepts any grid size, and its grammar has a
``/`` separator for a cell holding more than one object (measured: ``A/h`` reads back as
``('avatar', 'hole')``). So one env builds the whole atlas and also serves as the parity reference.

What a level string CANNOT express is an object's Orientation or RenderTileId, which is exactly the
alphabet zelda and the Mini-Grid games need. This prints, per game, whether the alphabet is
expressible, and if it is, builds the complete table and reconstructs the harvested frames from it.
"""
import collections
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
#: bit k of the neighbour mask is offset k. The 4 orthogonal ones come first so that a game whose
#: tiling is WALL_16 (4 neighbours) keys on the low 4 bits.
OFFSETS = [(0, -1), (-1, 0), (1, 0), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
EMPTY_CHAR = "."
N_FRAMES = 200


def read_yaml(rel):
    with open(os.path.join(GAMES_DIR, rel)) as f:
        src = f.read()
    return src, pyyaml.load(src, Loader=pyyaml.FullLoader)


def yaml_at(rel, tile_size=None):
    src, g = read_yaml(rel)
    sp = (g["Environment"].get("Observers") or {}).get("Sprite2D") or {}
    T = sp.get("TileSize", 24)
    T = 24 if isinstance(T, list) else int(T)
    if tile_size is None or int(tile_size) == T:
        return src, T
    g["Environment"].setdefault("Observers", {}).setdefault("Sprite2D", {})["TileSize"] = int(tile_size)
    return pyyaml.dump(g, sort_keys=False), int(tile_size)


def make(rel, tile_size=None):
    src, T = yaml_at(rel, tile_size)
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
    """(x, y) -> sorted tuple of object names. ``_``-prefixed pseudo objects are dropped.

    Orientation / PlayerId / RenderTileId are DELIBERATELY not in the key: a level string cannot set
    them, so a game whose picture depends on them is refused rather than approximated. Whether this
    game is such a game is decided by the reconstruction below, not by reading the yaml.
    """
    m = collections.defaultdict(list)
    for o in state["Objects"]:
        if o["Name"].startswith("_"):
            continue
        m[tuple(int(v) for v in o["Location"])].append(o["Name"])
    return {k: tuple(sorted(v)) for k, v in m.items()}


def map_chars(g):
    out = {}
    for o in g.get("Objects", []):
        mc = o.get("MapCharacter")
        if mc is not None:
            out[o["Name"]] = str(mc)
    return out


def level_string(grid, W, H, chars):
    """``grid`` = {(x, y): code} -> a griddly level string, space separated, ``/`` inside a cell."""
    rows = []
    for y in range(H):
        row = []
        for x in range(W):
            code = grid.get((x, y), ())
            row.append("/".join(chars[n] for n in code) if code else EMPTY_CHAR)
        rows.append(" ".join(row))
    return "\n".join(rows)


class Canvas:
    """Renders many (centre, neighbourhood) configurations per griddly frame, in ONE wrapper."""

    def __init__(self, env, T, chars, avatar, k=10):
        self.env, self.T, self.chars, self.avatar = env, T, chars, avatar
        self.k = k
        self.W, self.H = 3 * k + 2, 3 * k
        self.slots = [(3 * bi + 1, 3 * bj + 1) for bj in range(k) for bi in range(k)]
        self.loads = 0

    def render(self, configs):
        out = []
        for i in range(0, len(configs), len(self.slots)):
            chunk = configs[i:i + len(self.slots)]
            grid = {}
            for (cx, cy), (centre, nbs) in zip(self.slots, chunk):
                if centre:
                    grid[(cx, cy)] = centre
                for kk, code in nbs.items():
                    dx, dy = OFFSETS[kk]
                    if code:
                        grid[(cx + dx, cy + dy)] = code
            if self.avatar and not any(self.avatar in c for c in grid.values()):
                grid[(self.W - 1, 0)] = (self.avatar,)
            self.env.reset(level_string=level_string(grid, self.W, self.H, self.chars))
            self.loads += 1
            fr = frame(self.env)
            for (cx, cy), _ in zip(self.slots, chunk):
                out.append(fr[cy * self.T:(cy + 1) * self.T,
                              cx * self.T:(cx + 1) * self.T].tobytes())
        return out


def run(name, rel):
    t0 = time.time()
    res = {"game": name, "yaml": rel}
    src, g = read_yaml(rel)
    chars = map_chars(g)
    avatar = ((g["Environment"].get("Player") or {}).get("AvatarObject"))
    env, T = make(rel)
    env.seed(0)
    env.reset(level_id=0)
    fr0 = frame(env)
    GW, GH = fr0.shape[1] // T, fr0.shape[0] // T
    res.update(tile_size=T, grid=[GW, GH], avatar=avatar, n_map_chars=len(chars))

    # ---- harvest level 0
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
    singles = tuple((n,) for n in sorted(chars))
    harvested = {c for cm, _ in recs for c in cm.values()}
    alphabet = [()] + sorted(set(singles) | harvested, key=lambda c: (len(c), c))
    alphabet = [c for c in alphabet if c]
    alphabet = [()] + alphabet
    missing = sorted({n for c in alphabet for n in c} - set(chars))
    res.update(n_frames=len(recs), n_codes=len(alphabet),
               alphabet=[list(c) for c in alphabet], no_map_character=missing)
    if missing:
        res["verdict"] = "REFUSED: an object in the alphabet has no MapCharacter"
        res["seconds"] = round(time.time() - t0, 1)
        return res

    cid = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet)
    canvas = Canvas(env, T, chars, avatar)

    # ---- does every code round-trip through a level string?
    grid = {}
    for i, c in enumerate(alphabet):
        if c:
            grid[(3 * (i % canvas.k) + 1, 3 * (i // canvas.k) + 1)] = c
    env.reset(level_string=level_string(grid, canvas.W, canvas.H, chars))
    back = codes_of(env.get_state())
    bad = {str(k): [list(v), list(back.get(k, ()))] for k, v in grid.items() if back.get(k) != v}
    res["level_string_roundtrip"] = dict(placed=len(grid), wrong=len(bad), examples=list(bad.items())[:4])
    if bad:
        res["verdict"] = "REFUSED: a cell code does not round-trip through a level string"
        res["seconds"] = round(time.time() - t0, 1)
        return res

    # ---- which codes are distinguishable AS a neighbour, and which positions matter
    base = dict(zip(alphabet, canvas.render([(c, {}) for c in alphabet])))
    sig_cfg = [(c, {k: d for k in range(8)}) for d in alphabet for c in alphabet]
    sig_tiles = canvas.render(sig_cfg)
    sig = collections.defaultdict(list)
    for (d, c), t in zip([(d, c) for d in alphabet for c in alphabet], sig_tiles):
        sig[d].append(t)
    groups = collections.defaultdict(list)
    for d in alphabet:
        groups[tuple(sig[d])].append(d)
    classes = sorted(groups.values(), key=lambda gg: (0 if () in gg else 1, str(gg[0])))
    cls_of = {d: i for i, gg in enumerate(classes) for d in gg}
    m = len(classes)
    rep = [gg[0] for gg in classes]
    res["neighbour_classes"] = [[list(d) for d in gg] for gg in classes]
    res["m"] = m

    pos_cfg = [(c, {k: r}) for k in range(8) for r in rep[1:] for c in alphabet]
    pos_key = [(k, c) for k in range(8) for r in rep[1:] for c in alphabet]
    pos_tiles = canvas.render(pos_cfg)
    live = sorted({k for (k, c), t in zip(pos_key, pos_tiles) if t != base[c]})
    K = len(live)
    res.update(live_positions=[list(OFFSETS[k]) for k in live], n_live=K,
               table_entries=n * m ** K)
    if n * m ** K > 200000:
        res["verdict"] = f"REFUSED: table would need {n * m ** K} entries"
        res["seconds"] = round(time.time() - t0, 1)
        return res

    # ---- enumerate
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
    res.update(atlas_tiles=len(uniq), atlas_bytes=int(atlas.nbytes),
               table_shape=list(table.shape), canvas_loads=canvas.loads,
               enumeration_configs=len(cfgs))

    # ---- reconstruct the harvested frames
    grids = np.full((len(recs), GH, GW), -1, np.int32)
    for i, (cm, _) in enumerate(recs):
        for y in range(GH):
            for x in range(GW):
                grids[i, y, x] = cid.get(cm.get((x, y), ()), -1)
    if (grids < 0).any():
        res["verdict"] = "REFUSED: a harvested cell code is outside the alphabet"
        res["seconds"] = round(time.time() - t0, 1)
        return res
    cl = np.array([cls_of[c] for c in alphabet], np.int32)
    ref = np.stack([r[1] for r in recs])
    for oob in range(m):
        pad = np.full((len(recs), GH + 2, GW + 2), oob, np.int32)
        pad[:, 1:-1, 1:-1] = cl[grids]
        mask = np.zeros((len(recs), GH, GW), np.int64)
        for bit, k in enumerate(live):
            dx, dy = OFFSETS[k]
            mask += pad[:, 1 + dy:1 + dy + GH, 1 + dx:1 + dx + GW].astype(np.int64) * (m ** bit)
        idx = table[grids, mask]
        got = atlas[idx].transpose(0, 1, 3, 2, 4, 5).reshape(len(recs), GH * T, GW * T, 3)
        d = np.abs(got.astype(np.int32) - ref.astype(np.int32))
        res[f"recon_oob{oob}"] = dict(max_abs=int(d.max()),
                                      exact_frames=int((d.reshape(len(recs), -1).max(1) == 0).sum()))
        if d.max() == 0:
            res["oob_class"] = oob
            break
    res["verdict"] = ("OK" if res.get("oob_class") is not None
                      else "REFUSED: the (code, neighbour-class) table does not reproduce the frames")
    res["seconds"] = round(time.time() - t0, 1)
    return res


if __name__ == "__main__":
    out = {}
    for name in (sys.argv[1:] or list(GAMES)):
        try:
            out[name] = run(name, GAMES[name])
        except Exception as e:                                      # noqa: BLE001
            import traceback
            traceback.print_exc()
            out[name] = {"game": name, "error": f"{type(e).__name__}: {e}"}
        r = out[name]
        print(f"{name:14s} {r.get('verdict', r.get('error'))!s:70.70s} "
              f"grid={r.get('grid')} n_codes={r.get('n_codes')} m={r.get('m')} "
              f"live={r.get('n_live')} atlas={r.get('atlas_tiles')} loads={r.get('canvas_loads')} "
              f"{r.get('seconds')}s", flush=True)
    with open(os.path.join(OUT, "p5_builder.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "p5_builder.json"), flush=True)
    sys.stdout.flush()
    os._exit(0)
