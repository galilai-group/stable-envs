"""Probe 12: does the decoder stay constant under ARBITRARY levels, not just the shipped ones?

G3's threshold is "identical across episodes". The shipped games have 3-10 hand-written levels;
a data-generation pipeline would call reset(level_string=...) with generated maps. This builds
200 random maps per game from the game's own MapCharacters and re-runs both the tile-alphabet
count and the 3x3-stencil determinism test.
"""
import os, re, json, hashlib, collections
import numpy as np
import yaml as pyyaml
import griddly
from griddly import GymWrapper, gd

GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
# NOTE: griddly's level parser reads a DIGIT after a character as a player id, so games whose
# MapCharacters include digits (zelda's spider "3", clusters' "1"/"2"/"3") cannot be driven with
# a naive random map string. Restricted here to games with letter-only MapCharacters.
GAMES = [("sokoban", "Single-Player/GVGAI/sokoban.yaml", "wbhA."),
         ("labyrinth", "Single-Player/GVGAI/labyrinth.yaml", "wxtA."),
         ("drunkdwarf", "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml", "WgDdctbkA.")]


def blocks(fr, T):
    H, W, _ = fr.shape
    return fr.reshape(H // T, T, W // T, T, 3).transpose(0, 2, 1, 3, 4).reshape(-1, T, T, 3)


def cellmap(st):
    m = collections.defaultdict(list)
    for o in st["Objects"]:
        m[tuple(o["Location"])].append((o["Name"], o["Orientation"], o["PlayerId"], o["RenderTileId"]))
    return {k: tuple(sorted(v)) for k, v in m.items()}


def random_level(rng, chars, w, h, avatar="A"):
    body = [["w"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            body[y][x] = chars[rng.integers(len(chars))]
    # exactly one avatar
    for y in range(h):
        for x in range(w):
            if body[y][x] == avatar:
                body[y][x] = "."
    body[rng.integers(1, h - 1)][rng.integers(1, w - 1)] = avatar
    return "\n".join("".join(r) for r in body)


for name, yf, chars in GAMES:
    src = open(os.path.join(GAMES_DIR, yf)).read()
    env = GymWrapper(yaml_file=yf, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    T = env.get_tile_size(0)[0]
    rng = np.random.default_rng(0)
    W0, H0 = env.grid_width, env.grid_height
    print(f"  [{name}] native grid {W0}x{H0}; testing SAME-SIZE random maps first")
    tiles = {}
    stencil = {}
    coll = 0
    growth = []
    nmaps = 0
    for i in range(200):
        w, h = (W0, H0) if i < 150 else (int(rng.integers(6, 16)), int(rng.integers(6, 14)))
        lvl = random_level(rng, [c for c in chars if c != "A"], w, h)
        if i == 150:
            print(f"  [{name}] first 150 same-size maps OK: tiles={len(tiles)} stencil_collisions={coll}", flush=True)
        try:
            env.reset(level_string=lvl)
        except Exception as ex:
            print(f"  [{name}] reset(level_string) failed at i={i} ({w}x{h}): {ex}", flush=True)
            continue
        nmaps += 1
        st = env.get_state()
        fr = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
        assert fr.shape[0] == st["Grid"]["Height"] * T and fr.shape[1] == st["Grid"]["Width"] * T
        for b in blocks(fr, T):
            tiles[hashlib.md5(np.ascontiguousarray(b)).digest()] = None
        cm = cellmap(st)
        for y in range(st["Grid"]["Height"]):
            for x in range(st["Grid"]["Width"]):
                h_ = hashlib.md5(np.ascontiguousarray(fr[y*T:(y+1)*T, x*T:(x+1)*T])).digest()
                k = (cm.get((x, y), ()),
                     tuple(cm.get((x+dx, y+dy), ("__OUT__",))
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)))
                if stencil.setdefault(k, h_) != h_:
                    coll += 1
        if (i + 1) % 50 == 0:
            growth.append(len(tiles))
    print(f"{name}: {nmaps} random maps (sizes 6-15 x 6-13) | distinct {T}x{T} tiles={len(tiles)} "
          f"growth@50/100/150/200={growth} | 3x3-stencil keys={len(stencil)} COLLISIONS={coll}")
    del env
