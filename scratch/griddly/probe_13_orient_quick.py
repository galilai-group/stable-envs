"""Probe 13: the short version of probe 11 -- are an object's orientation tiles exact symmetries?

zelda's avatar changes appearance with Orientation (probe_03: 380/576 px of one tile). If the four
orientation tiles are exact rot90/flip images of one another, the atlas needs one entry per object;
if not, it needs four baked entries per oriented object (which is what MiniGrid already does for
agent_dir). Drives the avatar to all four orientations on the SAME cell of the SAME level.
"""
import itertools, collections
import numpy as np
import griddly
from griddly import GymWrapper, gd

T = 24
env = GymWrapper(yaml_file="Single-Player/GVGAI/zelda.yaml", level=0,
                 global_observer_type=gd.ObserverType.SPRITE_2D,
                 player_observer_type=gd.ObserverType.SPRITE_2D)
env.seed(0); env.reset(level_id=0)
rng = np.random.default_rng(0)

# collect avatar tiles keyed by (cell, orientation) while the surroundings stay empty
tiles = {}
for t in range(400):
    st = env.get_state()
    fr = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
    for o in st["Objects"]:
        if o["Name"] != "avatar":
            continue
        x, y = o["Location"]
        occupied = {tuple(q["Location"]) for q in st["Objects"]
                    if q["Name"] not in ("avatar", "_empty")}
        nb_clear = all((x + dx, y + dy) not in occupied
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        if nb_clear:
            tiles.setdefault(o["Orientation"], fr[y*T:(y+1)*T, x*T:(x+1)*T].copy())
    if len(tiles) >= 4:
        break
    env.step([int(rng.integers(2)), int(rng.integers(5))])

print("orientations captured:", sorted(tiles))
ops = {"identity": lambda a: a, "rot90": lambda a: np.rot90(a, 1),
       "rot180": lambda a: np.rot90(a, 2), "rot270": lambda a: np.rot90(a, 3),
       "fliplr": np.fliplr, "flipud": np.flipud,
       "transpose": lambda a: a.transpose(1, 0, 2),
       "anti-transpose": lambda a: np.rot90(a, 2).transpose(1, 0, 2)}
for a, b in itertools.combinations(sorted(tiles), 2):
    res = []
    for n, op in ops.items():
        d = int(np.abs(op(tiles[a]).astype(int) - tiles[b].astype(int)).max())
        nd = int((np.abs(op(tiles[a]).astype(int) - tiles[b].astype(int)).max(axis=2) > 0).sum())
        res.append((d, nd, n))
    res.sort()
    print(f"  {a:6s} vs {b:6s}: best = {res[0][2]:14s} max_abs={res[0][0]:3d} ndiff={res[0][1]}/{T*T}"
          f"   (exact symmetry: {res[0][0] == 0})")
