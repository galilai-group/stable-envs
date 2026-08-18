"""Probe 11: how many atlas entries does orientation cost?

(1) GLOBAL observer. Collect the T x T tile of an oriented object (zelda avatar, robot_tag
    tagger) for each of its orientations at the SAME cell with the SAME neighbourhood, and test
    whether the four tiles are exact rot90 / flip permutations of one another. If they are, the
    atlas is one sprite + a runtime rotation; if not, the atlas needs 4 baked entries per object
    (which is what MiniGrid already does for agent_dir).

(2) POV observer with RotateWithAvatar. Render the same underlying state with
    RotateWithAvatar true and false and test whether the rotated frame is an exact np.rot90 of
    the unrotated one. Reports max abs diff.
"""
import os, json, hashlib, collections, itertools
import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
T = 24


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
        m[tuple(o["Location"])].append((o["Name"], o["Orientation"], o["PlayerId"], o["RenderTileId"]))
    return {k: tuple(sorted(v)) for k, v in m.items()}


print("### (1) orientation tiles in the GLOBAL observer ###")
for name, yf, target in [("zelda", "Single-Player/GVGAI/zelda.yaml", "avatar"),
                         ("robot_tag_4", "Multi-Agent/robot_tag_4.yaml", "tagger"),
                         ("drunkdwarf", "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml", "drunk_dwarf")]:
    env = GymWrapper(yaml_file=yf, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    tiles = {}          # (neighbourhood_key, orientation) -> tile
    for ep in range(14):
        env.seed(ep); env.reset(level_id=ep % env.level_count)
        rng = np.random.default_rng(ep * 5)
        for t in range(60):
            st = env.get_state()
            fr = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
            cm = cellmap(st)
            for o in st["Objects"]:
                if o["Name"] != target:
                    continue
                x, y = o["Location"]
                nb = tuple(cm.get((x + dx, y + dy), ("__OUT__",))
                           for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0))
                others = tuple(v for v in cm.get((x, y), ()) if v[0] != target)
                key = (nb, others, o["PlayerId"], o["RenderTileId"])
                tiles.setdefault((key, o["Orientation"]), fr[y*T:(y+1)*T, x*T:(x+1)*T].copy())
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done):
                break
    groups = collections.defaultdict(dict)
    for (key, ori), tile in tiles.items():
        groups[key][ori] = tile
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"\n{name}/{target}: {len(tiles)} (context, orientation) tiles, "
          f"{len(multi)} contexts seen with >=2 orientations")
    ops = {"rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2),
           "rot270": lambda a: np.rot90(a, 3), "fliplr": np.fliplr, "flipud": np.flipud,
           "transpose": lambda a: a.transpose(1, 0, 2)}
    hit = collections.Counter(); tot = 0
    worst = []
    for key, d in list(multi.items())[:60]:
        for a, b in itertools.combinations(sorted(d), 2):
            tot += 1
            best = ("none", int(np.abs(d[a].astype(int) - d[b].astype(int)).max()))
            for opn, op in ops.items():
                m = int(np.abs(op(d[a]).astype(int) - d[b].astype(int)).max())
                if m < best[1]:
                    best = (opn, m)
            if best[1] == 0:
                hit[best[0]] += 1
            else:
                hit["NO EXACT SYMMETRY"] += 1
                worst.append((a, b, best))
    print(f"   {tot} orientation pairs compared -> {dict(hit)}")
    for w in worst[:4]:
        print(f"     e.g. {w[0]} vs {w[1]}: best op {w[2][0]} leaves max abs diff {w[2][1]}")
    del env

print("\n### (2) POV: is RotateWithAvatar an exact rot90 of the unrotated window? ###")
yf = "Single-Player/GVGAI/zelda_partially_observable.yaml"
src = open(os.path.join(GAMES_DIR, yf)).read()
ys_off = src.replace("RotateWithAvatar: true", "RotateWithAvatar: false")

ea = GymWrapper(yaml_file=yf, level=0, global_observer_type=gd.ObserverType.SPRITE_2D,
                player_observer_type=gd.ObserverType.SPRITE_2D)
eb = GymWrapper(yaml_string=ys_off, level=0, global_observer_type=gd.ObserverType.SPRITE_2D,
                player_observer_type=gd.ObserverType.SPRITE_2D)
stats = collections.Counter()
mx = collections.defaultdict(int)
for ep in range(6):
    ea.seed(ep); ea.reset(level_id=ep % ea.level_count)
    eb.seed(ep); eb.reset(level_id=ep % eb.level_count)
    rng = np.random.default_rng(ep)
    for t in range(40):
        sa = ea.get_state()
        ori = [o["Orientation"] for o in sa["Objects"] if o["Name"] == "avatar"]
        fa = np.array(ea._players[0].observe(), copy=True).swapaxes(0, 2)
        fb = np.array(eb._players[0].observe(), copy=True).swapaxes(0, 2)
        best = min(((k, int(np.abs(np.rot90(fb, k).astype(int) - fa.astype(int)).max()))
                    for k in range(4)), key=lambda z: z[1])
        o = ori[0] if ori else "?"
        stats[(o, best[0], best[1] == 0)] += 1
        mx[o] = max(mx[o], best[1])
        a = sample_action(ea, rng)
        try:
            ea.step(a); eb.step(a)
        except Exception:
            break
print("   (orientation, best_rot90_k, exact) -> count:", dict(stats))
print("   max abs diff of best rotation, per orientation:", dict(mx))
