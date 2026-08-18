"""Probe 3: why does zelda's load_state round-trip differ by 760 px?

Compares the state dict before save and after restore, and localises the differing pixels
to grid cells so we can name the object that is not restored.
"""
import json, hashlib
import numpy as np
import griddly
from griddly import GymWrapper, gd

YAML = "Single-Player/GVGAI/zelda.yaml"
TILE = 24


def gframe(env):
    return np.array(env.game.observe(), copy=True).swapaxes(0, 2)


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [int(rng.integers(k)) for k in sp.nvec]


def objs_of(st):
    return sorted((o["Name"], tuple(o["Location"]), o["Orientation"], o["PlayerId"],
                   o["RenderTileId"], json.dumps(o["Variables"], sort_keys=True))
                  for o in st["Objects"])


env = GymWrapper(yaml_file=YAML, level=0,
                 global_observer_type=gd.ObserverType.SPRITE_2D,
                 player_observer_type=gd.ObserverType.SPRITE_2D)
env.seed(1234)
env.reset(level_id=0)
rng = np.random.default_rng(0)

snaps = []
for i in range(8):
    for _ in range(6):
        env.step(sample_action(env, rng))
    snaps.append((env.get_state(), gframe(env)))

# churn
env.seed(9999); env.reset(level_id=1)
r2 = np.random.default_rng(7)
for _ in range(120):
    env.step(sample_action(env, r2))
env.reset(level_id=0)
for _ in range(40):
    env.step(sample_action(env, r2))

for i, (st, f0) in enumerate(snaps):
    env2 = env.load_state(st)
    st2 = env2.get_state()
    fr = gframe(env2)
    d = np.abs(fr.astype(int) - f0.astype(int))
    ndiff = int((d.max(axis=2) > 0).sum())
    a, b = objs_of(st), objs_of(st2)
    only_a = [x for x in a if x not in b]
    only_b = [x for x in b if x not in a]
    same_objects = (a == b)
    dv_keys = {k: (st.get(k), st2.get(k)) for k in st
               if k not in ("Objects", "Hash") and st.get(k) != st2.get(k)}
    print(f"--- ckpt {i}: maxdiff={int(d.max())} ndiff_px={ndiff} objects_identical={same_objects}")
    if not same_objects:
        print("    only in saved   :", only_a)
        print("    only in restored:", only_b)
    if dv_keys:
        for k, (x, y) in dv_keys.items():
            xs, ys = str(x), str(y)
            print(f"    {k}: saved={xs[:160]} restored={ys[:160]}")
    if ndiff:
        ys, xs = np.where(d.max(axis=2) > 0)
        cells = sorted({(int(x) // TILE, int(y) // TILE) for x, y in zip(xs, ys)})
        print("    differing cells (x,y):", cells)
        # what is at those cells according to the two states?
        for c in cells:
            in_a = [o for o in st["Objects"] if tuple(o["Location"]) == c]
            in_b = [o for o in st2["Objects"] if tuple(o["Location"]) == c]
            print(f"      cell {c} saved={[ (o['Name'],o['RenderTileId'],o['Variables']) for o in in_a]}"
                  f" restored={[ (o['Name'],o['RenderTileId'],o['Variables']) for o in in_b]}")
    del env2

# Second, cleaner G1 test: is the frame a function of the state INDEPENDENT of load_state?
# render the same restored state twice from two separate load_state calls.
print("=== two independent restores of the same state ===")
for i, (st, f0) in enumerate(snaps[:4]):
    e1 = env.load_state(st); f1 = gframe(e1)
    e2 = env.load_state(st); f2 = gframe(e2)
    print(f"  ckpt {i}: restore-vs-restore maxdiff={int(np.abs(f1.astype(int)-f2.astype(int)).max())}"
          f"  restore-vs-original maxdiff={int(np.abs(f1.astype(int)-f0.astype(int)).max())}")
    del e1, e2
