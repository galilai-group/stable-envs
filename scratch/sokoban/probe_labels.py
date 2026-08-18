"""Name the ten atlas colours by where they are painted, and check the tile alphabet is covered.

Two things a port needs and neither is a matter of opinion:

  * a recolouring variation is a palette over the atlas's DISTINCT colours, so each one needs a name
    that says what it paints. Printed here as the per-tile pixel counts and the bounding box of each
    colour inside each tile, which is enough to name them without inventing anything;
  * every colour must appear in the frames the conformance suite renders, or its knob looks inert and
    the check passes for the wrong reason. So: how often does each of the seven tile ids occur over a
    pool of rollout states, and does a 16-state batch contain all seven?
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pkg_resources
import imageio.v2 as imageio
import gym_sokoban  # noqa: F401
import gym

NAMES = ["wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target"]
A = np.stack([np.asarray(imageio.imread(pkg_resources.resource_filename(
    "gym_sokoban.envs", "/".join(("surface", n + ".png")))))[..., :3] for n in NAMES])
cols = np.unique(A.reshape(-1, 3), axis=0)
print(f"{len(cols)} colours\n")
for k, c in enumerate(cols):
    print(f"colour {k} {tuple(int(v) for v in c)}")
    for i, n in enumerate(NAMES):
        m = (A[i] == c).all(-1)
        if m.any():
            ys, xs = np.where(m)
            print(f"    {n:18s} {int(m.sum()):4d} px  rows {ys.min():2d}..{ys.max():2d} "
                  f"cols {xs.min():2d}..{xs.max():2d}")

env = gym.make("Sokoban-v0")
u = env.unwrapped
pool = []
for ep in range(40):
    u.reset()
    rng = np.random.default_rng(ep)
    for t in range(16):
        tile = u.room_state.copy()
        tile[(u.room_state == 5) & (u.room_fixed == 2)] = 6
        pool.append(tile.reshape(-1))
        u.step(int(rng.integers(0, 9)))
pool = np.stack(pool)
print(f"\npool {pool.shape}")
for i in range(7):
    has = (pool == i).any(1)
    print(f"  tile {i} {NAMES[i]:18s} present in {100*has.mean():5.1f}% of states, "
          f"mean {(pool == i).sum(1).mean():5.2f} cells")
rng = np.random.default_rng(0)
miss = 0
for _ in range(500):
    b = pool[rng.choice(len(pool), 16, replace=False)]
    if len({int(v) for v in np.unique(b)}) < 7:
        miss += 1
print(f"  16-state batches missing at least one tile id: {miss}/500")
