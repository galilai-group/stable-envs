"""Sokoban tile atlas: build it from the shipped PNGs and prove the gather IS the official frame.

Three questions, all answered by measurement rather than by reading ``room_to_rgb``:

  1. what the atlas is (shape, bytes, colour census) and whether a pure gather reproduces
     ``render(mode='rgb_array')`` bit for bit, over MANY rooms rather than one;
  2. what the per-cell alphabet is -- the render rewrites id 5 to 6 where ``room_structure == 2``, so
     the tile index is a function of two grids and the question is whether one 7-value grid recovers
     both of them;
  3. how many DISTINCT colours the seven tiles use between them, because that fixes the label
     decomposition a recolouring variation needs.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import time
import numpy as np
import imageio.v2 as imageio
import pkg_resources
import gym_sokoban  # noqa: F401
import gym

NAMES = ["wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target"]
PKG = "gym_sokoban.envs"


def atlas():
    out = np.zeros((7, 16, 16, 3), np.uint8)
    for i, n in enumerate(NAMES):
        p = pkg_resources.resource_filename(PKG, "/".join(("surface", n + ".png")))
        out[i] = np.asarray(imageio.imread(p))[..., :3]
    return out


A = atlas()
print(f"atlas {A.shape} {A.dtype} = {A.nbytes} bytes")
cols, inv = np.unique(A.reshape(-1, 3), axis=0, return_inverse=True)
print(f"distinct colours across the whole atlas: {len(cols)}")
for c in cols:
    where = sorted({NAMES[i] for i in np.unique(np.where((A == c).all(-1))[0])})
    print(f"   {tuple(int(v) for v in c)}  used by {where}")
print("per-tile colour count:", [len(np.unique(A[i].reshape(-1, 3), axis=0)) for i in range(7)])

# id -> (room_state, room_fixed) inverse, checked against real rooms below
IDMAP = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (3, 2), 4: (4, 1), 5: (5, 1), 6: (5, 2)}

env = gym.make("Sokoban-v0")
u = env.unwrapped
bad = 0
seen = set()
pairs = set()
t_ref = 0.0
for ep in range(12):
    u.reset()
    rng = np.random.default_rng(ep)
    for t in range(10):
        room, fixed = u.room_state, u.room_fixed
        t0 = time.perf_counter()
        f = u.render(mode="rgb_array")
        t_ref += time.perf_counter() - t0
        tile = room.copy()
        tile[(room == 5) & (fixed == 2)] = 6
        seen |= set(np.unique(tile).tolist())
        pairs |= {(int(a), int(b)) for a, b in zip(room.ravel(), fixed.ravel())}
        rebuilt = A[tile].transpose(0, 2, 1, 3, 4).reshape(room.shape[0] * 16, room.shape[1] * 16, 3)
        bad = max(bad, int(np.abs(rebuilt.astype(np.int16) - f.astype(np.int16)).max()))
        # the inverse: can one 7-value grid recover (room_state, room_fixed)?
        inv_state = np.array([IDMAP[int(v)][0] for v in tile.ravel()]).reshape(room.shape)
        inv_fixed = np.array([IDMAP[int(v)][1] for v in tile.ravel()]).reshape(room.shape)
        assert (inv_state == room).all(), "room_state is NOT recoverable from the tile id"
        assert (inv_fixed == fixed).all(), "room_fixed is NOT recoverable from the tile id"
        u.step(int(rng.integers(0, 9)))
print(f"\ngather vs official render over 120 frames: max abs diff {bad}")
print(f"tile ids seen: {sorted(seen)}")
print(f"(room_state, room_fixed) pairs seen: {sorted(pairs)}")
print(f"official renderer: {120 / t_ref:,.0f} frames/s  ({1000*t_ref/120:.2f} ms/frame)")
print(f"room {u.room_state.shape}  frame {u.render(mode='rgb_array').shape}")
