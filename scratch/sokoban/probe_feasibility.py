"""Is gym-sokoban a polygon env or a tile env? Measure the tile alphabet and the frame construction.

``render_utils.room_to_rgb`` blits one of seven 16x16 PNGs per cell. If that is literally all it does,
Sokoban belongs on the repo's existing tilemap path (exact by construction, like MiniGrid) and forcing
it through the polygon IR would re-derive images that already exist as assets.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import time
import numpy as np
import imageio.v2 as imageio
import pkg_resources
import gym_sokoban  # noqa: F401
import gym

PKG = "gym_sokoban.envs"
NAMES = ["wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target"]
tiles = {}
for n in NAMES:
    p = pkg_resources.resource_filename(PKG, "/".join(("surface", n + ".png")))
    tiles[n] = imageio.imread(p)
print("tile alphabet (room_to_rgb surfaces list, in room-id order):")
for i, n in enumerate(NAMES):
    t = tiles[n]
    print(f"  {i} {n:18s} {t.shape} dtype={t.dtype} distinct colours={len(np.unique(t.reshape(-1, t.shape[-1]), axis=0))}")
flat = np.stack([t[..., :3] for t in tiles.values()])
print(f"  atlas as one array: {flat.shape} = {flat.nbytes} bytes")

env = gym.make("Sokoban-v0")
env.reset()
u = env.unwrapped
f = u.render(mode="rgb_array")
print(f"\nroom {u.room_state.shape} ids present {sorted(set(u.room_state.ravel().tolist()))}")
print(f"frame {f.shape}  = room * 16")
print(f"state as a room grid: {u.room_state.size} bytes; frame: {f.nbytes} bytes "
      f"({f.nbytes / u.room_state.size:.0f}x)")

# is the frame exactly a gather from the atlas?
rebuilt = np.zeros_like(f)
for i in range(u.room_state.shape[0]):
    for j in range(u.room_state.shape[1]):
        rebuilt[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = flat[u.room_state[i, j]]
print(f"gather from the atlas vs official render: max abs diff "
      f"{int(np.abs(rebuilt.astype(np.int16) - f.astype(np.int16)).max())}")

for _ in range(3):
    u.render(mode="rgb_array")
t0 = time.perf_counter()
for _ in range(50):
    u.render(mode="rgb_array")
print(f"official renderer: {50/(time.perf_counter()-t0):,.0f} frames/s")

# how much of a Sokoban tile is expressible as convex polygons / discs?
w = tiles["wall"][..., :3]
print(f"\nwall tile: {len(np.unique(w.reshape(-1,3),axis=0))} distinct colours in 16x16 px")
print(f"player tile: {len(np.unique(tiles['player'][...,:3].reshape(-1,3),axis=0))} distinct colours")
