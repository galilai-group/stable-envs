"""Is CarRacing-v3 a Scene2D env? Measure the four things that decide it.

1. How many polygons a frame actually contains, and how many survive culling to the visible window.
2. Whether the 96x96 observation is a rasterization or a filtered image (``smoothscale`` from 1000x800).
3. Whether anything in the frame is not geometry (the reward text is drawn with a pygame font).
4. What the official renderer costs, which is the bar any port has to beat.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import time
import numpy as np
import gymnasium


def interior_frac(frames):
    import scipy.ndimage as ndi
    out = []
    for f in frames:
        g = f.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
                + np.abs(np.diff(g, axis=1)).sum(-1)[:-1]) == 0
        out.append(ndi.minimum_filter(flat, size=3).mean())
    return float(np.mean(out))


for mode, name in (("state_pixels", "96x96 observation"), ("rgb_array", "600x400 video")):
    env = gymnasium.make("CarRacing-v3", render_mode=mode)
    env.reset(seed=0)
    u = env.unwrapped
    for _ in range(30):
        env.step(np.array([0.0, 0.2, 0.0], np.float32))
    f = env.render()
    print(f"\n=== {name} ({mode}) ===")
    print(f"  frame {f.shape}  unique colours {len(np.unique(f.reshape(-1,3),axis=0))}")
    print(f"  road_poly quads {len(u.road_poly)}  track tiles {len(u.track)}  "
          f"grass quads {len(range(-20,20,2))**2}")
    print(f"  interior (flat 3x3) fraction of the frame: {interior_frac([f]):.4f}")
    cols, cnt = np.unique(f.reshape(-1, 3), axis=0, return_counts=True)
    o = np.argsort(-cnt)
    print("  top colours: " + ", ".join(f"{tuple(int(c) for c in cols[i])}:{cnt[i]}" for i in o[:6]))

    for _ in range(5):
        env.render()
    t0 = time.perf_counter()
    for _ in range(60):
        env.render()
    print(f"  official renderer: {60/(time.perf_counter()-t0):,.0f} frames/s")
    env.close()

# Does the frame depend on anything that is not geometry?
env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
env.reset(seed=0)
u = env.unwrapped
for _ in range(40):
    env.step(np.array([0.0, 0.3, 0.0], np.float32))
a = env.render().copy()
saved = u.reward
u.reward = saved + 137.0                      # only the on-screen score text changes
b = env.render().copy()
u.reward = saved
d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
print(f"\nreward text: changing only env.reward moves {int((d>0).sum())} of {d.size} pixels "
      f"(max {int(d.max())}/255) in the 96x96 observation")

# how much of the geometry is on screen at any time
u2 = env.unwrapped
zoom = 2.7 * 6.0
print(f"visible world window at ZOOM*SCALE={zoom:.1f}: "
      f"{1000/zoom:.1f} x {800/zoom:.1f} m; track detail step 3.5 m, "
      f"grass cell {2000/6/20:.1f} m")
