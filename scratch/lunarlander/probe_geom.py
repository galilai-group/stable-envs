"""The geometry constants a LunarLander port needs, read off the LIVE env rather than the source.

Fixture vertex order, the particle circle radius after ``* SCALE``, the terrain the sky polygons are
built from, how many particles are alive at snapshot time (BEFORE the render that decrements them),
and the ttl range a real rollout produces.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import gymnasium as gym

SCALE = 30.0
e = gym.make("LunarLander-v3", render_mode="rgb_array")
e.reset(seed=0)
u = e.unwrapped
u.render()

for name, b in (("lander", u.lander), ("leg0", u.legs[0]), ("leg1", u.legs[1])):
    for f in b.fixtures:
        print(f"{name}: shape {type(f.shape).__name__} vertices "
              f"{[tuple(round(v * SCALE, 6) for v in vv) for vv in f.shape.vertices]}")
print(f"lander colors {u.lander.color1} {u.lander.color2}")

e.step(2)
p = u.particles[-1]
f = p.fixtures[0]
print(f"particle: radius {f.shape.radius!r} pos {tuple(f.shape.pos)}  radius*SCALE "
      f"{f.shape.radius * SCALE!r} -> int {int(f.shape.radius * SCALE)}")

print(f"\nsky_polys[0] {[tuple(map(float, v)) for v in u.sky_polys[0]]}")
print(f"sky_polys[9] {[tuple(map(float, v)) for v in u.sky_polys[9]]}")
sy = [float(u.sky_polys[i][0][1]) for i in range(10)] + [float(u.sky_polys[9][1][1])]
cx = [float(u.sky_polys[i][0][0]) for i in range(10)] + [float(u.sky_polys[9][1][0])]
print(f"smooth_y {np.round(sy, 6).tolist()}")
print(f"chunk_x  {np.round(cx, 6).tolist()}  (expected {[20.0/10*i for i in range(11)]})")
print(f"helipad_y {u.helipad_y!r}  H {400/SCALE!r}  H/4 {400/SCALE/4!r}")

# how many particles exist at snapshot time (after step, before render), over full episodes
rng = np.random.default_rng(0)
peak_pre = peak_post = 0
ttls = []
for ep in range(8):
    e.reset(seed=100 + ep)
    for t in range(300):
        a = 2 if rng.random() < 0.7 else int(rng.integers(0, 4))
        _, _, te, tr, _ = e.step(a)
        peak_pre = max(peak_pre, len(u.particles))
        ttls += [float(p.ttl) for p in u.particles]
        u.render()
        peak_post = max(peak_post, len(u.particles))
        if te or tr:
            break
print(f"\nparticles alive: PRE-render peak {peak_pre}, POST-render peak {peak_post}")
print(f"ttl at snapshot: min {min(ttls):.4f} max {max(ttls):.4f}")
print(f"colour after one decrement at max ttl: "
      f"{int(max(0.2, 0.15 + max(ttls) - 0.15) * 255)} (must be <= 255)")
