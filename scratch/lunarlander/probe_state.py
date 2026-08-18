"""LunarLander-v3: is the frame a function of an EXTENDED state, and is the particle list bounded?

``docs/research/classic_control.md`` excluded this env because ``render()`` MUTATES what it draws
(``obj.ttl -= 0.15`` then ``_clean_particles``). That is an argument for widening the state, not for
excluding the env -- unless the thing being widened is unbounded or its update is random. So the three
deciding questions are asked here, each with a number:

  A. is the update inside ``render()`` DETERMINISTIC? (drive the same state twice from a fresh setup)
  B. is the particle count BOUNDED? measured over full episodes, rendering every step and not
     rendering at all, on the policy that fires the engines every step (the worst case).
  C. does a fixed-width state -- terrain + lander + legs + N particles -- determine the frame?
     Answered by driving the SAME env to the same state through two different histories and diffing.

It also records the geometry facts a port needs (which bodies are drawn, how many fixtures, the flag
being drawn once PER FIXTURE, the colours) so none of them has to be read off the source.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import gymnasium as gym

SCALE, W_VP, H_VP = 30.0, 600, 400


def make():
    e = gym.make("LunarLander-v3", render_mode="rgb_array")
    e.reset(seed=0)
    return e


e = make()
u = e.unwrapped
f = u.render()
print(f"frame {f.shape} {f.dtype}  unique colours {len(np.unique(f.reshape(-1,3),axis=0))} "
      f"mean {f.mean():.2f}   (blank would be 1 colour)")
print(f"drawlist: {[type(b).__name__ for b in u.drawlist]}  fixtures "
      f"{[len(b.fixtures) for b in u.drawlist]}")
print(f"lander colors {u.lander.color1} {u.lander.color2}; leg {u.legs[0].color1} {u.legs[0].color2}")
print(f"sky_polys {len(u.sky_polys)} each {len(u.sky_polys[0])} verts")
print(f"helipad x1 {u.helipad_x1} x2 {u.helipad_x2} y {u.helipad_y}   W {W_VP/SCALE} H {H_VP/SCALE}")

# ---------------------------------------------------------------- B. particle bound
def episode(policy, n=400, render_every=1):
    e = gym.make("LunarLander-v3", render_mode="rgb_array")
    e.reset(seed=1)
    u = e.unwrapped
    peak, ttls = 0, []
    for t in range(n):
        _, _, te, tr, _ = e.step(policy(t))
        if render_every and t % render_every == 0:
            u.render()
        peak = max(peak, len(u.particles))
        if u.particles:
            ttls.append(max(p.ttl for p in u.particles))
        if te or tr:
            break
    return peak, t + 1, (max(ttls) if ttls else 0.0)


fire_main = lambda t: 2
alternate = lambda t: [2, 1, 2, 3][t % 4]
rnd = np.random.default_rng(0)
random_pol = lambda t: int(rnd.integers(0, 4))
for name, pol in (("main every step", fire_main), ("main+side alternating", alternate),
                  ("uniform random", random_pol)):
    p1, n1, m1 = episode(pol, render_every=1)
    p0, n0, _ = episode(pol, render_every=0)
    print(f"particles, {name:22s}: render EVERY step peak {p1:3d} over {n1} steps (max ttl {m1:.3f}); "
          f"render NEVER peak {p0:4d} over {n0} steps")

# ---------------------------------------------------------------- A. determinism of render()
e = gym.make("LunarLander-v3", render_mode="rgb_array")
e.reset(seed=3)
u = e.unwrapped
for _ in range(6):
    e.step(2)
    u.render()
snap = [(float(p.position[0]), float(p.position[1]), float(p.ttl)) for p in u.particles]
print(f"\nparticles alive before the determinism probe: {len(snap)}")


def restore(u, snap, lander, legs, sky):
    """Put the env at the recorded state: terrain, lander, legs, and the particle list."""
    from Box2D.b2 import circleShape, fixtureDef
    while u.particles:
        u.world.DestroyBody(u.particles.pop(0))
    for x, y, ttl in snap:
        p = u.world.CreateDynamicBody(
            position=(x, y), angle=0.0,
            fixtures=fixtureDef(shape=circleShape(radius=2 / SCALE, pos=(0, 0)), density=1.0))
        p.ttl = ttl
        u.particles.append(p)
    u.lander.position, u.lander.angle = (lander[0], lander[1]), lander[2]
    for lg, (x, y, a) in zip(u.legs, legs):
        lg.position, lg.angle = (x, y), a
    u.sky_polys = [list(map(tuple, p)) for p in sky]


L = (float(u.lander.position[0]), float(u.lander.position[1]), float(u.lander.angle))
LG = [(float(b.position[0]), float(b.position[1]), float(b.angle)) for b in u.legs]
SKY = [[tuple(map(float, v)) for v in poly] for poly in u.sky_polys]

restore(u, snap, L, LG, SKY)
a = u.render().copy()
restore(u, snap, L, LG, SKY)
b = u.render().copy()
print(f"A. render() twice from the SAME restored state: max abs diff "
      f"{int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max())}   "
      f"(if 0, the mutation inside render() is deterministic)")
c = u.render().copy()          # WITHOUT restoring: this is the impurity the doc recorded
print(f"   render() again WITHOUT restoring: max abs diff {int(np.abs(a.astype(np.int16)-c.astype(np.int16)).max())} "
      f"(non-zero is the mutation, and is why the particles must be in the state)")

# ---------------------------------------------------------------- C. state sufficiency via 2 histories
e2 = gym.make("LunarLander-v3", render_mode="rgb_array")
e2.reset(seed=11)                       # a DIFFERENT terrain and a different history
u2 = e2.unwrapped
for _ in range(20):
    e2.step(1)
    u2.render()
restore(u2, snap, L, LG, SKY)
d = u2.render().copy()
print(f"C. a DIFFERENT env driven to the same extended state: max abs diff vs a "
      f"{int(np.abs(a.astype(np.int16)-d.astype(np.int16)).max())}")

# ---------------------------------------------------------------- the flag-per-fixture effect
def n_fixture_frame(k):
    """Render with k particles far off screen; the flags are drawn once per fixture, so if the count
    changes the frame at all, the AA fringe of the flag depends on how many particles exist."""
    e3 = gym.make("LunarLander-v3", render_mode="rgb_array")
    e3.reset(seed=5)
    u3 = e3.unwrapped
    restore(u3, [(-50.0, -50.0, 5.0)] * k, L, LG, SKY)
    return u3.render().copy()


f0, f1, f4 = n_fixture_frame(0), n_fixture_frame(1), n_fixture_frame(4)
print(f"flags drawn once per fixture: 0 vs 1 off-screen particle -> "
      f"{int(np.abs(f0.astype(np.int16)-f1.astype(np.int16)).max())}, 0 vs 4 -> "
      f"{int(np.abs(f0.astype(np.int16)-f4.astype(np.int16)).max())}, "
      f"differing px {int((np.abs(f0.astype(np.int16)-f4.astype(np.int16)).max(-1)>0).sum())}")
