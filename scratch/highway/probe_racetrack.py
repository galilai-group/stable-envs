"""racetrack-v0 / parking-v0: dump the road network and pin down the tyre sprites.

Two jobs. First, emit the lane list as Python source, read off the OFFICIAL ``road.network`` rather
than transcribed from ``_make_road`` by hand -- 30-odd lanes with hand-written radii and phases is a
transcription with a defect rate, and the emitted literal can be re-checked against the package at
``reference_env`` time.

Second, measure what makes these two envs different from the four already ported: their ego is a plain
``Vehicle``, so ``VehicleGraphics`` draws four tyre sprites, each its own rotate-and-blit INSIDE the
vehicle sprite, which is then rotated again. This records the sprite geometry (sizes, positions,
angles) at both scalings, and the vehicle classes actually present after reset.
"""
import os
os.environ["SDL_VIDEODRIVER"] = "offscreen"
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from highway_env.road.lane import CircularLane, LineType, SineLane, StraightLane


def cfg(env_id):
    return {"offscreen_rendering": True, "show_trajectories": False, "render_agent": False}


def dump(env_id):
    env = gym.make(env_id, render_mode="rgb_array", config=cfg(env_id))
    env.reset(seed=0)
    u = env.unwrapped
    print(f"\n================ {env_id}")
    c = u.config
    print(f"screen {c['screen_width']}x{c['screen_height']} scaling {c['scaling']} "
          f"centering {c['centering_position']}")
    print(f"vehicles {[type(v).__name__ for v in u.road.vehicles]}")
    print(f"ego is {type(u.vehicle).__name__}  action {u.vehicle.action}  "
          f"LENGTH {u.vehicle.LENGTH} WIDTH {u.vehicle.WIDTH}")
    print(f"objects {[(type(o).__name__, tuple(np.round(o.position,3)), round(float(o.heading),4), o.LENGTH, o.WIDTH) for o in u.road.objects]}")
    lines = []
    for _from in u.road.network.graph:
        for _to in u.road.network.graph[_from]:
            for l in u.road.network.graph[_from][_to]:
                lt = tuple(int(t) for t in l.line_types)
                if isinstance(l, SineLane):
                    lines.append(f"    Sine({list(np.round(l.start,6))}, {list(np.round(l.end,6))}, "
                                 f"{l.amplitude!r}, {l.pulsation!r}, {l.phase!r}, width={l.width!r}, "
                                 f"line_types={lt}),")
                elif isinstance(l, StraightLane):
                    lines.append(f"    Straight({list(np.round(l.start,6))}, {list(np.round(l.end,6))}, "
                                 f"width={l.width!r}, line_types={lt}),")
                elif isinstance(l, CircularLane):
                    lines.append(f"    Circular({list(np.round(l.center,6))}, {l.radius!r}, "
                                 f"{repr(float(l.start_phase))}, {repr(float(l.end_phase))}, "
                                 f"clockwise={l.direction > 0}, width={l.width!r}, line_types={lt}),")
                else:
                    lines.append(f"    # UNKNOWN LANE TYPE {type(l).__name__}")
    print(f"{len(lines)} lanes:")
    print("\n".join(lines))
    return env


for eid in ("racetrack-v0", "parking-v0"):
    dump(eid)

# ---------------------------------------------------------------- tyre sprite geometry
print("\n================ tyre sprite geometry (VehicleGraphics.display)")
for scaling in (5.5, 7.0):
    pix = lambda v: int(v * scaling)
    LEN, WID, TL, TW = 5.0, 2.0, 1.0, 0.3
    span = LEN + 2 * TL
    S, St = pix(span), pix(TL)
    body = (pix(TL), pix(span / 2 - WID / 2), pix(LEN), pix(WID))
    tire_rect = (0, pix(TL / 2 - TW / 2), pix(TL), pix(TW))
    pos = [[pix(TL), pix(span / 2 - WID / 2)], [pix(TL), pix(span / 2 + WID / 2)],
           [pix(span - TL), pix(span / 2 - WID / 2)], [pix(span - TL), pix(span / 2 + WID / 2)]]
    print(f"scaling {scaling}: vehicle sprite {S}x{S}, body rect {body}; "
          f"tyre sprite {St}x{St}, tyre rect {tire_rect}, positions {pos}")

# ---------------------------------------------------------------- what steering range occurs
env = gym.make("racetrack-v0", render_mode="rgb_array", config=cfg("racetrack-v0"))
env.reset(seed=0)
u = env.unwrapped
st = []
for ep in range(6):
    env.reset(seed=ep)
    rng = np.random.default_rng(ep)
    for _ in range(40):
        a = env.action_space.sample()
        env.step(a)
        st.append(float(u.vehicle.action["steering"]))
st = np.array(st)
print(f"\nracetrack ego steering over 240 steps: min {st.min():.4f} max {st.max():.4f} "
      f"|.| mean {np.abs(st).mean():.4f}   action_space {env.action_space}")
env2 = gym.make("parking-v0", render_mode="rgb_array", config=cfg("parking-v0"))
env2.reset(seed=0)
u2 = env2.unwrapped
st2 = []
for ep in range(6):
    env2.reset(seed=ep)
    for _ in range(40):
        env2.step(env2.action_space.sample())
        st2.append(float(u2.vehicle.action["steering"]))
st2 = np.array(st2)
print(f"parking ego steering over 240 steps: min {st2.min():.4f} max {st2.max():.4f}")
print(f"parking goal positions over 12 resets:")
seen = set()
for ep in range(12):
    env2.reset(seed=ep)
    g = u2.vehicle.goal
    seen.add((round(float(g.position[0]), 3), round(float(g.position[1]), 3),
              round(float(g.heading), 4)))
for s in sorted(seen):
    print("   ", s)
print(f"parking ego spawn heading is 2*pi*U(0,1); LENGTH/WIDTH {u2.vehicle.LENGTH}/{u2.vehicle.WIDTH}")
