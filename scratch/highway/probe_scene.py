"""Dump everything the highway-env renderer reads, for one reset of each target env id.

Purpose: find the minimal state. Anything printed here that is NOT a function of
(vehicle poses + a per-episode layout id) would sink the port.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import gymnasium
import highway_env  # noqa: F401
from highway_env.road.lane import StraightLane, SineLane, CircularLane, LineType

IDS = ["highway-v0", "merge-v0", "roundabout-v0", "intersection-v0", "racetrack-v0", "parking-v0"]


def lane_desc(l):
    if isinstance(l, SineLane):
        return f"Sine start={l.start} end={l.end} amp={l.amplitude} puls={l.pulsation:.5f} ph={l.phase:.4f} w={l.width} lt={l.line_types} len={l.length:.2f}"
    if isinstance(l, StraightLane):
        return f"Straight start={l.start} end={l.end} w={l.width} lt={l.line_types} len={l.length:.2f}"
    if isinstance(l, CircularLane):
        return f"Circular c={l.center} r={l.radius} p0={l.start_phase:.4f} p1={l.end_phase:.4f} cw={l.clockwise} w={l.width} lt={l.line_types} len={l.length:.2f}"
    return f"{type(l).__name__} {vars(l).keys()}"


for eid in IDS:
    print("=" * 100)
    env = gymnasium.make(eid, render_mode="rgb_array")
    env.reset(seed=0)
    u = env.unwrapped
    cfg = u.config
    print(f"{eid}: screen={cfg['screen_width']}x{cfg['screen_height']} scaling={cfg['scaling']} "
          f"centering={cfg['centering_position']} offscreen={cfg['offscreen_rendering']} "
          f"show_traj={cfg['show_trajectories']} record_history={u.road.record_history}")
    net = u.road.network.graph
    nl = sum(len(v) for d in net.values() for v in d.values())
    print(f"  lanes: {nl}")
    for a in net:
        for b in net[a]:
            for i, l in enumerate(net[a][b]):
                print(f"    ({a},{b},{i}) {lane_desc(l)}")
    print(f"  vehicles: {len(u.road.vehicles)}  objects: {len(u.road.objects)}")
    for v in u.road.vehicles[:6]:
        print(f"    {type(v).__name__:18s} pos={np.round(v.position,3)} h={v.heading:+.5f} "
              f"L={v.LENGTH} W={v.WIDTH} color={getattr(v,'color',None)} crashed={v.crashed} "
              f"act={getattr(v,'action',None)}")
    for o in u.road.objects:
        print(f"    OBJ {type(o).__name__:12s} pos={np.round(o.position,3)} h={o.heading:+.5f} "
              f"L={o.LENGTH} W={o.WIDTH} hit={getattr(o,'hit',None)} crashed={getattr(o,'crashed',None)}")
    f = env.render()
    cols, cnt = np.unique(f.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-cnt)
    print(f"  frame {f.shape} colours:")
    for i in order[:12]:
        print(f"    {tuple(int(c) for c in cols[i])}  {cnt[i]}")
    np.save(f"/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/highway/frame_{eid}.npy", f)
    env.close()
