"""Probe 2: G1 (purity) and G3 (shared decoder) for every MiniWorld id.

G3 first, because it is cheaper and it decides what the state can even be: reset each id under
several seeds and diff everything the renderer reads that is NOT agent/entity pose — room
outlines, wall polygons, texture names, entity roster, light and sky. Anything that varies is
either part of the state (and then it is per-episode CONTENT, which the fixed-width contract
forbids) or it is a constant of the class.

G1 second, and only for the ids that passed G3: take a state produced by episode A, write it into
the SAME env re-reset under seed B (a genuinely different internal history), render both, diff.
Threshold 0.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pyglet

pyglet.options["headless"] = True

import gymnasium as gym  # noqa: E402
import miniworld  # noqa: E402,F401

IDS = [
    "MiniWorld-OneRoom-v0", "MiniWorld-OneRoomS6-v0", "MiniWorld-OneRoomS6Fast-v0",
    "MiniWorld-Hallway-v0", "MiniWorld-TMazeLeft-v0", "MiniWorld-TMazeRight-v0",
    "MiniWorld-TMaze-v0", "MiniWorld-YMazeLeft-v0", "MiniWorld-YMazeRight-v0",
    "MiniWorld-YMaze-v0", "MiniWorld-FourRooms-v0", "MiniWorld-ThreeRooms-v0",
    "MiniWorld-Maze-v0", "MiniWorld-MazeS2-v0", "MiniWorld-MazeS3-v0",
    "MiniWorld-WallGap-v0", "MiniWorld-Sidewalk-v0", "MiniWorld-PutNext-v0",
    "MiniWorld-PickupObjects-v0", "MiniWorld-RoomObjects-v0",
    "MiniWorld-CollectHealth-v0", "MiniWorld-Sign-v0",
]


def h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:12]


def static_fingerprint(u) -> dict:
    """Everything the renderer reads that is not an agent/entity POSE."""
    rooms = []
    for r in u.rooms:
        rooms.append((h(r.outline), float(r.wall_height), bool(r.no_ceiling),
                      r.wall_tex.name, r.floor_tex.name, r.ceil_tex.name,
                      h(r.wall_verts), h(r.wall_texcs), h(r.floor_verts), h(r.ceil_verts)))
    ents = []
    for e in u.entities:
        if e is u.agent:
            continue
        d = {"cls": type(e).__name__, "static": bool(e.is_static),
             "radius": round(float(e.radius), 6), "height": round(float(e.height), 6)}
        for attr in ("color", "size", "mesh_name", "tex_name"):
            if hasattr(e, attr):
                v = getattr(e, attr)
                d[attr] = h(v) if isinstance(v, np.ndarray) else str(v)
        ents.append(tuple(sorted(d.items())))
    return {"n_rooms": len(u.rooms), "rooms": tuple(rooms), "n_ents": len(ents),
            "ents": tuple(ents), "sky": h(u.sky_color), "light_pos": h(u.light_pos),
            "light_color": h(u.light_color), "light_ambient": h(u.light_ambient)}


def pose_vector(u) -> np.ndarray:
    v = [*u.agent.pos, u.agent.dir]
    for e in u.entities:
        if e is u.agent:
            continue
        v += [*e.pos, e.dir]
    return np.array(v, np.float64)


def write_poses(u, v: np.ndarray) -> None:
    u.agent.pos = np.array(v[0:3], np.float64)
    u.agent.dir = float(v[3])
    i = 4
    for e in u.entities:
        if e is u.agent:
            continue
        e.pos = np.array(v[i:i + 3], np.float64)
        e.dir = float(v[i + 3])
        i += 4


def main():
    print(f"{'id':38s} {'G3 static':>10s} {'n_rooms':>8s} {'n_ents':>7s} {'pose_dim':>9s}  what varies")
    print("-" * 130)
    passed = []
    for eid in IDS:
        try:
            env = gym.make(eid, render_mode="rgb_array")
        except Exception as e:                                   # noqa: BLE001
            print(f"{eid:38s} {'BUILD-FAIL':>10s}  {type(e).__name__}: {e}")
            continue
        u = env.unwrapped
        fps, dims, nr, ne = [], set(), set(), set()
        for s in range(8):
            u.reset(seed=s)
            fps.append(static_fingerprint(u))
            dims.add(len(pose_vector(u)))
            nr.add(len(u.rooms))
            ne.add(len(u.entities) - 1)
        same = all(f == fps[0] for f in fps)
        why = []
        if not same:
            for k in fps[0]:
                if len({str(f[k]) for f in fps}) > 1:
                    why.append(k)
        if len(dims) > 1:
            why.append(f"POSE_DIM varies {sorted(dims)}")
        print(f"{eid:38s} {'PASS' if same else 'FAIL':>10s} {str(sorted(nr)):>8s} "
              f"{str(sorted(ne)):>7s} {str(sorted(dims)):>9s}  {','.join(why)}")
        if same and len(dims) == 1:
            passed.append(eid)
        env.close()

    print("\n" + "=" * 130)
    print("G1 purity on the ids that passed G3: write episode-A poses into an env re-reset at seed B")
    print("=" * 130)
    print(f"{'id':38s} {'max/255':>8s} {'%pix':>7s}   verdict")
    for eid in passed:
        env = gym.make(eid, render_mode="rgb_array")
        u = env.unwrapped
        worst, worstfrac = 0, 0.0
        for s in range(6):
            u.reset(seed=100 + s)
            v = pose_vector(u)
            u._render_static()
            a = u.render()
            # a genuinely different internal history: a fresh episode, then the poses written back
            u.reset(seed=900 + s)
            write_poses(u, v)
            u._render_static()
            b = u.render()
            d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
            if int(d.max()) > worst:
                worst = int(d.max())
            worstfrac = max(worstfrac, float((d > 0).mean()))
        print(f"{eid:38s} {worst:>8d} {100*worstfrac:>6.2f}%   "
              f"{'PURE' if worst == 0 else 'IMPURE'}")
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
