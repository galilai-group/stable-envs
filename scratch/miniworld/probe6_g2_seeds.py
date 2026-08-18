"""Probe 6: G2 compression, and a WIDER seed sweep on the ids whose layout is procedural.

Probe 2 sampled 8 seeds and reported MazeS2's static fingerprint as constant while its G1 purity
check (seeds 100-105 against 900-905) failed at 169/255. Only one of those can be right. 64 seeds
settles it, and the answer decides whether "8 seeds" was ever an adequate sample for this gate.
"""
from __future__ import annotations

import sys

import numpy as np
import pyglet

pyglet.options["headless"] = True

import gymnasium as gym  # noqa: E402
import miniworld  # noqa: E402,F401

sys.path.insert(0, "scratch/miniworld")
from probe2_gates import pose_vector, static_fingerprint  # noqa: E402

PROCEDURAL = ["MiniWorld-MazeS2-v0", "MiniWorld-MazeS3-v0", "MiniWorld-Maze-v0",
              "MiniWorld-PutNext-v0", "MiniWorld-PickupObjects-v0", "MiniWorld-RoomObjects-v0"]
CLEAN = ["MiniWorld-OneRoom-v0", "MiniWorld-OneRoomS6-v0", "MiniWorld-Hallway-v0",
         "MiniWorld-TMazeLeft-v0", "MiniWorld-TMazeRight-v0", "MiniWorld-TMaze-v0",
         "MiniWorld-YMazeLeft-v0", "MiniWorld-YMazeRight-v0", "MiniWorld-YMaze-v0",
         "MiniWorld-FourRooms-v0", "MiniWorld-ThreeRooms-v0", "MiniWorld-WallGap-v0",
         "MiniWorld-Sidewalk-v0", "MiniWorld-CollectHealth-v0", "MiniWorld-Sign-v0"]


def sweep(eid, n):
    env = gym.make(eid, render_mode="rgb_array")
    u = env.unwrapped
    fps, dims = set(), set()
    for s in range(n):
        u.reset(seed=s)
        fps.add(str(static_fingerprint(u)))
        dims.add(len(pose_vector(u)))
    env.close()
    return len(fps), sorted(dims)


def main():
    print("distinct static scenes over N seeds  (1 = the decoder is a constant of the class)")
    print(f"{'id':34s} {'N=8':>6s} {'N=64':>6s} {'pose dims':>12s}")
    for eid in PROCEDURAL + CLEAN:
        a, _ = sweep(eid, 8)
        b, d = sweep(eid, 64)
        print(f"{eid:34s} {a:>6d} {b:>6d} {str(d):>12s}")

    print()
    print("G2 compression: render-relevant state against the frame it produces")
    print(f"{'id':34s} {'dims':>5s} {'state B':>8s} {'obs 80x60 B':>12s} {'ratio':>10s} "
          f"{'render 800x600':>15s} {'ratio':>10s}")
    for eid in CLEAN:
        env = gym.make(eid, render_mode="rgb_array")
        u = env.unwrapped
        u.reset(seed=0)
        n = len(pose_vector(u))
        sb = n * 4
        print(f"{eid:34s} {n:>5d} {sb:>8d} {14400:>12d} {sb/14400:>10.5f} "
              f"{1440000:>15d} {sb/1440000:>10.7f}")
        env.close()


if __name__ == "__main__":
    main()
