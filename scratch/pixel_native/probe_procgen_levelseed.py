"""Is a Procgen level's tile grid a function of the level seed alone?

If it is, the 1.6-16 kB grid inside every serialized state never has to be stored: 4 bytes of level seed
regenerate it, and a trajectory costs (level seed, per-step entity list).  Two envs constructed with
different ``rand_seed`` but the same ``start_level`` are stepped identically and their state blobs are
compared -- both the whole blob and the grid block specifically.

Run under SLURM.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from decompose_procgen_state import find_grid


def grid_of(blob):
    g = find_grid(blob)
    if g is None:
        return None
    w, h, n, o, _ = g
    return (w, h, bytes(blob[o:o + 12 + 4 * n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", type=int, default=[0, 1, 7, 42, 1234])
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/procgen_levelseed.json")
    args = ap.parse_args()

    from procgen import ProcgenGym3Env
    from procgen.env import ENV_NAMES

    rows = []
    for name in ENV_NAMES:
        same_grid, same_frame, diff_level_grids = [], [], set()
        for lv in args.levels:
            grids, frames = [], []
            for rs in (0, 999):
                env = ProcgenGym3Env(num=1, env_name=name, rand_seed=rs, num_levels=1,
                                     start_level=lv, distribution_mode="hard")
                blob = env.callmethod("get_state")[0]
                _, ob, _ = env.observe()
                grids.append(grid_of(blob))
                frames.append(ob["rgb"].tobytes())
            same_grid.append(grids[0] is not None and grids[0] == grids[1])
            same_frame.append(frames[0] == frames[1])
            if grids[0] is not None:
                diff_level_grids.add(grids[0][2])
        d = {
            "env_name": name,
            "levels": args.levels,
            "grid_same_across_rand_seed": float(np.mean(same_grid)),
            "frame_same_across_rand_seed": float(np.mean(same_frame)),
            "distinct_grids_across_levels": len(diff_level_grids),
            "n_levels": len(args.levels),
        }
        rows.append(d)
        print(f"{name:<12} grid==f(level_seed): {100*d['grid_same_across_rand_seed']:5.0f}%   "
              f"frame==f(level_seed): {100*d['frame_same_across_rand_seed']:5.0f}%   "
              f"distinct grids over {d['n_levels']} levels: {d['distinct_grids_across_levels']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
