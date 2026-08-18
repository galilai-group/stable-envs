"""Does every qpos entry move a pixel? The conformance suite requires it; the agent is ~6 px wide.

``check_state_dims_matter`` perturbs each state dimension by 25 and fails the env if the frame is
byte-identical. For pointmaze that is trivially satisfied (the agent leaves the frame). For the ant and
the humanoid at the whole-maze top-down camera the body spans a handful of pixels, and 25 radians on a
hinge is 25 - 8*pi = 0.87 rad through the wrap, i.e. ~50 degrees on a limb whose tip is a fraction of a
pixel away -- so a genuinely live dimension can read as dead.

Measured here on MuJoCo itself (the reference, so the answer is about the ENV and not about our
renderer), over the same batch of 16 states the suite would use, at several resolutions.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from general_check import make2  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"


def free_cells(maze_map):
    return np.argwhere(np.asarray(maze_map) == 0)


def sample_states(u, batch, rng):
    """Free-cell positions + the env's own init pose, mildly perturbed. Agent always visible."""
    m = u.model
    cells = free_cells(u.maze_map)
    pick = cells[rng.integers(0, len(cells), batch)]
    xy = np.stack([pick[:, 1] * u._maze_unit - u._offset_x,
                   pick[:, 0] * u._maze_unit - u._offset_y], 1).astype(np.float64)
    xy += rng.uniform(-1.0, 1.0, xy.shape)
    q = np.tile(np.asarray(u.init_qpos, np.float64), (batch, 1))
    q[:, :2] = xy
    if m.nq > 2:
        q[:, 7:] += rng.uniform(-0.15, 0.15, (batch, m.nq - 7))
    return q.astype(np.float32)


def run(loco, size, img, eps=25.0, batch=16, seed=0):
    env, u = make2(loco, size, img)
    m = u.model
    rng = np.random.default_rng(seed)
    st = sample_states(u, batch, rng)

    def frames(states):
        out = []
        for q in states:
            u.set_state(np.asarray(q, np.float64), np.zeros(m.nv))
            out.append(u.render().copy())
        return np.stack(out)

    base = frames(st)
    dead, moved = [], {}
    for d in range(m.nq):
        p = st.copy()
        p[:, d] = p[:, d] + eps
        f = frames(p)
        n = int((np.abs(f.astype(np.int16) - base.astype(np.int16)).max(-1) > 0).sum())
        moved[d] = n
        if n == 0:
            dead.append(d)
    env.close()
    return dict(env=f"{loco}maze-{size}", img=img, nq=int(m.nq), dead_dims=dead,
                min_pixels_moved=int(min(moved.values())), pixels_moved=moved)


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "dimcheck.log"), "w")
    allr = []
    for loco, size, img in [("point", "medium", 224), ("ant", "large", 224), ("ant", "large", 384),
                            ("humanoid", "large", 224), ("humanoid", "large", 384),
                            ("humanoid", "giant", 224)]:
        try:
            r = run(loco, size, img)
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL {loco}-{size}-{img}\n{traceback.format_exc()}\n")
            log.flush()
            continue
        allr.append(r)
        log.write(json.dumps(r) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in r.items() if k != "pixels_moved"}), flush=True)
    log.close()
    with open(os.path.join(OUT, "dimcheck.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
