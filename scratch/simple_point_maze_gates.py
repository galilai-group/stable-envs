"""Run the four gates on ``swm/SimplePointMaze-v0`` before writing any renderer for it.

The env is a 5x5 world drawn with **matplotlib**: an Agg figure, ``tight_layout(pad=0)``, ``Rectangle``
patches for walls, an ``alpha=0.5`` ``Circle`` for the goal and an opaque one for the agent, read back
through ``canvas.tostring_argb()``. That is a different kind of reference from every other env in this
repo, so the question is not "can we draw circles" but "is the decoder a constant of the family, and is
the data->pixel map recoverable exactly".

Measured here, in the cost order the criterion prescribes:

  G3  is the decoder stable?  render twice, and after a rollout; compare.
  G1  is the frame a function of the state?  drive two histories to one state and diff.
  --  the affine map from data coordinates to pixels, and the spine, recovered by measurement:
      paint a wall of known extent and read off which pixels it covers.
  G2  compression: 2 floats against 500*500*3.
  G4  economics: official frames/s.
"""
from __future__ import annotations

import json
import time

import numpy as np


def env():
    import gymnasium
    import stable_worldmodel  # noqa: F401
    return gymnasium.make("swm/SimplePointMaze-v0")


def render_at(u, pos) -> np.ndarray:
    u.state = np.asarray(pos, np.float32)
    return np.ascontiguousarray(u.render("rgb_array"))


def main() -> None:
    out = {}
    e = env()
    u = e.unwrapped
    e.reset(seed=0, options={"variation": []})

    # ---- G3: is the decoder the same object twice, and after a rollout?
    a = render_at(u, [1.0, 1.0])
    b = render_at(u, [1.0, 1.0])
    out["G3_same_state_twice"] = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())
    for _ in range(20):
        e.step(np.array([0.1, 0.05], np.float32))
    c = render_at(u, [1.0, 1.0])
    out["G3_after_a_rollout"] = int(np.abs(a.astype(np.int16) - c.astype(np.int16)).max())
    e2 = env()
    e2.reset(seed=0, options={"variation": []})
    d = render_at(e2.unwrapped, [1.0, 1.0])
    out["G1_second_instance"] = int(np.abs(a.astype(np.int16) - d.astype(np.int16)).max())
    out["frame_shape"] = list(a.shape)

    # ---- the palette actually on screen
    cols, counts = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    out["distinct_colors"] = int(cols.shape[0])
    out["top_colors"] = [[[int(v) for v in cols[i]], int(counts[i])] for i in order[:10]]

    # ---- the data -> pixel affine map, recovered by painting a wall of known extent.
    # walls.number is Discrete(start=4), so walls cannot be removed; they are made invisible by
    # painting them the background colour, and then ONE of them is moved to a known rectangle.
    vs = u.variation_space
    vs["background"]["color"].set_value(np.array([255, 255, 255], np.uint8))
    vs["walls"]["color"].set_value(np.array([255, 255, 255], np.uint8))
    vs["agent"]["radius"].set_value(np.array(0.05, np.float32))
    vs["agent"]["position"].set_value(np.array([4.9, 4.9], np.float32))
    vs["goal"]["radius"].set_value(np.array(0.05, np.float32))
    vs["goal"]["position"].set_value(np.array([4.9, 0.1], np.float32))
    blank = render_at(u, [4.9, 4.9])
    out["blank_distinct_colors"] = int(np.unique(blank.reshape(-1, 3), axis=0).shape[0])
    # the figure background outside the axes, and the axes background
    out["corner_pixel"] = [int(v) for v in blank[0, 0]]
    out["centre_pixel"] = [int(v) for v in blank[250, 250]]

    n = int(vs["walls"]["number"].value)
    shape = np.array(vs["walls"]["shape"].value, np.float32).copy()
    pos = np.array(vs["walls"]["positions"].value, np.float32).copy()
    shape[:] = 0.5
    pos[:] = 0.0                                   # park the others in the bottom-left corner
    probes = {"unit_1_2": (1.0, 1.0, 1.0, 1.0), "unit_2_3": (2.0, 3.0, 1.0, 0.5)}
    boxes = {}
    for key, (x0, y0, w, h) in probes.items():
        sh, ps = shape.copy(), pos.copy()
        sh[0] = (w, h)
        ps[0] = (x0, y0)
        vs["walls"]["shape"].set_value(sh)
        vs["walls"]["positions"].set_value(ps)
        vs["walls"]["color"].set_value(np.array([0, 0, 255], np.uint8))
        f = render_at(u, [4.9, 4.9])
        m = (f[..., 2] > 200) & (f[..., 0] < 60)
        ys, xs = np.nonzero(m)
        boxes[key] = {"rows": [int(ys.min()), int(ys.max())], "cols": [int(xs.min()), int(xs.max())],
                      "n_px": int(m.sum()), "data_rect": [x0, y0, w, h]}
        vs["walls"]["color"].set_value(np.array([255, 255, 255], np.uint8))
    out["wall_probe"] = boxes
    if len(boxes) == 2:
        p, q = boxes["unit_1_2"], boxes["unit_2_3"]
        # x: data 1..2 -> cols p, data 2..3 -> cols q
        sx = (q["cols"][0] - p["cols"][0]) / 1.0
        out["px_per_unit_x"] = sx
        out["x0_px"] = p["cols"][0] - 1.0 * sx

    # ---- the goal's alpha=0.5 circle over a known background
    vs["goal"]["radius"].set_value(np.array(0.5, np.float32))
    vs["goal"]["position"].set_value(np.array([4.0, 1.0], np.float32))
    vs["goal"]["color"].set_value(np.array([0, 255, 0], np.uint8))
    vs["background"]["color"].set_value(np.array([200, 100, 40], np.uint8))
    f = render_at(u, [4.9, 4.9])
    cols, counts = np.unique(f.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    out["goal_alpha_top_colors"] = [[[int(v) for v in cols[i]], int(counts[i])] for i in order[:6]]

    # ---- G2 and G4
    out["G2_state_bytes"] = 8
    out["G2_frame_bytes"] = int(np.prod(a.shape))
    out["G2_ratio"] = out["G2_state_bytes"] / out["G2_frame_bytes"]
    t0 = time.perf_counter()
    for i in range(30):
        render_at(u, [1.0 + 0.01 * i, 1.0])
    out["G4_official_fps"] = 30 / (time.perf_counter() - t0)

    print(json.dumps(out, indent=1, default=str))
    with open("/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/spm_gates.json", "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
