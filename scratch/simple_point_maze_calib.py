"""Recover ``swm/SimplePointMaze-v0``'s drawing constants by measurement, not from matplotlib's source.

The gates already passed (see ``simple_point_maze_gates.py``): the frame is a pure function of
``(agent position, variation)``, 0/255 across a repeat render, a rollout and a second instance. What
is left is the part that decides whether a GPU port can be exact — the data-to-pixel map, the axes
spine, and the rule Agg uses to composite the ``alpha=0.5`` goal disc.

Each is measured by constructing a frame in which only the quantity of interest can be responsible
for a pixel.
"""
from __future__ import annotations

import json

import numpy as np

WHITE = np.array([255, 255, 255], np.uint8)


def build():
    import gymnasium
    import stable_worldmodel  # noqa: F401
    e = gymnasium.make("swm/SimplePointMaze-v0")
    u = e.unwrapped
    e.reset(seed=0, options={"variation": []})
    vs = u.variation_space
    vs["background"]["color"].set_value(WHITE.copy())
    vs["walls"]["color"].set_value(WHITE.copy())
    vs["agent"]["radius"].set_value(np.array(0.05, np.float32))
    vs["agent"]["position"].set_value(np.array([4.9, 4.9], np.float32))
    vs["goal"]["radius"].set_value(np.array(0.05, np.float32))
    vs["goal"]["position"].set_value(np.array([4.9, 0.1], np.float32))
    return e, u, vs


def render(u, pos) -> np.ndarray:
    u.state = np.asarray(pos, np.float32)
    return np.ascontiguousarray(u.render("rgb_array"))


def main() -> None:
    out = {}
    e, u, vs = build()

    # ---------------- the data -> pixel map, one wall at a time.
    # walls.number cannot go below 4, and every wall shares one colour, so the others are parked in
    # a known corner and excluded from the measurement by their column range rather than by colour.
    shape = np.full((6, 2), 0.5, np.float32)
    pos = np.zeros((6, 2), np.float32)
    rects = {"x1_2_y1_2": (1.0, 1.0, 1.0, 1.0), "x2_3_y3_35": (2.0, 3.0, 1.0, 0.5),
             "x3_4_y0_1": (3.0, 0.0, 1.0, 1.0), "x4_45_y45_5": (4.0, 4.5, 0.5, 0.5)}
    boxes = {}
    for key, (x0, y0, w, h) in rects.items():
        sh, ps = shape.copy(), pos.copy()
        sh[0] = (w, h)
        ps[0] = (x0, y0)
        vs["walls"]["shape"].set_value(sh)
        vs["walls"]["positions"].set_value(ps)
        vs["walls"]["color"].set_value(np.array([0, 0, 255], np.uint8))
        f = render(u, [4.9, 4.9])
        m = (f[..., 2] > 200) & (f[..., 0] < 60)
        m[:, :60] = False                      # the parked walls live in x < 0.5 -> cols < 50
        m[440:, :] = False                     # ... and y < 0.5 -> rows > 450
        ys, xs = np.nonzero(m)
        boxes[key] = {"data": [x0, y0, w, h], "rows": [int(ys.min()), int(ys.max())],
                      "cols": [int(xs.min()), int(xs.max())], "n_px": int(m.sum())}
        vs["walls"]["color"].set_value(WHITE.copy())
    out["wall_boxes"] = boxes
    a, b = boxes["x1_2_y1_2"], boxes["x2_3_y3_35"]
    out["px_per_unit_x"] = (b["cols"][0] - a["cols"][0]) / (b["data"][0] - a["data"][0])
    out["px_per_unit_y"] = (a["rows"][1] - b["rows"][1]) / (b["data"][1] - a["data"][1])
    out["col_at_x0"] = a["cols"][0] - a["data"][0] * out["px_per_unit_x"]
    out["row_at_y0"] = a["rows"][1] + a["data"][1] * out["px_per_unit_y"]

    # ---------------- the axes spine: a frame with nothing in it
    vs["walls"]["color"].set_value(WHITE.copy())
    blank = render(u, [4.9, 4.9])
    edge = {}
    for name, line in (("row0", blank[0]), ("row1", blank[1]), ("row2", blank[2]),
                       ("row499", blank[499]), ("col0", blank[:, 0]), ("col499", blank[:, 499])):
        vals, cnt = np.unique(line.reshape(-1, 3), axis=0, return_counts=True)
        edge[name] = [[[int(v) for v in vals[i]], int(cnt[i])] for i in np.argsort(-cnt)[:4]]
    out["edges"] = edge
    non_white = (blank != 255).any(-1)
    ys, xs = np.nonzero(non_white)
    out["blank_non_white_px"] = int(non_white.sum())
    out["blank_non_white_bbox"] = ([int(ys.min()), int(ys.max()),
                                    int(xs.min()), int(xs.max())] if non_white.any() else None)

    # ---------------- the alpha=0.5 goal blend, over several backgrounds
    vs["goal"]["radius"].set_value(np.array(0.5, np.float32))
    vs["goal"]["position"].set_value(np.array([4.0, 1.0], np.float32))
    blends = []
    for bg in ([255, 255, 255], [200, 100, 40], [0, 0, 0], [17, 200, 91]):
        for gc in ([0, 255, 0], [255, 0, 0], [10, 20, 30]):
            vs["background"]["color"].set_value(np.array(bg, np.uint8))
            vs["goal"]["color"].set_value(np.array(gc, np.uint8))
            f = render(u, [4.9, 4.9])
            centre = [int(v) for v in f[int(round(out["row_at_y0"] - 1.0 * out["px_per_unit_y"])),
                                        int(round(out["col_at_x0"] + 4.0 * out["px_per_unit_x"]))]]
            blends.append({"bg": bg, "goal": gc, "blended": centre,
                           "floor_half": [int(np.floor(0.5 * (g + b))) for g, b in zip(gc, bg)],
                           "agg_128": [int(np.floor(b + (g - b) * 128.0 / 255.0))
                                       for g, b in zip(gc, bg)],
                           "agg_127": [int(np.floor(b + (g - b) * 127.0 / 255.0))
                                       for g, b in zip(gc, bg)]})
    out["goal_blends"] = blends
    out["floor_half_matches"] = sum(x["blended"] == x["floor_half"] for x in blends)
    out["agg_128_matches"] = sum(x["blended"] == x["agg_128"] for x in blends)
    out["agg_127_matches"] = sum(x["blended"] == x["agg_127"] for x in blends)

    # ---------------- the agent disc: pixel radius for a known data radius
    vs["background"]["color"].set_value(WHITE.copy())
    vs["goal"]["color"].set_value(np.array([0, 255, 0], np.uint8))
    # radius BEFORE position: the position leaf's constrain_fn evaluates the collision test with the
    # CURRENT radius, so shrinking the goal after moving it rejects placements that are legal.
    vs["goal"]["radius"].set_value(np.array(0.05, np.float32))
    vs["goal"]["position"].set_value(np.array([1.0, 4.0], np.float32))
    discs = {}
    for r in (0.1, 0.3, 0.5):
        vs["agent"]["radius"].set_value(np.array(r, np.float32))
        vs["agent"]["color"].set_value(np.array([255, 0, 0], np.uint8))
        f = render(u, [2.5, 2.5])
        m = (f[..., 0] > 200) & (f[..., 1] < 60) & (f[..., 2] < 60)
        ys, xs = np.nonzero(m)
        discs[str(r)] = {"n_px": int(m.sum()), "rows": [int(ys.min()), int(ys.max())],
                         "cols": [int(xs.min()), int(xs.max())],
                         "expected_px_area": float(np.pi * (r * 100.0) ** 2)}
    out["agent_discs"] = discs

    print(json.dumps(out, indent=1, default=str))
    with open("/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/spm_calib.json", "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
