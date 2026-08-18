"""Benchmark NVIDIA Warp's own renderer (warp.render.OpenGLRenderer) in headless tiled mode.

This is candidate 3. Warp ships one rasterizer, and it is an on-screen OpenGL
scene-graph renderer with a `setup_tiled_rendering` mode that packs N views into a
single framebuffer, plus `get_pixels(..., use_uint8=True)` which lands the result in a
wp.array on device. That combination is the only way it can act as a batch backend.

The structural limit shows up immediately: every "batch element" is a tile inside one
window, so the batch is capped by framebuffer area / tile area, and every tile is drawn
by one serialized GL pass over the shared scene graph.

Geometry is fed through Warp's own shape API (register_shape / add_shape_instance), so
this file also has to translate MuJoCo geoms itself -- Warp knows nothing about MuJoCo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("PYGLET_HEADLESS", "1")
import pyglet  # noqa: E402

pyglet.options["headless"] = True

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
import warp.render  # noqa: E402

from mjgeom import tessellate  # noqa: E402


def build_renderer(mjm, mjd, res, ntiles, screen=(4096, 4096)):
    ncols = max(1, screen[0] // res)
    nrows = int(np.ceil(ntiles / ncols))
    if nrows * res > screen[1]:
        raise RuntimeError(
            f"tiled mode cannot fit {ntiles} tiles of {res}px in a {screen[0]}x{screen[1]} "
            f"framebuffer (max {ncols * (screen[1] // res)})"
        )
    r = warp.render.OpenGLRenderer(
        screen_width=screen[0], screen_height=screen[1], headless=True,
        up_axis="Z", draw_grid=False, draw_axis=False, vsync=False,
    )
    soup = tessellate(mjm)
    # one Warp mesh shape per geom, instanced once per tile
    inst_per_tile = []
    verts, faces, vg = soup["vert"], soup["face"], soup["vgeom"]
    shapes = {}
    for g in np.unique(vg):
        sel = vg == g
        vmap = -np.ones(len(vg), dtype=np.int64)
        vmap[sel] = np.arange(sel.sum())
        fsel = sel[faces[:, 0]]
        col = tuple(float(x) for x in soup["rgb"][sel][0])
        shapes[int(g)] = r.register_shape(
            f"geom{int(g)}",
            verts[sel].astype(np.float32),
            vmap[faces[fsel]].astype(np.int32).flatten(),
            color1=col, color2=col,
        )
    instances = []
    for t in range(ntiles):
        ids = []
        for g, s in shapes.items():
            name = f"t{t}_g{g}"
            ids.append(r.add_shape_instance(name, s, body=-1,
                                            pos=tuple(float(x) for x in mjd.geom_xpos[g]),
                                            rot=(0.0, 0.0, 0.0, 1.0)))
        instances.append(ids)
        inst_per_tile.append(ids)
    r.allocate_shape_instances()
    r.setup_tiled_rendering(instances, tile_width=res, tile_height=res)
    return r, inst_per_tile


def run(xml, ntiles, res, iters=10):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    r, _ = build_renderer(mjm, mjd, res, ntiles)
    buf = wp.zeros((ntiles, res, res, 3), dtype=wp.uint8)

    def frame():
        r.begin_frame(0.0)
        r.end_frame()
        r.get_pixels(buf, split_up_tiles=True, mode="rgb", use_uint8=True)

    frame()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        frame()
    wp.synchronize()
    dt = time.perf_counter() - t0
    fps = ntiles * iters / dt
    row = {"backend": "warp_opengl_tiled", "scene": os.path.basename(xml), "batch": ntiles,
           "res": res, "ms_per_batch": 1e3 * dt / iters, "fps": fps,
           "mpix_s": fps * res * res / 1e6}
    print(json.dumps(row), flush=True)
    r.close()
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="antmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        try:
            rows.append(run(os.path.join(args.dir, args.scene), b, args.res))
        except Exception as e:
            print(json.dumps({"backend": "warp_opengl_tiled", "batch": b,
                              "error": f"{type(e).__name__}: {e}"}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
