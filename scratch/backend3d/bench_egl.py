"""Baseline: MuJoCo's own OpenGL/EGL offscreen renderer.

This is the thing every image dataset in the field was generated with, so it is
the number the GPU backends have to beat, and the reference the fidelity check
compares against.

Two questions:
  1. what does one EGL context actually deliver at 224x224?
  2. do multiple contexts / threads help, or is it serialized?

MuJoCo's Python `Renderer` holds one `MjrContext`; a context is bound to the
thread that made it, so the multi-thread case builds one model+data+context per
worker and renders disjoint slices of the batch.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np


def qpos_batch(mjm, mjd, batch, seed=0):
    rng = np.random.default_rng(seed)
    q = np.tile(np.asarray(mjd.qpos, dtype=np.float64), (batch, 1))
    q[:, :2] += rng.uniform(-4.0, 4.0, size=(batch, 2))
    if mjm.nq > 7:
        q[:, 7:] += rng.uniform(-0.3, 0.3, size=(batch, mjm.nq - 7))
    return q


def render_slice(xml, res, cam, qs, out):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, res)
    mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, res)
    mjd = mujoco.MjData(mjm)
    r = mujoco.Renderer(mjm, height=res, width=res)
    for i, q in enumerate(qs):
        mjd.qpos[:] = q
        mujoco.mj_forward(mjm, mjd)
        r.update_scene(mjd, camera=cam)
        out[i] = r.render()
    r.close()


def run(xml, batch, res, cam, threads=1, label=""):
    import threading

    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    qs = qpos_batch(mjm, mjd, batch)
    out = np.zeros((batch, res, res, 3), dtype=np.uint8)

    # warm up one context so shader/context creation is not in the timing
    render_slice(xml, res, cam, qs[: min(4, batch)], out[: min(4, batch)])

    t0 = time.perf_counter()
    if threads == 1:
        render_slice(xml, res, cam, qs, out)
    else:
        bounds = np.linspace(0, batch, threads + 1).astype(int)
        ts = [
            threading.Thread(target=render_slice, args=(xml, res, cam, qs[a:b], out[a:b]))
            for a, b in zip(bounds[:-1], bounds[1:])
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    dt = time.perf_counter() - t0

    fps = batch / dt
    row = {
        "backend": "mujoco_egl",
        "scene": os.path.basename(xml),
        "label": label or f"threads={threads}",
        "batch": batch,
        "res": res,
        "threads": threads,
        "ms_per_batch": 1e3 * dt,
        "fps": fps,
        "mpix_s": fps * res * res / 1e6,
    }
    print(json.dumps(row), flush=True)
    return row


def dump_image(xml, res, cam, path, qpos=None):
    import imageio.v2 as imageio

    mjm = mujoco.MjModel.from_xml_path(xml)
    mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, res)
    mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, res)
    mjd = mujoco.MjData(mjm)
    if qpos is not None:
        mjd.qpos[:] = qpos
    mujoco.mj_forward(mjm, mjd)
    r = mujoco.Renderer(mjm, height=res, width=res)
    r.update_scene(mjd, camera=cam)
    imageio.imwrite(path, r.render())
    r.close()
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml,cube_single.xml,humanoidmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--out", default="")
    ap.add_argument("--images", default="")
    args = ap.parse_args()

    rows = []
    for scene in args.scenes.split(","):
        xml = os.path.join(args.dir, scene)
        if not os.path.exists(xml):
            continue
        for th in [int(x) for x in args.threads.split(",")]:
            try:
                rows.append(run(xml, args.batch, args.res, 0, threads=th))
            except Exception as e:
                print(json.dumps({"scene": scene, "threads": th,
                                  "error": f"{type(e).__name__}: {e}"}), flush=True)
    if args.images:
        os.makedirs(args.images, exist_ok=True)
        for scene in args.scenes.split(","):
            xml = os.path.join(args.dir, scene)
            if os.path.exists(xml):
                dump_image(xml, args.res, 0, os.path.join(args.images, scene.replace(".xml", "_egl.png")))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
