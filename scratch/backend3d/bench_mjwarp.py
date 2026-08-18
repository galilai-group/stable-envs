"""Benchmark the MJWarp (mujoco_warp) built-in batch renderer.

Integration surface under test, which is exactly the one stable-envs needs:

    model + (B, nq) qpos  ->  (B, H, W, 4) uint8 on device, zero copy

The render context owns a packed uint32 RGB buffer of shape (nworld, npixel); a
torch view of it costs nothing, so the measured number is the real cost of going
from a state vector to a tensor a network can consume.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


def build(xml: str, batch: int, res: int, cam: int, **rc_kwargs):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=batch, nconmax=batch * 8, njmax=batch * 8)
    rc = mjw.create_render_context(
        mjm,
        nworld=batch,
        cam_res=(res, res),
        render_rgb=True,
        render_depth=False,
        cam_active=[i == cam for i in range(mjm.ncam)],
        **rc_kwargs,
    )
    return mjm, mjd, m, d, rc


def qpos_batch(mjm, mjd, batch, seed=0):
    """A batch of plausible states: reference qpos plus noise on the free/root dofs."""
    rng = np.random.default_rng(seed)
    q = np.tile(np.asarray(mjd.qpos, dtype=np.float32), (batch, 1))
    q[:, :2] += rng.uniform(-4.0, 4.0, size=(batch, 2)).astype(np.float32)
    if mjm.nq > 7:
        q[:, 7:] += rng.uniform(-0.3, 0.3, size=(batch, mjm.nq - 7)).astype(np.float32)
    return q


def rgb_torch(rc, batch, res):
    """Zero-copy (B, H, W, 4) uint8 torch view of the packed uint32 RGB buffer."""
    import torch

    t = wp.to_torch(rc.rgb_data)  # (nworld, npixel) int32/uint32
    # pack_rgba_to_uint32 is (a<<24)|(r<<16)|(g<<8)|b, so the byte view is BGRA
    return t.view(torch.uint8).reshape(batch, res, res, 4)


def run(xml, batch, res, cam, iters=30, warmup=5, graph=True, label="", **rc_kwargs):
    mjm, mjd, m, d, rc = build(xml, batch, res, cam, **rc_kwargs)
    q = qpos_batch(mjm, mjd, batch)
    d.qpos.assign(wp.array(q, dtype=wp.float32, ndim=2))

    def step():
        mjw.kinematics(m, d)
        mjw.camlight(m, d)
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)

    for _ in range(warmup):
        step()
    wp.synchronize()

    g = None
    if graph:
        with wp.ScopedCapture() as cap:
            step()
        g = cap.graph

    t0 = time.perf_counter()
    for _ in range(iters):
        if g is not None:
            wp.capture_launch(g)
        else:
            step()
    wp.synchronize()
    dt = time.perf_counter() - t0

    fps = batch * iters / dt
    out = {
        "backend": "mjwarp",
        "scene": os.path.basename(xml),
        "label": label,
        "batch": batch,
        "res": res,
        "graph": graph,
        "ms_per_batch": 1e3 * dt / iters,
        "fps": fps,
        "mpix_s": fps * res * res / 1e6,
        **{k: v for k, v in rc_kwargs.items()},
    }
    print(json.dumps(out), flush=True)
    return out, (mjm, mjd, m, d, rc)


def dump_image(xml, res, cam, path, **rc_kwargs):
    """One frame at the model's reference qpos, for the fidelity comparison."""
    import imageio.v2 as imageio

    mjm, mjd, m, d, rc = build(xml, 1, res, cam, **rc_kwargs)
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    a = rc.rgb_data.numpy().view(np.uint8).reshape(1, res, res, 4)
    imageio.imwrite(path, a[0][..., [2, 1, 0]])
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml,cube_single.xml,humanoidmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batches", default="32,128,512,1024,2048,4096")
    ap.add_argument("--out", default="")
    ap.add_argument("--images", default="")
    args = ap.parse_args()

    wp.init()
    print(wp.get_device(), flush=True)

    rows = []
    for scene in args.scenes.split(","):
        xml = os.path.join(args.dir, scene)
        if not os.path.exists(xml):
            print(f"skip missing {xml}")
            continue
        for b in [int(x) for x in args.batches.split(",")]:
            for label, kw in [
                ("textures+shadows_off", dict(use_textures=True, use_shadows=False)),
                ("notex", dict(use_textures=False, use_shadows=False)),
                ("shadows", dict(use_textures=True, use_shadows=True)),
            ]:
                try:
                    r, _ = run(xml, b, args.res, 0, label=label, **kw)
                    rows.append(r)
                except Exception as e:
                    print(json.dumps({"scene": scene, "batch": b, "label": label,
                                      "error": f"{type(e).__name__}: {e}"}), flush=True)
    if args.images:
        os.makedirs(args.images, exist_ok=True)
        for scene in args.scenes.split(","):
            xml = os.path.join(args.dir, scene)
            if os.path.exists(xml):
                dump_image(xml, args.res, 0, os.path.join(args.images, scene.replace(".xml", "_mjwarp.png")))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
