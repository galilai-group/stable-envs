"""The thesis in one number: is regenerating an episode's frames from state cheaper than storing them?

"Nobody stores image datasets anymore" is a claim about a trade, so it has to be priced on both
sides with the same clock:

  regenerate = compile the episode's MJCF once  +  (set qpos, mj_forward, render) x T
  store      = decode T JPEGs from disk

The compile term is the one that gets forgotten, and for RoboCasa it is not small -- the scene is
rebuilt per episode, so it is paid per episode, not once. Measured here end to end at the same
resolution and episode length, on the same machine.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/pinned_scene.xml")
    ap.add_argument("--qpos", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/qpos_bank.npy")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()

    import mujoco
    from PIL import Image

    rec = {"res": args.res, "T": args.T}

    # ---- compile ------------------------------------------------------------------------------
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        m = mujoco.MjModel.from_xml_path(args.xml)
        ts.append(time.perf_counter() - t0)
    rec["compile_s"] = {"first": ts[0], "mean": float(np.mean(ts)), "all": ts}
    d = mujoco.MjData(m)

    bank = np.load(args.qpos)
    qpos = bank[np.arange(args.T) % len(bank)]

    # ---- renderer construction (GL context + texture upload) ---------------------------------
    t0 = time.perf_counter()
    r = mujoco.Renderer(m, height=args.res, width=args.res)
    r.update_scene(d, camera=0)
    r.render()
    rec["renderer_init_s"] = time.perf_counter() - t0

    # ---- regenerate the episode --------------------------------------------------------------
    frames = np.empty((args.T, args.res, args.res, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(args.T):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=0)
        frames[i] = r.render()
    rec["render_loop_s"] = time.perf_counter() - t0
    rec["render_fps"] = args.T / rec["render_loop_s"]
    rec["regen_total_s"] = rec["compile_s"]["mean"] + rec["renderer_init_s"] + rec["render_loop_s"]

    # ---- the alternative: encode once, decode many -------------------------------------------
    bufs = []
    t0 = time.perf_counter()
    for f in frames:
        b = io.BytesIO()
        Image.fromarray(f).save(b, format="JPEG", quality=90)
        bufs.append(b.getvalue())
    rec["jpeg_encode_s"] = time.perf_counter() - t0
    rec["jpeg_bytes_total"] = int(sum(len(b) for b in bufs))
    rec["jpeg_MB_per_episode"] = rec["jpeg_bytes_total"] / 1e6

    t0 = time.perf_counter()
    for b in bufs:
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    rec["jpeg_decode_s"] = time.perf_counter() - t0
    rec["jpeg_decode_fps"] = args.T / rec["jpeg_decode_s"]

    rec["state_MB_per_episode"] = args.T * m.nq * 4 / 1e6
    rec["scene_xml_gzip_MB"] = len(__import__("gzip").compress(open(args.xml, "rb").read())) / 1e6
    rec["storage_ratio_images_over_state"] = rec["jpeg_MB_per_episode"] / (
        rec["state_MB_per_episode"] + rec["scene_xml_gzip_MB"])
    rec["time_ratio_regen_over_decode"] = rec["regen_total_s"] / rec["jpeg_decode_s"]

    print(json.dumps(rec, indent=2))
    with open(os.path.join(args.out, "regen.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
