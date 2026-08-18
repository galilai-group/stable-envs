"""G4, re-measured: what is the 4.03 s regeneration actually made of?

The original ``probe_regen.py`` reported one end-to-end number (compile 2.03 s + renderer init
0.95 s + render loop 1.05 s) and a 34.7x loss to JPEG decode. Two objections to that baseline are
worth taking seriously:

  1. Regenerating from *stored* states needs no physics. ``probe_regen.py`` called ``mj_forward``,
     which runs the full dynamics pipeline; a state->frame decoder only needs
     ``mj_kinematics`` + ``mj_camlight``. If that is a large share of the loop the render side was
     under-reported.
  2. The compile is per *episode*, not per frame, so it amortizes over anything that re-renders the
     same episode (more cameras, more resolutions, more styles) and over any batch of episodes that
     share a compiled kitchen.

So the loop is decomposed stage by stage, and the compile is priced two ways: as an XML compile and
as a binary ``.mjb`` load, which is the only way to make it cheap -- and which costs storage.

Run in any env with mujoco + EGL. The pinned scene XML carries absolute asset paths, so it loads
without robosuite.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _stats(xs):
    xs = np.asarray(xs, float)
    return {"mean": float(xs.mean()), "median": float(np.median(xs)),
            "min": float(xs.min()), "max": float(xs.max()), "n": int(xs.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(HERE, "out", "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "out", "qpos_bank.npy"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--tag", default="egl")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    from PIL import Image

    rec = {"res": args.res, "T": args.T, "mujoco": mujoco.__version__, "cam": args.cam}

    # ---- compile: cold (first in process, cold-ish page cache) then warm -----------------------
    t0 = time.perf_counter()
    m = mujoco.MjModel.from_xml_path(args.xml)
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(3):
        t0 = time.perf_counter()
        mujoco.MjModel.from_xml_path(args.xml)
        warm.append(time.perf_counter() - t0)
    rec["compile_xml_s"] = {"cold": cold, "warm": _stats(warm)}

    # ---- compile, avoided: store the compiled binary instead of the XML -----------------------
    # half a gigabyte per episode: written to a temp dir, never into the repo
    with tempfile.TemporaryDirectory() as td:
        mjb = os.path.join(td, "pinned_scene.mjb")
        mujoco.mj_saveModel(m, mjb, None)
        rec["mjb_bytes"] = os.path.getsize(mjb)
        raw = open(mjb, "rb").read()
        t0 = time.perf_counter()
        gz = gzip.compress(raw, 1)
        rec["mjb_gzip_s"] = time.perf_counter() - t0
        rec["mjb_gzip_bytes"] = len(gz)
        del raw, gz
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            mujoco.MjModel.from_binary_path(mjb)
            ts.append(time.perf_counter() - t0)
        rec["load_mjb_s"] = _stats(ts)
    rec["xml_bytes"] = os.path.getsize(args.xml)
    rec["xml_gzip_bytes"] = len(gzip.compress(open(args.xml, "rb").read()))

    d = mujoco.MjData(m)
    bank = np.load(args.qpos)
    qpos = bank[np.arange(args.T) % len(bank)]
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if cam_id < 0:
        cam_id = 0

    # robosuite draws group 1 only.
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1

    # ---- renderer construction: GL context + texture upload -----------------------------------
    t0 = time.perf_counter()
    r = mujoco.Renderer(m, height=args.res, width=args.res)
    r.update_scene(d, camera=cam_id, scene_option=opt)
    r.render()
    rec["renderer_init_s"] = time.perf_counter() - t0

    # ---- per-stage cost of one regenerated frame ----------------------------------------------
    N = min(args.T, 200)
    t_fwd, t_kin, t_scene, t_render = [], [], [], []
    for i in range(N):
        d.qpos[:] = qpos[i]
        t0 = time.perf_counter(); mujoco.mj_forward(m, d); t_fwd.append(time.perf_counter() - t0)
        d.qpos[:] = qpos[i]
        t0 = time.perf_counter()
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        t_kin.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); r.update_scene(d, camera=cam_id, scene_option=opt); t_scene.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); r.render(); t_render.append(time.perf_counter() - t0)
    rec["per_frame_ms"] = {
        "mj_forward": _stats(np.array(t_fwd) * 1e3),
        "mj_kinematics+camlight": _stats(np.array(t_kin) * 1e3),
        "update_scene": _stats(np.array(t_scene) * 1e3),
        "render": _stats(np.array(t_render) * 1e3),
    }

    # ---- full loops: as probe_regen did it, and as a state->frame decoder should ---------------
    frames = np.empty((args.T, args.res, args.res, 3), np.uint8)
    t0 = time.perf_counter()
    for i in range(args.T):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam_id, scene_option=opt)
        frames[i] = r.render()
    rec["loop_forward_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(args.T):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        r.update_scene(d, camera=cam_id, scene_option=opt)
        frames[i] = r.render()
    rec["loop_kinematics_s"] = time.perf_counter() - t0
    rec["loop_kinematics_fps"] = args.T / rec["loop_kinematics_s"]

    # ---- the alternative: decode stored JPEGs, one core ----------------------------------------
    bufs = []
    for f in frames:
        b = io.BytesIO()
        Image.fromarray(f).save(b, format="JPEG", quality=90)
        bufs.append(b.getvalue())
    rec["jpeg_MB_per_episode"] = sum(len(b) for b in bufs) / 1e6
    for _ in range(2):
        t0 = time.perf_counter()
        for b in bufs:
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        dec = time.perf_counter() - t0
    rec["jpeg_decode_s"] = dec
    rec["jpeg_decode_fps"] = args.T / dec

    # ---- what a regenerated episode costs, decomposed -------------------------------------------
    setup = rec["compile_xml_s"]["warm"]["mean"] + rec["renderer_init_s"]
    rec["episode_s"] = {
        "setup_compile+ctx": setup,
        "render_loop": rec["loop_kinematics_s"],
        "total": setup + rec["loop_kinematics_s"],
        "setup_share": setup / (setup + rec["loop_kinematics_s"]),
    }
    rec["vs_decode"] = {
        "end_to_end_x": rec["episode_s"]["total"] / dec,
        "loop_only_x": rec["loop_kinematics_s"] / dec,
    }

    print(json.dumps(rec, indent=2))
    with open(os.path.join(args.out, f"g4_{args.tag}.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
