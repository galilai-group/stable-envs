"""If the scene were pinned, how far does MuJoCo's own renderer already get you?

The reduced target for stable-envs on RoboCasa is "one compiled kitchen, batched over qpos". Before
writing a GPU rasterizer to serve that target, it is worth knowing what the incumbent already does,
because the incumbent is the thing a new renderer has to beat. Two numbers decide it:

  * aggregate throughput -- N independent EGL contexts on one GPU, rendering the same pinned model;
  * GPU memory per context -- 390 MB of texture data per scene is the number that says whether
    "thousands of envs" is a memory question or a compute question.

Run with --procs to sweep the number of worker processes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def gpu_mem_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode()
        return float(out.strip().split("\n")[0])
    except Exception:
        return -1.0


def worker(xml_path, res, n_frames, qpos_bank, ret, idx):
    os.environ["MUJOCO_GL"] = "egl"
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=res, width=res)
    cid = 0
    for i in range(5):
        d.qpos[:] = qpos_bank[i % len(qpos_bank)]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cid)
        r.render()
    t0 = time.perf_counter()
    for i in range(n_frames):
        d.qpos[:] = qpos_bank[i % len(qpos_bank)]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cid)
        r.render()
    ret[idx] = time.perf_counter() - t0
    r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    from robocasa.utils.env_utils import create_env

    rec = {"res": args.res}
    mem0 = gpu_mem_mb()

    env = create_env(env_name=args.env_name, robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=args.res, camera_heights=args.res,
                     seed=0, render_onscreen=False)
    env.reset()
    m = env.sim.model._model
    rec["pinned"] = {"nq": int(m.nq), "ngeom": int(m.ngeom), "nmesh": int(m.nmesh),
                     "nmeshface": int(m.nmeshface), "ntex": int(m.ntex),
                     "texdata_MB": float(m.tex_data.size / 1e6),
                     "meshvert_MB": float(m.mesh_vert.size * 4 / 1e6)}

    # freeze this exact kitchen to disk so workers all compile the SAME model
    xml_path = os.path.join(args.out, "pinned_scene.xml")
    xml = env.sim.model.get_xml()
    with open(xml_path, "w") as f:
        f.write(xml)
    rec["pinned"]["xml_bytes"] = len(xml)

    # a bank of reachable qpos, from a real rollout (so the batch renders plausible states)
    bank = []
    for _ in range(32):
        env.step(np.random.uniform(-0.3, 0.3, env.action_dim))
        bank.append(env.sim.data.qpos.copy())
    bank = np.stack(bank)
    np.save(os.path.join(args.out, "qpos_bank.npy"), bank)
    rec["qpos_bank"] = {"shape": list(bank.shape), "bytes_float32": int(bank.astype(np.float32).nbytes)}

    mem1 = gpu_mem_mb()
    rec["gpu_mem_one_context_MB"] = mem1 - mem0
    print(f"GPU mem after 1 env+context: {mem1:.0f} MB (delta {mem1-mem0:.0f} MB)")

    # can a plain mujoco.MjModel be rebuilt from the dumped xml (assets resolved)?
    try:
        m2 = mujoco.MjModel.from_xml_path(xml_path)
        rec["xml_reloads_standalone"] = True
        rec["xml_reload_matches_shape"] = bool(m2.nq == m.nq and m2.ngeom == m.ngeom)
    except Exception as e:
        rec["xml_reloads_standalone"] = False
        rec["xml_reload_error"] = str(e)[:400]
        print("standalone xml reload FAILED:", str(e)[:400])

    if rec.get("xml_reloads_standalone"):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        rec["throughput"] = {}
        for p in args.procs:
            ret = ctx.Manager().dict()
            procs = [ctx.Process(target=worker,
                                 args=(xml_path, args.res, args.frames, bank, ret, i))
                     for i in range(p)]
            t0 = time.perf_counter()
            for q in procs:
                q.start()
            for q in procs:
                q.join()
            wall = time.perf_counter() - t0
            tot = p * args.frames
            rec["throughput"][str(p)] = {
                "procs": p, "wall_s": wall, "aggregate_fps": tot / wall,
                "per_proc_fps": float(np.mean([args.frames / ret[i] for i in range(p) if i in ret])),
                "gpu_mem_MB_after": gpu_mem_mb(),
            }
            print(f"procs={p:2d}  aggregate {tot/wall:8.1f} fps   "
                  f"per-proc {rec['throughput'][str(p)]['per_proc_fps']:7.1f} fps")

    # storage comparison for one 500-step episode, 3 cameras @128
    T, C, R = 500, 3, 128
    rec["storage_per_episode"] = {
        "raw_images_MB": T * C * R * R * 3 / 1e6,
        "jpeg_q90_est_MB": T * C * (R * R * 3 / 12) / 1e6,
        "qpos_float32_MB": T * int(m.nq) * 4 / 1e6,
        "full_mjstate_float64_MB": T * (1 + int(m.nq) + int(m.nv)) * 8 / 1e6,
        "scene_xml_gzip_MB": len(__import__("gzip").compress(xml.encode())) / 1e6,
        "asset_library_GB": 24.07,
    }

    print("\n" + json.dumps(rec, indent=2))
    with open(os.path.join(args.out, "batch.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
