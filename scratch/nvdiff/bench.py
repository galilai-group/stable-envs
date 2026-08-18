"""Throughput and memory, as a (ratio, batch) pair with its crossover -- never a bare peak.

nvdiffrast's instanced mode wants the full ``(B, V, 4)`` clip-space vertex buffer, so device memory
grows with batch x scene vertex count where MJWarp shares one BVH across worlds. That is the property
that caps the usable batch, and the honest way to report it is the whole curve plus the batch at which
it OOMs, per scene and per resolution.

Geom frames are taken as given on device: producing them from a state vector is MJWarp's or MuJoCo's
job and costs 0.6 % of a frame (``docs/research/backend3d.md``), so including CPU kinematics here would
measure the wrong thing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from common import emit, save  # noqa: E402


def frames_for(m, scene, batch, seed=0):
    import mujoco
    import torch
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = np.asarray(d.qpos, np.float64).copy()
    xp = np.zeros((batch, len(scene.geoms), 3), np.float32)
    xm = np.zeros((batch, len(scene.geoms), 3, 3), np.float32)
    for b in range(min(batch, 32)):
        q = q0.copy()
        q[:min(3, m.nq)] += rng.uniform(-0.2, 0.2, min(3, m.nq))
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        xp[b] = np.asarray(d.geom_xpos, np.float64)[scene.geoms]
        xm[b] = np.asarray(d.geom_xmat, np.float64).reshape(-1, 3, 3)[scene.geoms]
    for b in range(32, batch):
        xp[b], xm[b] = xp[b % 32], xm[b % 32]
    return (torch.as_tensor(xp, device="cuda"), torch.as_tensor(xm, device="cuda"))


def run(xml, res, batches, label, cam_name=None, iters=10, warmup=3):
    import mujoco
    import torch

    from stable_envs.render import nvdiff as ND
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rows = []
    for b in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            br = ND.BatchRenderer(m, res=res, device="cuda")
            cam, _ = br.resolve_from_mujoco(m, d, camera=None)
            if cam_name:
                cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                cam = ND.Camera.from_model(m, d, cid)
            xp, xm = frames_for(m, br.scene, b)
            with torch.no_grad():
                for _ in range(warmup):
                    out = br.render(xp, xm, cam)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    out = br.render(xp, xm, cam)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            row = {"scene": label, "res": res, "batch": b, "verts": br.scene.V,
                   "tris": int(br.scene.face.shape[0]), "geoms": br.scene.G,
                   "ms_per_batch": 1e3 * dt / iters, "fps": b * iters / dt,
                   "peak_mb": torch.cuda.max_memory_allocated() / 1e6,
                   "out_shape": list(out.shape)}
            del br, xp, xm, out
        except Exception as e:                                        # noqa: BLE001
            row = {"scene": label, "res": res, "batch": b,
                   "error": f"{type(e).__name__}: {str(e)[:140]}"}
        emit("bench", row)
        rows.append(row)
        save(f"bench_partial_{label}_{res}.json", rows)
        torch.cuda.empty_cache()
        if "error" in row:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[224, 64])
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 8, 64, 256, 1024, 4096])
    ap.add_argument("--only", default="", help="run ONE scene, so a hard crash costs one row")
    args = ap.parse_args()
    import torch
    emit("device", {"name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "total_mb": torch.cuda.get_device_properties(0).total_memory / 1e6})
    S = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/backend3d/scenes"
    C = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/corpus"
    scenes = [(os.path.join(S, "antmaze_medium.xml"), "antmaze-medium", None),
              (os.path.join(S, "humanoidmaze_medium.xml"), "humanoidmaze-medium", None),
              (os.path.join(S, "cube_single.xml"), "cube-single", None),
              (os.path.join(C, "L52S03", "ep000.xml"), "robocasa-L52S03",
               "robot0_agentview_left")]
    rows = []
    for res in args.res:
        for xml, label, cam in scenes:
            if not os.path.exists(xml) or (args.only and args.only != label):
                continue
            rows += run(xml, res, args.batches, label, cam)
    tag = args.only or "all"
    save(f"bench_{tag}_{'_'.join(str(r) for r in args.res)}.json", rows)


if __name__ == "__main__":
    main()
