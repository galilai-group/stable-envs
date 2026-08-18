"""G4 on the GPU: what does regenerating a RoboCasa episode cost with MJWarp's batch raytracer?

The original G4 verdict (34.7x slower than JPEG decode) was measured against MuJoCo's single-world
EGL rasterizer with a per-episode MJCF compile. Two things could flip it: the render loop moving to
a batched GPU raytracer, and the per-episode setup amortizing over a batch that shares a kitchen.

So the setup is priced separately from the loop, at every batch size, and the episode cost is
reported as a function of how many episodes share one compiled scene:

    episode_seconds(K) = (compile + put_model + put_data + render_context + capture) / K  +  T / fps(B)

K = 1 is what RoboCasa actually gives you (a fresh kitchen per reset, and its own replay code stores
a fresh MJCF per demo). K -> inf is the pinned-kitchen regime. The crossover K against JPEG decode
is the number this whole question reduces to.

Run in the se3d venv (mujoco 3.11, mujoco-warp 3.11, warp 1.16, torch cu130).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(HERE, "out", "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "out", "qpos_bank.npy"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256, 512, 1024, 4096])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--tag", default="mjwarp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw

    def gpu_mb():
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e6

    rec = {"res": args.res, "T": args.T, "cam": args.cam,
           "mujoco": mujoco.__version__,
           "mujoco_warp": getattr(mjw, "__version__", "?"),
           "warp": wp.config.version,
           "gpu": torch.cuda.get_device_name(0)}

    t0 = time.perf_counter()
    mjm = mujoco.MjModel.from_xml_path(args.xml)
    rec["compile_xml_s"] = time.perf_counter() - t0
    mjd = mujoco.MjData(mjm)
    bank = np.load(args.qpos).astype(np.float32)
    mjd.qpos[:] = bank[0]
    mujoco.mj_forward(mjm, mjd)
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if cam_id < 0:
        cam_id = 0
    rec["nq"], rec["ngeom"], rec["nmesh"] = int(mjm.nq), int(mjm.ngeom), int(mjm.nmesh)

    base_mb = gpu_mb()
    t0 = time.perf_counter()
    m = mjw.put_model(mjm)
    wp.synchronize()
    rec["put_model_s"] = time.perf_counter() - t0
    rec["put_model_gpu_MB"] = gpu_mb() - base_mb

    rows = []
    for B in args.batches:
        row = {"batch": B}
        try:
            before = gpu_mb()
            t0 = time.perf_counter()
            d = mjw.put_data(mjm, mjd, nworld=B)
            wp.synchronize()
            row["put_data_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            rc = mjw.create_render_context(
                mjm, nworld=B, cam_res=(args.res, args.res), render_rgb=True,
                render_depth=False, render_seg=False, use_textures=True,
                use_shadows=False, render_skybox=True, enabled_geom_groups=[1],
                cam_active=[i == cam_id for i in range(mjm.ncam)],
            )
            wp.synchronize()
            row["render_ctx_s"] = time.perf_counter() - t0

            bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(B, args.res, args.res, 4)
            qdev = torch.as_tensor(bank[np.arange(B) % len(bank)], device="cuda")

            def frame():
                mjw.kinematics(m, d)
                mjw.camlight(m, d)
                mjw.refit_bvh(m, d, rc)
                mjw.render(m, d, rc)

            frame()
            wp.synchronize()
            t0 = time.perf_counter()
            with wp.ScopedCapture() as cap:
                frame()
            graph = cap.graph
            wp.synchronize()
            row["graph_capture_s"] = time.perf_counter() - t0
            row["gpu_MB_total"] = gpu_mb()
            row["gpu_MB_batch"] = gpu_mb() - before

            qwarp = wp.to_torch(d.qpos)

            def timeit(fn, iters=args.iters, warmup=3):
                for _ in range(warmup):
                    fn()
                wp.synchronize(); torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(iters):
                    fn()
                wp.synchronize(); torch.cuda.synchronize()
                return (time.perf_counter() - t) / iters

            def eager():
                qwarp.copy_(qdev)
                frame()

            def graphed():
                qwarp.copy_(qdev)
                wp.capture_launch(graph)

            def graphed_rgb():
                qwarp.copy_(qdev)
                wp.capture_launch(graph)
                return bgra[..., [2, 1, 0]]

            host = torch.empty((B, args.res, args.res, 3), dtype=torch.uint8, device="cpu",
                               pin_memory=True)

            def graphed_rgb_host():
                qwarp.copy_(qdev)
                wp.capture_launch(graph)
                host.copy_(bgra[..., [2, 1, 0]], non_blocking=True)

            row["eager_ms"] = 1e3 * timeit(eager)
            g = timeit(graphed)
            row["graph_ms"] = 1e3 * g
            row["graph_fps"] = B / g
            gr = timeit(graphed_rgb)
            row["graph_rgb_ms"] = 1e3 * gr
            row["graph_rgb_fps"] = B / gr
            gh = timeit(graphed_rgb_host, iters=max(5, args.iters // 4))
            row["graph_rgb_host_ms"] = 1e3 * gh
            row["graph_rgb_host_fps"] = B / gh

            setup = (rec["compile_xml_s"] + rec["put_model_s"] + row["put_data_s"]
                     + row["render_ctx_s"] + row["graph_capture_s"])
            row["setup_s"] = setup
            row["loop_s_per_episode"] = args.T / row["graph_rgb_fps"]
            row["episode_s_K1"] = setup + row["loop_s_per_episode"]
            for K in (1, 10, 100, 1000):
                row[f"episode_s_K{K}"] = setup / K + row["loop_s_per_episode"]

            del rc, d, graph, bgra, qdev, qwarp, host
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
        except Exception as ex:
            row["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
            print(json.dumps(row), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)

    rec["batches"] = rows
    with open(os.path.join(args.out, f"g4_{args.tag}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: v for k, v in rec.items() if k != "batches"}, indent=2))


if __name__ == "__main__":
    main()
