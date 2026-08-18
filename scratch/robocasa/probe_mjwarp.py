"""The decisive experiment: can MJWarp batch-render a *real, unmodified, textured* RoboCasa kitchen?

MJWarp ships a first-party GPU batch renderer with UV-mapped mesh textures, and mesh BVHs plus
texture memory are shared across worlds -- which is exactly the architecture the stable-envs thesis
wants. But the closest published artifact is DeepMind's own ``benchmarks/kitchen``, whose README
says textures were replaced by uniform colors and mesh UVs were stripped. So the interesting scene
has never been benchmarked, and the question has to be answered by running it.

Uses the exact kitchen this repo's other probes measured (``pinned_scene.xml``, dumped from
robocasa 1.0.1: 1251 geoms, 213 meshes, 553,830 faces, 388.7 MB of texture data), unmodified.

Reports three things, in this order of importance:
  1. does it load and render at all;
  2. throughput vs. world count, against the 1,467 fps that MuJoCo's EGL rasterizer already
     delivers on this GPU;
  3. whether the picture matches the official renderer -- MJWarp is a raytracer with lambertian +
     specular shading, MuJoCo is an OpenGL Phong rasterizer, so "batched" and "matches the official
     renderer" may simply not be the same target.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def gpu_mem_mb():
    try:
        import subprocess
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().strip().split("\n")[0])
    except Exception:
        return -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/pinned_scene.xml")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--nworld", type=int, nargs="+", default=[1, 16, 64, 256, 1024])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--groups", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    import warp as wp
    import mujoco_warp as mjw

    rec = {"res": args.res, "mujoco": mujoco.__version__,
           "mujoco_warp": getattr(mjw, "__version__", "?"), "warp": wp.__version__,
           "enabled_geom_groups": args.groups}
    try:
        import subprocess
        rec["gpu_name"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip().split("\n")[0]
    except Exception:
        rec["gpu_name"] = "?"

    t0 = time.time()
    mjm = mujoco.MjModel.from_xml_path(args.xml)
    rec["t_compile_s"] = time.time() - t0
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    rec["scene"] = {"ngeom": int(mjm.ngeom), "nmesh": int(mjm.nmesh),
                    "nmeshface": int(mjm.nmeshface), "ntex": int(mjm.ntex),
                    "texdata_MB": float(mjm.tex_data.size / 1e6), "nq": int(mjm.nq)}
    print("loaded:", json.dumps(rec["scene"]))

    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    rec["cam_id"] = int(cam_id)

    # ---- reference frame from the OFFICIAL OpenGL rasterizer ---------------------------------
    ref = None
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        r = mujoco.Renderer(mjm, height=args.res, width=args.res)
        r.update_scene(mjd, camera=cam_id)
        ref = r.render().copy()
        r.close()
        import imageio.v2 as imageio
        imageio.imwrite(os.path.join(args.out, "mjw_ref_opengl.png"), ref)
    except Exception as e:
        rec["opengl_ref_error"] = str(e)[:300]
        print("OpenGL reference failed:", str(e)[:300])

    # ---- MJWarp batch renderer ----------------------------------------------------------------
    mem0 = gpu_mem_mb()
    rec["throughput"] = {}
    for nw in args.nworld:
        try:
            t0 = time.time()
            m = mjw.put_model(mjm)
            # rendering never steps physics, but put_data still sizes the contact/constraint
            # arenas; this kitchen needs >= 595 contacts per world, so let MJWarp size them.
            d = mjw.put_data(mjm, mjd, nworld=nw)
            t_put = time.time() - t0

            t0 = time.time()
            rc = mjw.create_render_context(
                mjm, nworld=nw, cam_res=(args.res, args.res),
                render_rgb=True, render_depth=False, render_seg=False,
                use_textures=True, enabled_geom_groups=list(args.groups),
                cam_active=[i == cam_id for i in range(mjm.ncam)],
            )
            t_ctx = time.time() - t0

            mjw.refit_bvh(m, d, rc)
            mjw.render(m, d, rc)
            wp.synchronize()

            t0 = time.perf_counter()
            for _ in range(args.iters):
                mjw.refit_bvh(m, d, rc)
                mjw.render(m, d, rc)
            wp.synchronize()
            dt = (time.perf_counter() - t0) / args.iters

            entry = {"nworld": nw, "t_put_s": t_put, "t_context_s": t_ctx,
                     "s_per_batch": dt, "fps": nw / dt,
                     "gpu_mem_MB": gpu_mem_mb() - mem0}
            rec["throughput"][str(nw)] = entry
            print(f"nworld={nw:5d}  {nw/dt:10.0f} fps  batch {dt*1e3:7.2f} ms  "
                  f"ctx {t_ctx:6.1f}s  mem {entry['gpu_mem_MB']:.0f} MB")

            # pixel comparison against the OpenGL reference, once
            if nw == args.nworld[0] and ref is not None:
                rgb = rc.rgb_data.numpy()
                px = rgb[:args.res * args.res].view(np.uint8).reshape(args.res, args.res, 4)[..., :3]
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(args.out, "mjw_warp.png"), px)
                e = np.abs(px.astype(np.int16) - ref.astype(np.int16)).max(-1)
                rec["vs_official_renderer"] = {
                    "mean_absdiff": float(e.mean()), "max_absdiff": int(e.max()),
                    "frac_pixels_within_8": float((e <= 8).mean()),
                    "frac_pixels_exact": float((e == 0).mean()),
                }
                print("vs official OpenGL:", json.dumps(rec["vs_official_renderer"]))
                imageio.imwrite(os.path.join(args.out, "mjw_diff.png"),
                                np.clip(np.abs(px.astype(np.int16) - ref.astype(np.int16)), 0, 255).astype(np.uint8))
            del rc, m, d
        except Exception as ex:
            rec["throughput"][str(nw)] = {"error": f"{type(ex).__name__}: {str(ex)[:400]}"}
            print(f"nworld={nw}: FAILED {type(ex).__name__}: {str(ex)[:400]}")
            break

    print("\n" + json.dumps(rec, indent=2))
    with open(os.path.join(args.out, "mjwarp.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
