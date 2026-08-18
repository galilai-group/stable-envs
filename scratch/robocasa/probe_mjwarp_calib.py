"""How close can MJWarp's batch raytracer be *made* to MuJoCo's OpenGL rasterizer?

The first pass scored 6.3% pixel-exact, but scored it at MJWarp's defaults -- skybox off, an
arbitrary dark-maroon background, ambient/specular/emission all on -- against a MuJoCo frame that
draws a skybox and uses a headlight. Reporting that number as "the renderers disagree" would be the
mistake ``calibrate.py`` exists to prevent: it measures the defaults, not the renderer.

So the settings are swept rather than assumed, on a realistic qpos from a rollout (not the default
pose, which puts the camera inside a bottle), and the best achievable agreement is reported along
with which flags got it there. If even the best configuration is far off, that is a real finding
about raytracer-vs-rasterizer; if it lands near-exact, the batch renderer is a drop-in.

Also extends the throughput curve to larger world counts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/pinned_scene.xml")
    ap.add_argument("--qpos", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out/qpos_bank.npy")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--nworld-sweep", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    import imageio.v2 as imageio

    mjm = mujoco.MjModel.from_xml_path(args.xml)
    mjd = mujoco.MjData(mjm)
    qpos = np.load(args.qpos)[0]
    mjd.qpos[:] = qpos
    mujoco.mj_forward(mjm, mjd)
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)

    # ---- official reference -------------------------------------------------------------------
    # robosuite renders group 1 only (base.py sets geomgroup[0]=render_collision_mesh=False).
    # A bare Renderer would also paint the collision primitives and the comparison would be
    # against a frame RoboCasa never produces.
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    r = mujoco.Renderer(mjm, height=args.res, width=args.res)
    r.update_scene(mjd, camera=cam_id, scene_option=opt)
    ref = r.render().copy()
    r.close()
    imageio.imwrite(os.path.join(args.out, "calib_ref_opengl.png"), ref)

    rec = {"res": args.res, "cam": args.cam,
           "mujoco": mujoco.__version__, "mujoco_warp": getattr(mjw, "__version__", "?")}

    def score(px):
        e = np.abs(px.astype(np.int16) - ref.astype(np.int16)).max(-1)
        return {"mean": float(e.mean()), "max": int(e.max()),
                "exact": float((e == 0).mean()), "within8": float((e <= 8).mean()),
                "within32": float((e <= 32).mean())}

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=1)
    # push the same qpos into the warp Data
    d.qpos.assign(np.tile(qpos.astype(np.float32), (1, 1)))

    # ---- sweep the settings that could plausibly explain the gap ------------------------------
    results = []
    grid = list(itertools.product(
        [True, False],                     # render_skybox
        [True, False],                     # use_ambient_lighting
        [True, False],                     # enable_specular
        [(0.0, 0.0, 0.0, 1.0), (0.1, 0.1, 0.2, 1.0)],   # background
        [[1], [0, 1, 2]],                  # robosuite renders group 1 only (collision geoms are group 0)
    ))
    best = None
    for sky, amb, spec, bg, groups in grid:
        try:
            rc = mjw.create_render_context(
                mjm, nworld=1, cam_res=(args.res, args.res), render_rgb=True,
                render_depth=False, render_seg=False, use_textures=True,
                use_shadows=False, use_ambient_lighting=amb, enable_specular=spec,
                enable_emission=True, enable_per_light_ambient=True,
                render_skybox=sky, background_color=bg,
                enabled_geom_groups=groups,
                cam_active=[i == cam_id for i in range(mjm.ncam)],
            )
            mjw.refit_bvh(m, d, rc)
            mjw.render(m, d, rc)
            wp.synchronize()
            px = rc.rgb_data.numpy()[:args.res * args.res].view(np.uint8)\
                   .reshape(args.res, args.res, 4)[..., :3]
            s = score(px)
            s.update({"skybox": sky, "ambient": amb, "specular": spec,
                      "bg": bg, "groups": groups})
            results.append(s)
            print(f"sky={int(sky)} amb={int(amb)} spec={int(spec)} bg={bg[0]:.1f} "
                  f"grp={groups} -> mean {s['mean']:6.2f} exact {s['exact']*100:5.2f}% "
                  f"w8 {s['within8']*100:5.2f}%")
            if best is None or s["mean"] < best[0]["mean"]:
                best = (s, px.copy())
            del rc
        except Exception as ex:
            print("cfg failed:", type(ex).__name__, str(ex)[:200])

    rec["sweep"] = results
    rec["best"] = best[0] if best else None
    if best:
        imageio.imwrite(os.path.join(args.out, "calib_best_mjwarp.png"), best[1])
        imageio.imwrite(os.path.join(args.out, "calib_sheet.png"),
                        np.concatenate([ref, np.zeros((args.res, 3, 3), np.uint8), best[1],
                                        np.zeros((args.res, 3, 3), np.uint8),
                                        np.abs(best[1].astype(np.int16) - ref.astype(np.int16))
                                        .clip(0, 255).astype(np.uint8)], axis=1))
        print("\nBEST:", json.dumps(best[0], default=str))

    # ---- throughput at larger world counts ----------------------------------------------------
    del d
    rec["throughput"] = {}
    for nw in args.nworld_sweep:
        try:
            dd = mjw.put_data(mjm, mjd, nworld=nw)
            rc = mjw.create_render_context(
                mjm, nworld=nw, cam_res=(args.res, args.res), render_rgb=True,
                render_depth=False, render_seg=False, use_textures=True,
                use_shadows=False, render_skybox=True,
                enabled_geom_groups=[1],
                cam_active=[i == cam_id for i in range(mjm.ncam)])
            mjw.refit_bvh(m, dd, rc)
            mjw.render(m, dd, rc)
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                mjw.refit_bvh(m, dd, rc)
                mjw.render(m, dd, rc)
            wp.synchronize()
            dt = (time.perf_counter() - t0) / args.iters
            rec["throughput"][str(nw)] = {"fps": nw / dt, "batch_ms": dt * 1e3}
            print(f"nworld={nw:5d}  {nw/dt:10.0f} fps  ({dt*1e3:.1f} ms/batch)")
            del rc, dd
        except Exception as ex:
            rec["throughput"][str(nw)] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
            print(f"nworld={nw}: FAILED {type(ex).__name__}: {str(ex)[:300]}")
            break

    with open(os.path.join(args.out, "mjwarp_calib.json"), "w") as f:
        json.dump(rec, f, indent=2, default=str)


if __name__ == "__main__":
    main()
