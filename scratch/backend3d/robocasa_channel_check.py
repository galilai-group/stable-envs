"""Re-measure the RoboCasa MJWarp agreement with the channel order corrected.

`scratch/robocasa/probe_mjwarp_calib.py` reads MJWarp's packed buffer as

    rc.rgb_data.numpy().view(np.uint8).reshape(res, res, 4)[..., :3]

but the pack is `(a<<24)|(r<<16)|(g<<8)|b`, so those three bytes are B, G, R. That leaves
geometry, camera, pose and shadows perfectly aligned while inverting red and blue -- which
is exactly the reported symptom ("colour does not line up", mean error 77.8/255).

This script reuses that probe's own pinned scene, qpos and geom-group convention and
reports both readings side by side, so the question is settled by measurement rather
than by inspecting the source.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

ROBOCASA = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(ROBOCASA, "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(ROBOCASA, "qpos_bank.npy"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--out", default="")
    ap.add_argument("--image", default="")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    mjm = mujoco.MjModel.from_xml_path(args.xml)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = np.load(args.qpos)[0]
    mujoco.mj_forward(mjm, mjd)
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)

    # robosuite renders group 1 only; matching that on both sides is the probe's own convention
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    r = mujoco.Renderer(mjm, height=args.res, width=args.res)
    r.update_scene(mjd, camera=cam_id, scene_option=opt)
    ref = r.render().copy()
    r.close()

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=1)
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(args.res, args.res), render_rgb=True,
        cam_active=[i == cam_id for i in range(mjm.ncam)],
        enabled_geom_groups=[1], use_textures=True, use_shadows=True,
        render_skybox=True, use_ambient_lighting=True, enable_specular=True)
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    bgra = rc.rgb_data.numpy()[: args.res * args.res].view(np.uint8).reshape(args.res, args.res, 4)

    rows = {}
    for name, img in (("as_probe_reads_it_BGR", bgra[..., :3]),
                      ("channel_corrected_RGB", bgra[..., [2, 1, 0]])):
        a, b = ref.astype(np.int32), img.astype(np.int32)
        dd = np.abs(a - b)
        rows[name] = dict(mean=float(dd.mean()), max=int(dd.max()),
                          exact=float((dd.max(-1) == 0).mean()),
                          within8=float((dd.max(-1) <= 8).mean()),
                          within32=float((dd.max(-1) <= 32).mean()))
    print(json.dumps({"scene": os.path.basename(args.xml), "res": args.res,
                      "ngeom": int(mjm.ngeom), "ntex": int(mjm.ntex), **rows}, indent=1), flush=True)

    if args.image:
        import imageio.v2 as imageio
        sep = np.zeros((args.res, 4, 3), np.uint8)
        imageio.imwrite(args.image, np.concatenate(
            [ref, sep, bgra[..., :3], sep, bgra[..., [2, 1, 0]]], axis=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
