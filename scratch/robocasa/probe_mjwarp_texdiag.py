"""Is the MJWarp mismatch a shading difference, or is texture sampling specifically broken?

The best-calibrated MJWarp frame has the right geometry, camera and robot pose but renders white
marble as saturated speckled green -- a mean error of 77.8/255. Two very different stories fit that:
a raytracer-vs-rasterizer shading gap (expected, and a fair reason to say "different renderer"), or
a texture-sampling failure on RoboCasa's assets (a bug, and a fixable one).

They separate cleanly: turn textures off on *both* sides and compare again. If the untextured frames
agree well while the textured ones do not, the shading model is fine and texturing is the fault. If
both disagree equally, it is shading.
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out"
XML = os.path.join(OUT, "pinned_scene.xml")
RES = 128


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    import imageio.v2 as imageio

    mjm = mujoco.MjModel.from_xml_path(XML)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = np.load(os.path.join(OUT, "qpos_bank.npy"))[0]
    mujoco.mj_forward(mjm, mjd)
    cam = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_agentview_left")

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1

    def mj_shot(textured):
        saved = mjm.mat_texid.copy()
        if not textured:
            mjm.mat_texid[:] = -1
        r = mujoco.Renderer(mjm, height=RES, width=RES)
        r.update_scene(mjd, camera=cam, scene_option=opt)
        px = r.render().copy()
        r.close()
        mjm.mat_texid[:] = saved
        return px

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=1)

    def warp_shot(textured):
        rc = mjw.create_render_context(
            mjm, nworld=1, cam_res=(RES, RES), render_rgb=True, render_depth=False,
            render_seg=False, use_textures=textured, use_shadows=False,
            use_ambient_lighting=True, enable_specular=True,
            enabled_geom_groups=[1], background_color=(0.0, 0.0, 0.0, 1.0),
            cam_active=[i == cam for i in range(mjm.ncam)])
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        wp.synchronize()
        return rc.rgb_data.numpy()[:RES * RES].view(np.uint8).reshape(RES, RES, 4)[..., :3].copy()

    def sc(a, b):
        e = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
        return {"mean": float(e.mean()), "max": int(e.max()),
                "exact": float((e == 0).mean()), "within8": float((e <= 8).mean()),
                "within32": float((e <= 32).mean())}

    rec = {}
    mj_t, mj_f = mj_shot(True), mj_shot(False)
    w_t, w_f = warp_shot(True), warp_shot(False)
    rec["textured_mj_vs_warp"] = sc(mj_t, w_t)
    rec["untextured_mj_vs_warp"] = sc(mj_f, w_f)
    # how much does each renderer change when textures are removed?
    rec["mj_texture_effect"] = sc(mj_t, mj_f)
    rec["warp_texture_effect"] = sc(w_t, w_f)

    for k, v in rec.items():
        print(f"{k:28s} {json.dumps(v)}")

    sheet = np.concatenate([
        np.concatenate([mj_t, np.zeros((RES, 3, 3), np.uint8), w_t], axis=1),
        np.concatenate([mj_f, np.zeros((RES, 3, 3), np.uint8), w_f], axis=1),
    ], axis=0)
    imageio.imwrite(os.path.join(OUT, "texdiag_sheet.png"), sheet)
    print("\nsheet: rows = [textured, untextured], cols = [MuJoCo GL, MJWarp]")
    with open(os.path.join(OUT, "mjwarp_texdiag.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
