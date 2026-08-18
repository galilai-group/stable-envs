"""Fidelity: how close is each GPU backend to MuJoCo's own EGL renderer?

Same model, same camera, same qpos -> one image per backend, plus PSNR / mean abs
error against the EGL reference and a side-by-side strip. The point is not that a
backend matches bit for bit (none will), but whether the difference is a *look*
difference (shading, ambient) or a *content* difference (missing textures, wrong
geometry), because only the second one changes what a policy can see.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def egl_image(mjm, mjd, res, cam):
    mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, res)
    mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, res)
    r = mujoco.Renderer(mjm, height=res, width=res)
    r.update_scene(mjd, camera=cam)
    img = r.render().copy()
    r.close()
    return img


def mjwarp_image(xml, mjd, res, cam, **kw):
    import warp as wp

    import mujoco_warp as mjw

    mjm = mujoco.MjModel.from_xml_path(xml)
    md = mujoco.MjData(mjm)
    md.qpos[:] = mjd.qpos
    mujoco.mj_forward(mjm, md)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, md, nworld=1)
    rc = mjw.create_render_context(mjm, nworld=1, cam_res=(res, res), render_rgb=True,
                                   cam_active=[i == cam for i in range(mjm.ncam)], **kw)
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    # pack_rgba_to_uint32 is (a<<24)|(r<<16)|(g<<8)|b, so the byte view is BGRA
    return rc.rgb_data.numpy().view(np.uint8).reshape(res, res, 4)[..., [2, 1, 0]].copy()


def nvdiffrast_image(xml, mjd, res, cam):
    import torch

    from bench_nvdiffrast import NvdiffrastMJ

    mjm = mujoco.MjModel.from_xml_path(xml)
    md = mujoco.MjData(mjm)
    md.qpos[:] = mjd.qpos
    mujoco.mj_forward(mjm, md)
    r = NvdiffrastMJ(mjm, md, res, cam)
    xpos = torch.as_tensor(np.asarray(md.geom_xpos, np.float32)[None], device="cuda")
    xmat = torch.as_tensor(np.asarray(md.geom_xmat, np.float32).reshape(1, -1, 3, 3), device="cuda")
    with torch.no_grad():
        return r(xpos, xmat)[0].cpu().numpy()


def metrics(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mae = float(np.abs(a - b).mean())
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    # structure-only: compare per-pixel luminance after removing global mean/scale,
    # which separates "different shading" from "different geometry"
    la, lb = a.mean(-1), b.mean(-1)
    za = (la - la.mean()) / (la.std() + 1e-6)
    zb = (lb - lb.mean()) / (lb.std() + 1e-6)
    corr = float((za * zb).mean())
    return dict(mae=mae, psnr=psnr, lum_corr=corr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml,cube_single.xml,humanoidmaze_medium.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.makedirs(args.images, exist_ok=True)
    report = []
    for scene in args.scenes.split(","):
        xml = os.path.join(args.dir, scene)
        if not os.path.exists(xml):
            continue
        mjm = mujoco.MjModel.from_xml_path(xml)
        mjd = mujoco.MjData(mjm)
        rng = np.random.default_rng(3)
        mjd.qpos[:2] += rng.uniform(-2, 2, size=2)
        mujoco.mj_forward(mjm, mjd)
        cam = 0 if mjm.ncam > 0 else -1

        imgs = {}
        imgs["mujoco_egl"] = egl_image(mjm, mjd, args.res, cam)
        for name, kw in [("mjwarp", dict(use_textures=True, use_shadows=False)),
                         ("mjwarp_shadow", dict(use_textures=True, use_shadows=True)),
                         ("mjwarp_notex", dict(use_textures=False, use_shadows=False))]:
            try:
                imgs[name] = mjwarp_image(xml, mjd, args.res, max(cam, 0), **kw)
            except Exception as e:
                print(f"{scene} {name} failed: {type(e).__name__}: {e}")
        try:
            imgs["nvdiffrast"] = nvdiffrast_image(xml, mjd, args.res, cam)
        except Exception as e:
            print(f"{scene} nvdiffrast failed: {type(e).__name__}: {e}")

        ref = imgs["mujoco_egl"]
        for k, v in imgs.items():
            imageio.imwrite(os.path.join(args.images, f"{scene[:-4]}_{k}.png"), v)
            if k != "mujoco_egl":
                row = dict(scene=scene, backend=k, **metrics(ref, v))
                print(json.dumps(row), flush=True)
                report.append(row)
        strip = np.concatenate([imgs[k] for k in imgs], axis=1)
        imageio.imwrite(os.path.join(args.images, f"{scene[:-4]}_STRIP.png"), strip)
        print(f"strip order: {list(imgs)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
