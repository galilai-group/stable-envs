"""Pixel agreement between MJWarp's batch raytracer and MuJoCo's own EGL renderer.

Throughput is easy to measure and easy to trust. Agreement is neither, and it is the
thing stable-envs actually depends on: a backend that renders a *different-looking*
scene at a million fps is useless, because the whole premise is that a rendered frame
is interchangeable with one MuJoCo would have produced.

Three traps this script is built around, each of which silently fabricates a result:

  1. Channel order. MJWarp packs `(a<<24)|(r<<16)|(g<<8)|b` into uint32, so a byte view
     of `rc.rgb_data` is B,G,R,A -- not R,G,B,A. Reading it as RGB leaves geometry,
     camera and pose pixel-aligned while every colour is wrong, which looks exactly
     like "the renderer ignores materials". Both interpretations are reported.
  2. Geom groups. `mujoco.Renderer` shows groups 0-2 by default; a model whose collision
     geoms live in group 0 will paint them over the visual mesh on one side only. Set
     explicitly on both sides here.
  3. Render flags. Shadows/reflection/skybox live on `MjvScene.flags` (mjtRndFlag), not
     on `MjvOption.flags` (mjtVisFlag). Setting them on the option object changes nothing
     and yields a suspiciously perfect zero difference between two configs.

Reported per config: fraction of exactly-equal pixels, mean abs error, PSNR, and the
same three under the swapped-channel reading.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")

import mujoco
import numpy as np

GROUPS = (0, 1, 2)


def mujoco_egl(mjm, mjd, res, cam, shadow, reflection, skybox, groups=GROUPS):
    mjm.vis.global_.offwidth = max(mjm.vis.global_.offwidth, res)
    mjm.vis.global_.offheight = max(mjm.vis.global_.offheight, res)
    opt = mujoco.MjvOption()
    for g in range(len(opt.geomgroup)):
        opt.geomgroup[g] = 1 if g in groups else 0
    r = mujoco.Renderer(mjm, height=res, width=res)
    r.update_scene(mjd, camera=cam, scene_option=opt)
    # mjtRndFlag lives on the SCENE, and must be set after update_scene
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(reflection)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = int(skybox)
    img = r.render().copy()
    r.close()
    return img


def mjwarp(xml, mjd, res, cam, groups=GROUPS, **kw):
    import warp as wp

    import mujoco_warp as mjw

    mjm = mujoco.MjModel.from_xml_path(xml)
    md = mujoco.MjData(mjm)
    md.qpos[:] = mjd.qpos
    mujoco.mj_forward(mjm, md)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, md, nworld=1)
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(res, res), render_rgb=True,
        cam_active=[i == cam for i in range(mjm.ncam)],
        enabled_geom_groups=list(groups), **kw)
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    return rc.rgb_data.numpy().view(np.uint8).reshape(res, res, 4)


def _fg_mask(ref):
    """Pixels that are not the single most common colour.

    A uniform sky or floor can be most of the frame, and agreeing on it is nearly free;
    quoting only whole-image agreement would flatter any backend on an empty scene.
    """
    flat = ref.reshape(-1, 3)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    bg = vals[counts.argmax()]
    return (np.abs(ref.astype(np.int32) - bg.astype(np.int32)).max(-1) > 4)


def agreement(ref, bgra):
    """`ref` is MuJoCo RGB; `bgra` is MJWarp's raw byte view."""
    fg = _fg_mask(ref)
    out = {"fg_frac": float(fg.mean())}
    for name, img in (("rgb_swapped", bgra[..., [2, 1, 0]]), ("raw_bytes_as_rgb", bgra[..., :3])):
        a, b = ref.astype(np.int32), img.astype(np.int32)
        d = np.abs(a - b).max(-1)
        mse = float(((a - b) ** 2).mean())
        rec = dict(exact_frac=float((d == 0).mean()), mae=float(np.abs(a - b).mean()),
                   psnr=float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse)),
                   within_2=float((d <= 2).mean()), within_8=float((d <= 8).mean()))
        if fg.any():
            rec["fg_exact_frac"] = float((d[fg] == 0).mean())
            rec["fg_mae"] = float(np.abs(a - b)[fg].mean())
            rec["fg_within_8"] = float((d[fg] <= 8).mean())
        out[name] = rec
    return out


def flatten_scene(xml, out_xml):
    """Remove texture *maps* while keeping every material and its colour.

    This is the ablation that separates the two failure modes. Materials stay, so the
    scene keeps its yellow ant and blue floor -- which also keeps the channel-order
    diagnostic meaningful, since a swap is invisible on achromatic pixels.
    """
    import shutil
    import xml.etree.ElementTree as ET

    t = ET.parse(xml)
    root = t.getroot()
    for asset in root.iter("asset"):
        for el in list(asset):
            if el.tag == "texture":
                asset.remove(el)
    for el in root.iter("material"):
        el.attrib.pop("texture", None)
        el.attrib.pop("texrepeat", None)
        el.attrib.pop("texuniform", None)
    os.makedirs(os.path.dirname(out_xml), exist_ok=True)
    t.write(out_xml)
    for el in t.iter():
        f = el.get("file")
        if f:
            s = os.path.join(os.path.dirname(xml), os.path.basename(f))
            if os.path.exists(s):
                shutil.copy(s, os.path.join(os.path.dirname(out_xml), os.path.basename(f)))
    return out_xml


def sweep(xml, res, cam, label, images=None):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    rng = np.random.default_rng(3)
    mjd.qpos[:2] += rng.uniform(-2, 2, size=2)
    mujoco.mj_forward(mjm, mjd)

    rows = []
    best = None
    for shadow, skybox, textures, ambient, specular in itertools.product(
            (True, False), (True, False), (True, False), (True, False), (True, False)):
        try:
            ref = mujoco_egl(mjm, mjd, res, cam, shadow=shadow, reflection=False, skybox=skybox)
            got = mjwarp(xml, mjd, res, cam, use_shadows=shadow, render_skybox=skybox,
                         use_textures=textures, use_ambient_lighting=ambient,
                         enable_specular=specular)
        except Exception as e:
            print(json.dumps({"scene": label, "error": f"{type(e).__name__}: {e}"}), flush=True)
            continue
        cfg = dict(shadow=shadow, skybox=skybox, mjwarp_textures=textures,
                   ambient=ambient, specular=specular)
        row = dict(scene=label, res=res, **cfg, **agreement(ref, got))
        rows.append(row)
        key = row["rgb_swapped"]["exact_frac"]
        if best is None or key > best[0]:
            best = (key, row, ref, got)
    if best is not None:
        _, row, ref, got = best
        print(json.dumps({"scene": label, "BEST": {k: row[k] for k in
                          ("shadow", "skybox", "mjwarp_textures", "ambient", "specular")},
                          "rgb_swapped": row["rgb_swapped"],
                          "raw_bytes_as_rgb": row["raw_bytes_as_rgb"]}), flush=True)
        if images:
            import imageio.v2 as imageio
            os.makedirs(images, exist_ok=True)
            strip = np.concatenate([ref, got[..., [2, 1, 0]], got[..., :3]], axis=1)
            imageio.imwrite(os.path.join(images, f"agree_{label}.png"), strip)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml,humanoidmaze_medium.xml,cube_single.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--images", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "images"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    allrows = []
    for scene in args.scenes.split(","):
        xml = os.path.join(args.dir, scene)
        if not os.path.exists(xml):
            continue
        mjm = mujoco.MjModel.from_xml_path(xml)
        cam = 0 if mjm.ncam > 0 else -1
        allrows += sweep(xml, args.res, cam, scene[:-4], args.images)
        flat = flatten_scene(xml, os.path.join(args.dir, "flat", scene))
        allrows += sweep(flat, args.res, cam, scene[:-4] + "_FLAT", args.images)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(allrows, f, indent=1)
