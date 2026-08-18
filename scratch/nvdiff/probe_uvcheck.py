"""Compare OUR texture coordinate to MuJoCo's, pixel by pixel, on the real scene.

Scoring a rendered frame answers "do the two pictures agree" and not "does the texgen agree", and
those come apart: a checker painted half a period out of phase is a texgen error that looks like a
filtering error. So this removes everything else -- the model's texture is repainted as a linear
ramp, its material set to pure emission and the lights zeroed, so MuJoCo's frame IS its texture
sample -- and the same modified model is handed to our renderer. Any disagreement left is texgen.

Restricted to the near band, because in the far field MuJoCo's mip average of a WRAPPED ramp is not
the ramp and the comparison would be measuring the filter instead.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from common import emit, save  # noqa: E402


def ramp(m, tex):
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[tex])
    jj, ii = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pat = np.zeros((h, w, nc), np.uint8)
    pat[..., 0] = np.round(ii * 255.0 / (w - 1))
    if nc > 1:
        pat[..., 1] = np.round(jj * 255.0 / (h - 1))
    if nc > 2:
        pat[..., 2] = 128
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    buf[a0:a0 + w * h * nc] = pat.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--kind", default="gym", choices=("gym", "ogbench"))
    ap.add_argument("--geom", default="floor")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--band", type=float, default=0.25, help="nearest fraction of the surface")
    args = ap.parse_args()

    import gymnasium
    import mujoco
    import torch

    from stable_envs.render import nvdiff as ND

    if args.kind == "gym":
        env = gymnasium.make(args.env, render_mode="rgb_array", width=args.size, height=args.size)
        env.reset(seed=0)
        u_ = env.unwrapped
        cam = mujoco.mj_name2id(u_.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        cam = cam if cam >= 0 else 0
        cam_arg = int(cam)
    else:
        import ogbench  # noqa: F401
        loco, maze = args.env.split(":")
        env = gymnasium.make(f"{loco}maze-{maze}-v0", render_mode="rgb_array", ob_type="pixels",
                             width=args.size, height=args.size)
        env.reset(options={"task_id": 1})
        u_ = env.unwrapped
        cam_arg = u_.custom_camera
    m, d = u_.model, u_.data
    m.vis.quality.offsamples = 0

    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, args.geom)
    mid = int(m.geom_matid[g])
    tex = ND._tex_of(m, mid)
    ramp(m, tex)
    m.mat_emission[mid] = 1.0
    m.mat_specular[mid] = 0.0
    m.mat_reflectance[mid] = 0.0
    for li in range(m.nlight):
        m.light_diffuse[li] = 0
        m.light_specular[li] = 0
        m.light_ambient[li] = 0
    m.vis.headlight.diffuse[:] = 0
    m.vis.headlight.specular[:] = 0
    m.vis.headlight.ambient[:] = 0
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    Am, bm = ND.texgen(m, g, drawn_plane=20.0 * float(m.stat.extent))
    emit("setup", {"env": args.env, "geom": g, "type": int(m.geom_type[g]),
                   "size": np.asarray(m.geom_size[g]).tolist(), "tex": [w, h],
                   "texrepeat": np.asarray(m.mat_texrepeat[mid]).tolist(),
                   "texuniform": int(m.mat_texuniform[mid]),
                   "ours_A": np.diag(Am).tolist(), "ours_b": bm.tolist()})

    mujoco.mj_forward(m, d)
    ren = mujoco.Renderer(m, height=args.size, width=args.size)
    ren.update_scene(d, camera=cam_arg)
    for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
        ren.scene.flags[f] = 0
    ref = ren.render().copy()
    ren.enable_segmentation_rendering()
    ren.update_scene(d, camera=cam_arg)
    seg = ren.render()[..., 0].copy()
    ren.disable_segmentation_rendering()
    ren.enable_depth_rendering()
    ren.update_scene(d, camera=cam_arg)
    dep = ren.render().copy()
    ren.disable_depth_rendering()

    br = ND.BatchRenderer(m, res=args.size, device="cuda", tex_policy="exact", skybox=False)
    ndcam, _ = br.resolve_from_mujoco(m, d, camera=cam_arg if args.kind == "ogbench" else None)
    br.mipmap = False
    q = torch.as_tensor(np.asarray(d.qpos, np.float64)[None], dtype=torch.float32)
    ours = (br.render_qpos(q, camera_id=cam_arg if args.kind == "gym" else -1,
                           cam=None if args.kind == "gym" else ndcam)[0].cpu().numpy())

    sel = seg == g
    if sel.sum():
        thr = np.quantile(dep[sel], args.band)
        sel = sel & (dep <= thr)
    ur = ref[..., 0].astype(np.float64) * (w - 1) / 255.0
    uo = ours[..., 0].astype(np.float64) * (w - 1) / 255.0
    vr = ref[..., 1].astype(np.float64) * (h - 1) / 255.0
    vo = ours[..., 1].astype(np.float64) * (h - 1) / 255.0
    du = (uo - ur + w / 2) % w - w / 2                      # texel offset, shortest way round
    dv = (vo - vr + h / 2) % h - h / 2
    emit("uv", {"npx": int(sel.sum()),
                "du_texels_median": float(np.median(du[sel])),
                "du_texels_p10": float(np.quantile(du[sel], 0.1)),
                "du_texels_p90": float(np.quantile(du[sel], 0.9)),
                "dv_texels_median": float(np.median(dv[sel])),
                "dv_texels_p10": float(np.quantile(dv[sel], 0.1)),
                "dv_texels_p90": float(np.quantile(dv[sel], 0.9)),
                "frac_within_1_texel": float((np.abs(du[sel]) <= 1).mean()),
                "tex_w": w, "tex_h": h})
    np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/uv_{args.env}.npy",
            np.stack([ur, uo, vr, vo, sel.astype(float), dep], -1))
    save(f"uv_{args.env.replace(':', '_')}.json", {"ok": True})


if __name__ == "__main__":
    main()
