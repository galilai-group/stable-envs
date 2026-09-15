"""Step 7 / W5 phase 0 -- price the one open fidelity risk: the floor texture's minification.

`stable_envs/render/nvdiff.py` records that `mujoco_warp` builds every texture at
`num_mip_levels=1` and samples with no level of detail, and that this single fact carries
99.45-100.00 % of the interior error on the Gymnasium locomotion floors. These two scenes have
exactly one 2-D texture -- a 512x512 `grid` on the floor PLANE -- and the floor is 72-85 % of the
frame, so the same failure mode is the largest unmeasured risk in the GPU verdict.

This measures the TEXEL FOOTPRINT PER PIXEL analytically: cast the registered camera's rays at the
floor plane, take the world-space hit points, convert to texel coordinates under MuJoCo's PLANE
texgen law (`texuniform=1`, so the texture repeats every `texrepeat` WORLD units), and take the
finite difference between neighbouring pixels. A footprint of 1 texel/pixel is magnification --
point sampling and mipmapping agree. A footprint of N >> 1 is minification, where MuJoCo averages
N x N texels and a point sampler picks one of them.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for eid in a.envs:
        env = gym.make(eid); u = env.unwrapped
        env.reset(seed=0); m, d = u.model, u.data
        h, w = u._render_height, u._render_width
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
        fovy = float(m.cam_fovy[cid])
        f = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
        cpos = np.array(d.cam_xpos[cid]); R = np.array(d.cam_xmat[cid]).reshape(3, 3)

        fl = [i for i in range(m.ngeom) if int(m.geom_type[i]) == 0]
        rec = {"env_id": eid, "h": h, "w": w, "fovy": fovy, "n_plane_geoms": len(fl)}
        gi = fl[0]
        mat = int(m.geom_matid[gi])
        texid = [int(x) for x in np.atleast_1d(m.mat_texid[mat]) if int(x) >= 0]
        tw = int(m.tex_width[texid[0]]); th = int(m.tex_height[texid[0]])
        rep = np.array(m.mat_texrepeat[mat], float)
        uniform = int(m.mat_texuniform[mat])
        z0 = float(d.geom_xpos[gi][2])
        rec.update({"floor_geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi),
                    "texture": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, texid[0]),
                    "tex_w": tw, "tex_h": th, "texrepeat": rep.tolist(),
                    "texuniform": uniform, "floor_z": z0,
                    "geom_size": m.geom_size[gi].tolist()})

        # camera rays, in the same parameterisation probe_determinism_bars uses
        ys, xs = np.mgrid[0:h, 0:w]
        cx = (xs + 0.5 - w / 2.0) / f
        cy = -(ys + 0.5 - h / 2.0) / f
        dirs_cam = np.stack([cx, cy, -np.ones_like(cx)], -1)
        dirs = dirs_cam @ R.T
        denom = dirs[..., 2]
        t = np.where(np.abs(denom) > 1e-9, (z0 - cpos[2]) / denom, np.nan)
        t = np.where(t > 0, t, np.nan)
        P = cpos[None, None, :] + t[..., None] * dirs          # world hit points
        # A MuJoCo PLANE with nonzero size[0:2] is BOUNDED: half-extents in the plane's own frame.
        # Ignoring that puts the whole horizon in the footprint statistic and turns a bounded
        # 2 m x 2 m tile into a locomotion floor, which is the opposite of the truth here.
        gR = np.array(d.geom_xmat[gi]).reshape(3, 3)
        gp = np.array(d.geom_xpos[gi])
        local = (P - gp[None, None, :]) @ gR
        half = np.array(m.geom_size[gi][:2], float)
        inside = np.ones(t.shape, bool)
        for k in (0, 1):
            if half[k] > 0:
                inside &= np.abs(local[..., k]) <= half[k]
        hit = np.isfinite(t) & inside
        rec["plane_bounded"] = [float(x) for x in half]
        rec["floor_ray_hit_share"] = float(hit.mean())

        # PLANE texgen with texuniform=1: texture coordinate is world position / texrepeat
        uv = P[..., :2] / rep[None, None, :]
        texel = uv * np.array([tw, th], float)
        du = np.abs(np.diff(texel[:, :, 0], axis=1))
        dv = np.abs(np.diff(texel[:, :, 1], axis=0))
        fp = np.fmax(np.nan_to_num(du[:-1, :], nan=0.0), np.nan_to_num(dv[:, :-1], nan=0.0))
        msk = hit[:-1, :-1] & np.isfinite(du[:-1, :]) & np.isfinite(dv[:, :-1])
        v = fp[msk]
        if v.size:
            rec["texel_per_pixel"] = {
                "median": float(np.median(v)), "mean": float(v.mean()),
                "p90": float(np.percentile(v, 90)), "p99": float(np.percentile(v, 99)),
                "max": float(v.max()),
                "frac_minified_gt1": float((v > 1).mean()),
                "frac_minified_gt2": float((v > 2).mean()),
                "frac_minified_gt4": float((v > 4).mean()),
                "n_floor_pixels": int(v.size),
                "floor_share_of_frame": float(msk.mean()),
            }
        (out / f"floor_footprint_{eid.split('/')[-1].replace('-v0','')}.json").write_text(
            json.dumps(rec, indent=1, sort_keys=True))
        tp = rec.get("texel_per_pixel", {})
        print(f"{eid}: floor {tp.get('floor_share_of_frame', 0)*100:.1f}% of frame; "
              f"texel/pixel median {tp.get('median', 0):.2f} p90 {tp.get('p90', 0):.2f} "
              f"p99 {tp.get('p99', 0):.2f} max {tp.get('max', 0):.2f}; "
              f"minified >1 {tp.get('frac_minified_gt1', 0)*100:.1f}% "
              f">2 {tp.get('frac_minified_gt2', 0)*100:.1f}% >4 {tp.get('frac_minified_gt4', 0)*100:.1f}%",
              flush=True)
        env.close()


if __name__ == "__main__":
    main()
