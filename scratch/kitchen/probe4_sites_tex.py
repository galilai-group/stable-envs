"""Probe 4: name the visible site(s), and measure how flat the three cube textures actually are.

Both feed a design decision:
  * a visible site attached to a MOVING body has to be re-expressed as a geom (mujoco_warp draws no
    sites), or hidden on both sides at a measured cost;
  * the 29 mover geoms MJWarp cannot UV carry metal1/wood1 cube textures. If those textures are nearly
    uniform, MJWarp's single-texel sample is nearly the right colour and the cost is small; if they are
    patterned, it is not. The number is the texture's own spread, and it is measurable without
    rendering anything.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_envs.render import mjwarp as MW


def main():
    from gymnasium_robotics.envs.franka_kitchen import franka_env
    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    m = mujoco.MjModel.from_xml_path(xml)

    print("== sites ==")
    for s in range(m.nsite):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or f"site{s}"
        b = int(m.site_bodyid[s])
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        vis = int(m.site_group[s]) in MW.DEFAULT_GROUPS
        print(f"  {s:2d} {nm!r:22s} group={int(m.site_group[s])} visible={vis} "
              f"body={bn!r} weldid={int(m.body_weldid[b])} type={int(m.site_type[s])} "
              f"size={np.round(m.site_size[s],4).tolist()} rgba={np.round(m.site_rgba[s],3).tolist()}")

    print("\n== textures ==")
    for t in range(m.ntex):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t) or f"tex{t}"
        w, h = int(m.tex_width[t]), int(m.tex_height[t])
        nc = int(m.tex_nchannel[t])
        adr = int(m.tex_adr[t])
        data = np.asarray(m.tex_data[adr:adr + w * h * nc], np.uint8).reshape(h, w, nc)
        rgb = data[..., :3].astype(np.float64)
        print(f"  {t:2d} {nm!r:18s} type={int(m.tex_type[t])} {w}x{h}x{nc} "
              f"mean={np.round(rgb.reshape(-1,3).mean(0),1).tolist()} "
              f"std={np.round(rgb.reshape(-1,3).std(0),1).tolist()} "
              f"texel00={data[0,0,:3].tolist()}")

    print("\n== materials on the 29 mover texture-gap geoms ==")
    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    vmovers = [g for g in vis if g not in static]
    gaps = {g for g, _, _ in MW.texture_gaps(m)}
    mats = {}
    for g in sorted(gaps & set(vmovers)):
        mid = int(m.geom_matid[g])
        mats.setdefault(mid, []).append(g)
    for mid, gs in mats.items():
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid)
        t = np.asarray(m.mat_texid[mid]).reshape(-1)
        t = t[t >= 0]
        print(f"  material {nm!r}: {len(gs)} geoms, texid={t.tolist()}, "
              f"rgba={np.round(m.mat_rgba[mid],3).tolist()}, "
              f"texrepeat={np.round(m.mat_texrepeat[mid],2).tolist()}")


if __name__ == "__main__":
    main()
