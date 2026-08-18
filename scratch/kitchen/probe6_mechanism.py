"""Probe 6: WHY does a baked plate get 20% of its non-mover pixels wrong? Shadows, reflections, or both.

probe5 measured the damage; this attributes it. The knobs are mjvScene RENDER flags, not model fields:
setting ``model.vis.quality.shadowsize = 0`` after a Renderer exists is a no-op, because the shadow FBO
was allocated when the MjrContext was built. ``scn.flags[mjRND_SHADOW]`` is the switch that works.

Every site is hidden on both sides here, so the ``end_effector`` sphere -- which mujoco_warp cannot draw
at all -- is not a confound, and the plate's state-independence can be checked honestly.
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
from stable_envs.testing.conformance import interior_mask
from probe5_contamination import build, pool

RES = 480


def main():
    m, d, camid = build()
    r = mujoco.Renderer(m, width=RES, height=RES)
    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    static = [g for g in range(m.ngeom)
              if int(m.body_weldid[m.geom_bodyid[g]]) == 0
              and int(m.body_mocapid[m.geom_bodyid[g]]) < 0]
    vmovers = np.array([g for g in vis if g not in static], np.int64)

    # every site hidden, on both sides, for the whole probe
    m.site_group[:] = 5

    def shot(q, hide=None, seg=False, shadow=True, reflect=True):
        keep = np.array(m.geom_group, copy=True)
        if hide is not None:
            m.geom_group[hide] = 5
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        (r.enable_segmentation_rendering if seg else r.disable_segmentation_rendering)()
        r.update_scene(d, camera=camid)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadow)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = int(reflect)
        img = r.render().copy()
        r.disable_segmentation_rendering()
        m.geom_group[:] = keep
        return img

    states = pool(12)

    # is MuJoCo's reflection pass even firing on this scene?
    a = shot(states[0], reflect=True)
    b = shot(states[0], reflect=False)
    dr = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    a2 = shot(states[0], shadow=True)
    b2 = shot(states[0], shadow=False)
    ds = np.abs(a2.astype(np.int16) - b2.astype(np.int16)).max(-1)
    print(f"== do the passes fire at all? (one state, full scene) ==")
    print(f"  reflection pass changes {float((dr>0).mean()):.4f} of pixels, max {int(dr.max())}")
    print(f"  shadow     pass changes {float((ds>0).mean()):.4f} of pixels, max {int(ds.max())}")

    for shadow, reflect in ((True, True), (False, True), (True, False), (False, False)):
        plate = shot(states[0], hide=vmovers, shadow=shadow, reflect=reflect)
        pl2 = shot(states[-1], hide=vmovers, shadow=shadow, reflect=reflect)
        dp = int(np.abs(plate.astype(np.int16) - pl2.astype(np.int16)).max())

        ex, iex, mean_out, sil_f = [], [], [], []
        for q in states:
            full = shot(q, shadow=shadow, reflect=reflect)
            sg = shot(q, seg=True, shadow=shadow, reflect=reflect)
            sil = np.isin(sg[..., 0], vmovers) & (sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            outside = ~sil
            dd = np.abs(full.astype(np.int16) - plate.astype(np.int16)).max(-1)
            ex.append(float((dd[outside] == 0).mean()))
            mean_out.append(float(dd[outside].mean()))
            sil_f.append(float(sil.mean()))
            im = interior_mask(full[None])[0]
            oi = outside[:-1, :-1] & im
            iex.append(float((dd[:-1, :-1][oi] == 0).mean()) if oi.any() else np.nan)
        print(f"\n== shadow={shadow} reflect={reflect} ==")
        print(f"  plate state-independence: max {dp}/255")
        print(f"  mover silhouette: {np.mean(sil_f):.4f}")
        print(f"  non-mover exact: {np.mean(ex):.6f}   mean|diff| {np.mean(mean_out):.4f}/255")
        print(f"  non-mover INTERIOR exact: {np.nanmean(iex):.6f} (worst {np.nanmin(iex):.6f})")


if __name__ == "__main__":
    main()
