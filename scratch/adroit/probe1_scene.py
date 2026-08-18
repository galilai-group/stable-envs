"""What is actually in the four Adroit scenes, and what can MJWarp draw of it?

Decides, BEFORE any renderer is written, whether the Fetch pattern (baked static plate + mujoco_warp
mesh ray-cast for the movers) transfers. Three properties are fatal to that pattern and all three are
visible from the compiled model plus one segmentation render:

  * a TEXTURED geom whose type is not PLANE or MESH -- MJWarp's sample_texture derives UVs for those
    two only, so anything else samples texel (0,0) and renders as one flat colour;
  * a TRANSLUCENT geom -- MJWarp's shader reads color[0..2] and packs a constant 255, so it has no
    alpha path at all;
  * a geom that MOVES, since a static one is baked through MuJoCo itself and is exact regardless.

The cross of (moves) x (MJWarp cannot draw it) is the thing that decides the family, so the pixel
coverage of exactly that set is what this prints.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import sys

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.render import mjwarp as MW

SCENES = {"Door": "adroit_door.xml", "Hammer": "adroit_hammer.xml",
          "Pen": "adroit_pen.xml", "Relocate": "adroit_relocate.xml"}
REF_ID = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
          "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
#: the per-episode MODEL fields each env writes in reset_model(), read off the reference classes
PER_EPISODE = {"Door": ("body_pos", "frame"), "Hammer": ("body_pos", "nail_board"),
               "Pen": ("body_quat", "target"), "Relocate": ("body_pos", "Object")}
GROUPS = MW.DEFAULT_GROUPS


def asset_dir():
    import gymnasium_robotics as gr
    return os.path.join(os.path.dirname(gr.__file__), "envs", "assets", "adroit_hand")


def effective_rgba(m, g):
    rgba = np.asarray(m.geom_rgba[g], np.float64)
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[mid], np.float64)
    return rgba


def main():
    out = {}
    for scene, xml in SCENES.items():
        m = mujoco.MjModel.from_xml_path(os.path.join(asset_dir(), xml))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        gname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}"
        bname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

        # --- who moves ------------------------------------------------------------------
        # A geom moves if its body is not welded to the world, OR if it hangs off the body whose
        # body_pos/body_quat the env rewrites at every reset (which is a per-episode MODEL field,
        # not a qpos entry -- that is the part of Adroit that Fetch does not have).
        field, per_ep_body = PER_EPISODE[scene]
        pe_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, per_ep_body)
        assert pe_bid >= 0, f"{scene}: no body {per_ep_body!r}"
        subtree = set()
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == pe_bid:
                    subtree.add(b)
                    break
                x = int(m.body_parentid[x])

        visible = [g for g in range(m.ngeom) if int(m.geom_group[g]) in GROUPS]
        moves, static = [], []
        for g in visible:
            b = int(m.geom_bodyid[g])
            if int(m.body_weldid[b]) != 0 or b in subtree or int(m.body_mocapid[b]) >= 0:
                moves.append(g)
            else:
                static.append(g)

        # --- what MJWarp cannot draw ----------------------------------------------------
        gaps = {g for g, _, _ in MW.texture_gaps(m, GROUPS)}
        alpha = MW.effective_alpha(m, "geom")
        translucent = {g for g in visible if 0.0 < alpha[g] < 1.0}
        invisible = {g for g in visible if alpha[g] <= 0.0}
        undrawable = (gaps | translucent) - invisible

        # --- pixel coverage of each category --------------------------------------------
        # MuJoCo's own segmentation buffer, at the official camera. Rendered through
        # mujoco.Renderer rather than through gymnasium's viewer: the viewer's segmentation branch
        # does `rgb[:,:,1] * 2**8` on a uint8 array, which is an OverflowError under numpy 2 -- so
        # the official path for this particular buffer does not run at all here.
        #
        # The env is built with render_mode=None (no viewer, no EGL context of its own) and driven
        # only to produce states; the camera is the env's own DEFAULT_CAMERA_CONFIG applied to a
        # free MjvCamera, which is what the official rgb_array render uses.
        import gymnasium, gymnasium_robotics                                    # noqa: F401
        gymnasium.register_envs(gymnasium_robotics)
        env = gymnasium.make(REF_ID[scene])
        u = env.unwrapped
        u.reset(seed=0)
        env.action_space.seed(0)
        cc = u.__class__.__module__
        import importlib
        CC = importlib.import_module(cc).DEFAULT_CAMERA_CONFIG
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.asarray(CC.get("lookat", u.model.stat.center), np.float64)
        cam.distance = float(CC.get("distance", u.model.stat.extent * 1.5))
        cam.azimuth = float(CC.get("azimuth", 90.0))
        cam.elevation = float(CC.get("elevation", -45.0))
        rend = mujoco.Renderer(u.model, width=480, height=480)
        cov = {"moving_undrawable": 0.0, "static_undrawable": 0.0, "moving": 0.0, "static": 0.0}
        n = 8
        for i in range(n):
            for _ in range(int(np.random.default_rng(i).integers(0, 26))):
                u.step(env.action_space.sample())
            rend.enable_segmentation_rendering()
            rend.update_scene(u.data, camera=cam)
            seg = rend.render()
            rend.disable_segmentation_rendering()
            ids = seg[..., 0].astype(np.int64)        # mjtObj type
            objs = seg[..., 1].astype(np.int64)
            gpix = (ids == int(mujoco.mjtObj.mjOBJ_GEOM))
            tot = gpix.size
            for key, sel in (("moving", moves), ("static", static),
                             ("moving_undrawable", [g for g in moves if g in undrawable]),
                             ("static_undrawable", [g for g in static if g in undrawable])):
                sel_a = np.asarray(sel, np.int64) if sel else np.zeros(0, np.int64)
                cov[key] += float((np.isin(objs, sel_a) & gpix).sum() / tot) / n
            u.reset(seed=100 + i)
        # env is NOT closed and the renderer is NOT closed: releasing an EGL context corrupts every
        # other live one in the process, silently.

        out[scene] = {
            "ngeom_visible": len(visible), "n_moving": len(moves), "n_static": len(static),
            "per_episode_field": f"{field}[{per_ep_body}]",
            "texture_gap_geoms": sorted(gname(g) for g in gaps),
            "translucent_geoms": {gname(g): round(float(alpha[g]), 3) for g in sorted(translucent)},
            "moving_undrawable": sorted(gname(g) for g in moves if g in undrawable),
            "static_undrawable": sorted(gname(g) for g in static if g in undrawable),
            "coverage": {k: round(v, 5) for k, v in cov.items()},
        }
        print(f"\n===== {scene} =====")
        print(json.dumps(out[scene], indent=2))

    with open("/mnt/vast/home/randall.balestriero/se_logs/adroit/probe1.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
