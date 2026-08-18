"""Fetch probe 4: the two fixes, measured separately and together.

FIX A -- the quantisation rule. Probe 3 measured the signed difference on FLAT INTERIOR pixels as
{0: 86.8%, -1: 13.1%} and essentially nothing else, i.e. MJWarp is never ABOVE MuJoCo and is one
level below on an eighth of the frame. That is what "MuJoCo's GL pipeline rounds float->uint8 and
``pack_rgba_to_uint32`` calls ``int()``, which truncates" predicts exactly. The test IS the fix:
replace the pack function with a rounding one before the megakernel is built and re-measure.

FIX B -- the baked static plate, the OGBench design. Fetch's static half (floor, table, skybox) is
91% of the frame and carries 61% of MJWarp's error; it is also provably state-independent here,
because every material has ``reflectance=0`` and the only model light has ``castshadow="false"``, so
a mover casts no shadow and appears in no mirror. Bake it through MuJoCo once and it is exact by
construction; use MJWarp's SEGMENTATION buffer to decide which pixels are movers.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mujoco
import gymnasium
import gymnasium_robotics  # noqa: F401
import torch
import warp as wp
import mujoco_warp as mjw

import fetchmodel as FM

gymnasium.register_envs(gymnasium_robotics)
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.testing.conformance import pixel_agreement                  # noqa: E402
from scratch.fetch.probe3_split import rollout_states, asset_dir             # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
SCENES = {"Reach": ("FetchReach-v4", "reach.xml"),
          "Push": ("FetchPush-v4", "push.xml"),
          "Slide": ("FetchSlide-v4", "slide.xml"),
          "PickAndPlace": ("FetchPickAndPlace-v4", "pick_and_place.xml")}
SIZE = int(os.environ.get("FETCH_SIZE", 480))
N = int(os.environ.get("FETCH_N", 16))
KEEP = []


@wp.func
def pack_rgba_round(r: float, g: float, b: float, a: float) -> wp.uint32:
    """``pack_rgba_to_uint32`` with GL's rounding rule instead of C truncation."""
    return wp.uint32((int(a) << int(24)) | (int(r + 0.5) << int(16))
                     | (int(g + 0.5) << int(8)) | int(b + 0.5))


def patch_rounding(on: bool):
    import mujoco_warp._src.render as R
    if not hasattr(R, "_se_orig_pack"):
        R._se_orig_pack = R.pack_rgba_to_uint32
    R.pack_rgba_to_uint32 = pack_rgba_round if on else R._se_orig_pack


def static_geoms(m, info):
    """Geoms whose world pose cannot depend on the state: welded to the world, not mocap, not the goal."""
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if g == info["target_geom"]:
            continue
        if int(m.body_weldid[b]) == 0 and int(m.body_mocapid[b]) < 0:
            out.append(g)
    return out


def render_mujoco(m, d, r, info, qpos, mp, mq, goal):
    out = []
    for i in range(len(qpos)):
        d.qpos[:] = qpos[i]
        d.qvel[:] = 0.0
        d.mocap_pos[:] = mp[i].reshape(d.mocap_pos.shape)
        d.mocap_quat[:] = mq[i].reshape(d.mocap_quat.shape)
        m.geom_pos[info["target_geom"]] = goal[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=info["cam"])
        out.append(r.render().copy())
    return np.stack(out)


def warp_render(m, info, qpos, mp, mq, goal, size, rounding, bench=False):
    patch_rounding(rounding)
    n = len(qpos)
    md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)
    wm = mjw.put_model(m, batch_sizes={"geom_pos": n})
    wd = mjw.put_data(m, md, nworld=n)
    wp.to_torch(wm.geom_pos)[:, info["target_geom"]] = torch.as_tensor(
        goal, dtype=torch.float32).cuda()
    wp.to_torch(wd.qpos).copy_(torch.as_tensor(qpos, dtype=torch.float32).cuda())
    wp.to_torch(wd.mocap_pos).copy_(torch.as_tensor(mp.reshape(n, -1, 3), dtype=torch.float32).cuda())
    wp.to_torch(wd.mocap_quat).copy_(torch.as_tensor(mq.reshape(n, -1, 4), dtype=torch.float32).cuda())
    rc = mjw.create_render_context(
        m, nworld=n, cam_res=(size, size), render_rgb=True, render_seg=True,
        cam_active=[i == info["cam"] for i in range(m.ncam)],
        enabled_geom_groups=[0, 1, 2], use_textures=True, use_shadows=False,
        render_skybox=True, enable_specular=True)

    def frame():
        mjw.kinematics(wm, wd)
        mjw.camlight(wm, wd)
        mjw.refit_bvh(wm, wd, rc)
        mjw.render(wm, wd, rc)

    frame()
    wp.synchronize()
    fps = None
    if bench:
        t0 = time.perf_counter()
        for _ in range(5):
            frame()
        wp.synchronize()
        fps = 5 * n / (time.perf_counter() - t0)
    rgb = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(n, size, size, 4)[..., [2, 1, 0]]
    seg = wp.to_torch(rc.seg_data).reshape(n, size, size, 2)[..., 0]
    patch_rounding(False)
    return rgb.cpu().numpy(), seg.cpu().numpy(), fps


def main():
    global KEEP
    report = {}
    for name, (env_id, xml) in SCENES.items():
        rep = {}
        env = gymnasium.make(env_id, render_mode="rgb_array", width=SIZE, height=SIZE)
        KEEP.append(env)
        u = env.unwrapped
        env.reset(seed=0)
        env.render()
        pos, fwd, up = FM.free_camera_pose(u.model, u.data, u.mujoco_renderer.viewer.cam)
        quat = FM.quat_from_forward_up(fwd, up)
        m, info = FM.build(os.path.join(asset_dir(), xml), pos, quat,
                           float(u.model.vis.global_.fovy))
        m.vis.quality.offsamples = int(u.model.vis.quality.offsamples)
        d = mujoco.MjData(m)
        rend = mujoco.Renderer(m, width=SIZE, height=SIZE)
        KEEP.append(rend)
        m.geom_group[info["translucent"]] = FM.HIDDEN_GROUP

        qpos, mp, mq, goal = rollout_states(env, N)
        official = []
        for i in range(N):
            u.data.qpos[:] = qpos[i]
            u.data.qvel[:] = 0.0
            u.data.mocap_pos[:] = mp[i].reshape(u.data.mocap_pos.shape)
            u.data.mocap_quat[:] = mq[i].reshape(u.data.mocap_quat.shape)
            u.goal = goal[i].copy()
            mujoco.mj_forward(u.model, u.data)
            official.append(np.ascontiguousarray(u.render()))
        official = np.stack(official)

        # ---- the plate, and the assertion that it IS a constant
        stat = static_geoms(m, info)
        movers = [g for g in range(m.ngeom) if g not in stat]
        keep_group = m.geom_group.copy()
        m.geom_group[movers] = FM.HIDDEN_GROUP
        plates = render_mujoco(m, d, rend, info, qpos[:4], mp[:4], mq[:4], goal[:4])
        rep["plate_is_constant_max"] = int(np.abs(plates.astype(np.int16)
                                                  - plates[:1].astype(np.int16)).max())
        plate = plates[0]
        m.geom_group[:] = keep_group
        m.geom_group[info["translucent"]] = FM.HIDDEN_GROUP
        rep["static_geoms"] = [info["geom_names"][g] for g in stat]
        rep["mover_geoms"] = len(movers)

        res = {}
        for rounding in (False, True):
            rgb, seg, fps = warp_render(m, info, qpos, mp, mq, goal, SIZE, rounding, bench=True)
            comp = np.where((seg[..., None] >= 0) & np.isin(seg, movers)[..., None],
                            rgb, plate[None])
            tag = "round" if rounding else "trunc"
            res[f"{tag}_raw"] = pixel_agreement(rgb, official)
            res[f"{tag}_plate"] = pixel_agreement(comp, official)
            res[f"{tag}_fps"] = fps
            if rounding:
                np.save(f"{OUT}/p4_comp_{name}.npy", comp[:4])
                np.save(f"{OUT}/p4_plate_{name}.npy", plate)
                np.save(f"{OUT}/p4_official_{name}.npy", official[:4])
        rep["results"] = res
        report[name] = rep
        print(f"\n=== {name}  plate_constant={rep['plate_is_constant_max']}  "
              f"static={len(stat)} movers={len(movers)}", flush=True)
        for k in ("trunc_raw", "trunc_plate", "round_raw", "round_plate"):
            v = res[k]
            print(f"  {k:12s} exact {v['exact_frac']:.5f} interior {v['interior_exact_frac']:.6f} "
                  f"mean {v['mean_abs']:.4f} within8 {v['within8_frac']:.5f}", flush=True)
        print(f"  fps trunc {res['trunc_fps']:.0f}  round {res['round_fps']:.0f}", flush=True)

    with open(f"{OUT}/probe4.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("WROTE", f"{OUT}/probe4.json")


if __name__ == "__main__":
    main()
