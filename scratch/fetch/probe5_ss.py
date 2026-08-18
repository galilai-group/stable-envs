"""Fetch probe 5: supersampling, the composite variants, throughput, and the plate-invariance check.

Variants scored against the official renderer, all with the GL rounding rule applied:

  A  1x MJWarp, 1x plate composite                       (probe 4: 0.9228 exact / 0.99868 interior)
  C  2x MJWarp raw, box-downsampled to 1x
  D  2x composite where a pixel has ANY mover sub-sample, 1x plate where it has none
     -- so a static pixel is MuJoCo's own 1x rasterisation (exact, including the table's own
     silhouette against the floor) and a mover pixel is antialiased against the 2x plate.

Also: is the plate really a constant? probe 4 reported max 255 on the three object scenes and 0 on
Reach, which is either a real state dependence or a handful of pixels; a max is not an answer.
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
from scratch.fetch.probe4_plate import patch_rounding, static_geoms, render_mujoco  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
SCENES = {"Reach": ("FetchReach-v4", "reach.xml"),
          "Push": ("FetchPush-v4", "push.xml"),
          "Slide": ("FetchSlide-v4", "slide.xml"),
          "PickAndPlace": ("FetchPickAndPlace-v4", "pick_and_place.xml")}
SIZE = int(os.environ.get("FETCH_SIZE", 480))
N = int(os.environ.get("FETCH_N", 16))
KEEP = []


def down(x, f):
    """Exact f x f box average of a uint8 stack, rounded once."""
    n, H, W, c = x.shape
    v = x.astype(np.float64).reshape(n, H // f, f, W // f, f, c).mean(axis=(2, 4))
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)


def warp_frames(m, info, qpos, mp, mq, goal, size, movers, bench_batches=()):
    n = len(qpos)
    md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)
    patch_rounding(True)
    wm = mjw.put_model(m, batch_sizes={"geom_pos": n})
    wd = mjw.put_data(m, md, nworld=n)
    wp.to_torch(wm.geom_pos)[:, info["target_geom"]] = torch.as_tensor(goal, dtype=torch.float32).cuda()
    wp.to_torch(wd.qpos).copy_(torch.as_tensor(qpos, dtype=torch.float32).cuda())
    wp.to_torch(wd.mocap_pos).copy_(torch.as_tensor(mp.reshape(n, -1, 3), dtype=torch.float32).cuda())
    wp.to_torch(wd.mocap_quat).copy_(torch.as_tensor(mq.reshape(n, -1, 4), dtype=torch.float32).cuda())
    rc = mjw.create_render_context(
        m, nworld=n, cam_res=(size, size), render_rgb=True, render_seg=True,
        cam_active=[i == info["cam"] for i in range(m.ncam)],
        enabled_geom_groups=[0, 1, 2], use_textures=True, use_shadows=False,
        render_skybox=True, enable_specular=True)
    for _ in range(2):
        mjw.kinematics(wm, wd)
        mjw.camlight(wm, wd)
        mjw.refit_bvh(wm, wd, rc)
        mjw.render(wm, wd, rc)
    wp.synchronize()
    patch_rounding(False)
    rgb = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(n, size, size, 4)[..., [2, 1, 0]].cpu().numpy()
    seg = wp.to_torch(rc.seg_data).reshape(n, size, size, 2)[..., 0].cpu().numpy()
    mover_mask = np.isin(seg, movers)
    return rgb, mover_mask


def bench(m, info, qpos0, size, batches):
    out = {}
    md = mujoco.MjData(m)
    mujoco.mj_forward(m, md)
    for B in batches:
        patch_rounding(True)
        wm = mjw.put_model(m, batch_sizes={"geom_pos": B})
        wd = mjw.put_data(m, md, nworld=B)
        rc = mjw.create_render_context(
            m, nworld=B, cam_res=(size, size), render_rgb=True, render_seg=True,
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
        with wp.ScopedCapture() as cap:
            frame()
        graph = cap.graph
        wp.capture_launch(graph)
        wp.synchronize()
        t0 = time.perf_counter()
        reps = max(2, int(20 // max(1, B // 64)))
        for _ in range(reps):
            wp.capture_launch(graph)
        wp.synchronize()
        out[B] = reps * B / (time.perf_counter() - t0)
        patch_rounding(False)
        del rc, wd, wm, graph
    return out


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
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 2 * SIZE)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), 2 * SIZE)
        d = mujoco.MjData(m)
        r1 = mujoco.Renderer(m, width=SIZE, height=SIZE)
        r2 = mujoco.Renderer(m, width=2 * SIZE, height=2 * SIZE)
        KEEP += [r1, r2]
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

        stat = static_geoms(m, info)
        movers = np.array([g for g in range(m.ngeom) if g not in stat])
        keep_group = m.geom_group.copy()
        m.geom_group[movers] = FM.HIDDEN_GROUP
        p1 = render_mujoco(m, d, r1, info, qpos[:8], mp[:8], mq[:8], goal[:8])
        p2 = render_mujoco(m, d, r2, info, qpos[:2], mp[:2], mq[:2], goal[:2])
        dd = np.abs(p1.astype(np.int16) - p1[:1].astype(np.int16)).max(-1)
        rep["plate_invariance"] = dict(max=int(dd.max()), frac_nonzero=float((dd > 0).mean()),
                                       n_nonzero=int((dd > 0).sum()))
        plate1, plate2 = p1[0], p2[0]
        m.geom_group[:] = keep_group
        m.geom_group[info["translucent"]] = FM.HIDDEN_GROUP

        rgb1, mv1 = warp_frames(m, info, qpos, mp, mq, goal, SIZE, movers)
        rgb2, mv2 = warp_frames(m, info, qpos, mp, mq, goal, 2 * SIZE, movers)

        A = np.where(mv1[..., None], rgb1, plate1[None])
        C = down(rgb2, 2)
        comp2 = np.where(mv2[..., None], rgb2, plate2[None])
        Dd = down(comp2, 2)
        any_mover = mv2.reshape(N, SIZE, 2, SIZE, 2).any(axis=(2, 4))
        D = np.where(any_mover[..., None], Dd, plate1[None])

        rep["A_1x_plate"] = pixel_agreement(A, official)
        rep["B_1x_raw"] = pixel_agreement(rgb1, official)
        rep["C_2x_raw"] = pixel_agreement(C, official)
        rep["D_2x_plate"] = pixel_agreement(D, official)
        rep["fps"] = bench(m, info, qpos[:1], SIZE, (64, 256))
        rep["fps_2x"] = bench(m, info, qpos[:1], 2 * SIZE, (64,))

        np.save(f"{OUT}/p5_D_{name}.npy", D[:4])
        np.save(f"{OUT}/p5_official_{name}.npy", official[:4])
        report[name] = rep
        print(f"\n=== {name}  plate_invariance {rep['plate_invariance']}", flush=True)
        for k in ("B_1x_raw", "A_1x_plate", "C_2x_raw", "D_2x_plate"):
            v = rep[k]
            print(f"  {k:12s} exact {v['exact_frac']:.5f} interior {v['interior_exact_frac']:.6f} "
                  f"mean {v['mean_abs']:.4f} within8 {v['within8_frac']:.5f} max {v['max_abs']}",
                  flush=True)
        print("  fps 1x:", {k: round(v) for k, v in rep["fps"].items()},
              " 2x:", {k: round(v) for k, v in rep["fps_2x"].items()}, flush=True)

    with open(f"{OUT}/probe5.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("WROTE", f"{OUT}/probe5.json")


if __name__ == "__main__":
    main()
