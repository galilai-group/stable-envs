"""Fetch probe 2: the diagnostic ladder, then an MJWarp config sweep.

The ladder separates three claims that a single parity number would fold together:

  L1  official env (free camera, MuJoCo)  vs  surgical model (se:cam, MuJoCo), markers KEPT
      -> is the camera and the material surgery right?
  L2  surgical model in MuJoCo, markers HIDDEN  vs  markers KEPT
      -> what do the four translucent mocap markers cost?
  L3  surgical model in MJWarp  vs  surgical model in MuJoCo (markers hidden on both sides)
      -> what does the renderer cost, with the model out of the picture?
  L4  MJWarp  vs  the OFFICIAL frame. The number that ships.
"""
from __future__ import annotations

import json
import os
import sys

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
from stable_envs.testing.conformance import pixel_agreement          # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
SCENES = {"Reach": ("FetchReach-v4", "reach.xml"),
          "Push": ("FetchPush-v4", "push.xml"),
          "Slide": ("FetchSlide-v4", "slide.xml"),
          "PickAndPlace": ("FetchPickAndPlace-v4", "pick_and_place.xml")}
SIZE = int(os.environ.get("FETCH_SIZE", 480))
N = int(os.environ.get("FETCH_N", 16))


def asset_dir():
    import gymnasium_robotics as gr
    return os.path.join(os.path.dirname(gr.__file__), "envs", "assets", "fetch")


def sample_states(env, n, seed=100):
    """The env's OWN reset distribution, plus the two quantities that are not in qpos."""
    u = env.unwrapped
    qpos, mocap_pos, mocap_quat, goal, qvel = [], [], [], [], []
    for i in range(n):
        env.reset(seed=seed + i)
        qpos.append(u.data.qpos.copy())
        qvel.append(u.data.qvel.copy())
        mocap_pos.append(u.data.mocap_pos.copy().reshape(-1))
        mocap_quat.append(u.data.mocap_quat.copy().reshape(-1))
        goal.append(np.asarray(u.goal, np.float64).copy())
    return (np.stack(qpos), np.stack(qvel), np.stack(mocap_pos), np.stack(mocap_quat),
            np.stack(goal))


def official_render(env, qpos, mocap_pos, mocap_quat, goal):
    u = env.unwrapped
    m, d = u.model, u.data
    out = []
    for i in range(len(qpos)):
        d.qpos[:] = qpos[i]
        d.qvel[:] = 0.0
        d.mocap_pos[:] = mocap_pos[i].reshape(d.mocap_pos.shape)
        d.mocap_quat[:] = mocap_quat[i].reshape(d.mocap_quat.shape)
        u.goal = goal[i].copy()
        mujoco.mj_forward(m, d)
        out.append(np.ascontiguousarray(u.render()))
    return np.stack(out)


def surgical_render(m, d, r, info, qpos, mocap_pos, mocap_quat, goal, site_body_pos):
    out = []
    for i in range(len(qpos)):
        d.qpos[:] = qpos[i]
        d.qvel[:] = 0.0
        d.mocap_pos[:] = mocap_pos[i].reshape(d.mocap_pos.shape)
        d.mocap_quat[:] = mocap_quat[i].reshape(d.mocap_quat.shape)
        m.geom_pos[info["target_geom"]] = goal[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=info["cam"])
        out.append(r.render().copy())
    return np.stack(out)


def main():
    report = {}
    for name, (env_id, xml) in SCENES.items():
        rep = {}
        env = gymnasium.make(env_id, render_mode="rgb_array", width=SIZE, height=SIZE)
        u = env.unwrapped
        env.reset(seed=0)
        env.render()
        vcam = u.mujoco_renderer.viewer.cam
        pos, fwd, up = FM.free_camera_pose(u.model, u.data, vcam)
        quat = FM.quat_from_forward_up(fwd, up)
        fovy = float(u.model.vis.global_.fovy)
        rep["camera"] = dict(pos=pos.tolist(), quat=quat.tolist(), fovy=fovy)

        m, info = FM.build(os.path.join(asset_dir(), xml), pos, quat, fovy)
        m.vis.quality.offsamples = int(u.model.vis.quality.offsamples)
        d = mujoco.MjData(m)
        rend = mujoco.Renderer(m, width=SIZE, height=SIZE)
        rep["surgery"] = dict(ngeom=int(m.ngeom), nmat=int(m.nmat), target=info["target_geom"],
                              translucent=info["translucent"])

        qpos, qvel, mp, mq, goal = sample_states(env, N)
        # the target site's body is static, so `site_pos = goal - body_pos` is the reference's own
        # placement rule; our geom lives on the worldbody so it takes the goal directly.
        official = official_render(env, qpos, mp, mq, goal)

        # ---- L1: surgical model, markers kept
        ours_keep = surgical_render(m, d, rend, info, qpos, mp, mq, goal, None)
        rep["L1_model_and_camera"] = pixel_agreement(ours_keep, official)

        # ---- L2: what the translucent mocap markers cost
        keep_group = m.geom_group.copy()
        m.geom_group[info["translucent"]] = FM.HIDDEN_GROUP
        ours_hide = surgical_render(m, d, rend, info, qpos, mp, mq, goal, None)
        rep["L2_markers_cost"] = pixel_agreement(ours_hide, official)
        rep["L2_markers_vs_kept"] = pixel_agreement(ours_hide, ours_keep)

        # ---- qvel irrelevance, measured
        d.qpos[:] = qpos[0]
        d.qvel[:] = 0.0
        d.mocap_pos[:] = mp[0].reshape(d.mocap_pos.shape)
        d.mocap_quat[:] = mq[0].reshape(d.mocap_quat.shape)
        m.geom_pos[info["target_geom"]] = goal[0]
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera=info["cam"])
        a = rend.render().copy()
        d.qvel[:] = qvel[0] * 100.0 + 1.0
        mujoco.mj_forward(m, d)
        rend.update_scene(d, camera=info["cam"])
        b = rend.render().copy()
        rep["qvel_irrelevant_max"] = int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())

        # ---- L3/L4: MJWarp
        mw = mujoco.MjData(m)
        mujoco.mj_forward(m, mw)
        wm = mjw.put_model(m, batch_sizes={"geom_pos": N})
        wd = mjw.put_data(m, mw, nworld=N)
        gp = wp.to_torch(wm.geom_pos)                      # (N, ngeom, 3)
        gp[:, info["target_geom"]] = torch.as_tensor(goal, dtype=gp.dtype, device=gp.device)
        wp.to_torch(wd.qpos).copy_(torch.as_tensor(qpos, dtype=torch.float32).cuda())
        wp.to_torch(wd.mocap_pos).copy_(
            torch.as_tensor(mp.reshape(N, -1, 3), dtype=torch.float32).cuda())
        wp.to_torch(wd.mocap_quat).copy_(
            torch.as_tensor(mq.reshape(N, -1, 4), dtype=torch.float32).cuda())

        sweep = {}
        for shadows in (True, False):
            for specular in (True, False):
                for skybox in (True,):
                    rc = mjw.create_render_context(
                        m, nworld=N, cam_res=(SIZE, SIZE), render_rgb=True,
                        cam_active=[i == info["cam"] for i in range(m.ncam)],
                        enabled_geom_groups=[0, 1, 2], use_textures=True,
                        use_shadows=shadows, render_skybox=skybox, enable_specular=specular)
                    mjw.kinematics(wm, wd)
                    mjw.camlight(wm, wd)
                    mjw.refit_bvh(wm, wd, rc)
                    mjw.render(wm, wd, rc)
                    wp.synchronize()
                    bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(N, SIZE, SIZE, 4)
                    rgb = bgra[..., [2, 1, 0]].cpu().numpy()
                    k = f"sh{int(shadows)}_sp{int(specular)}_sky{int(skybox)}"
                    sweep[k] = dict(vs_official=pixel_agreement(rgb, official),
                                    vs_mujoco_same_model=pixel_agreement(rgb, ours_hide))
                    if k == "sh1_sp1_sky1":
                        np.save(f"{OUT}/mjw_{name}.npy", rgb[:4])
                        np.save(f"{OUT}/mjmodel_{name}.npy", ours_hide[:4])
                        np.save(f"{OUT}/official_{name}.npy", official[:4])
                    del rc
        rep["mjwarp_sweep"] = sweep
        m.geom_group[:] = keep_group

        report[name] = rep
        print(f"\n=== {name}", flush=True)
        print("  L1 model+camera:", {k: round(v, 5) if isinstance(v, float) else v
                                     for k, v in rep["L1_model_and_camera"].items()}, flush=True)
        print("  L2 markers hidden vs official:",
              {k: round(v, 5) if isinstance(v, float) else v
               for k, v in rep["L2_markers_cost"].items()}, flush=True)
        print("  qvel irrelevant (max/255):", rep["qvel_irrelevant_max"], flush=True)
        for k, v in sweep.items():
            print(f"  {k}: vs_official {v['vs_official']['exact_frac']:.4f} exact / "
                  f"interior {v['vs_official']['interior_exact_frac']} / "
                  f"mean {v['vs_official']['mean_abs']:.3f} | vs_mj "
                  f"{v['vs_mujoco_same_model']['exact_frac']:.4f} / "
                  f"mean {v['vs_mujoco_same_model']['mean_abs']:.3f}", flush=True)

    with open(f"{OUT}/probe2.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("WROTE", f"{OUT}/probe2.json")


if __name__ == "__main__":
    main()
