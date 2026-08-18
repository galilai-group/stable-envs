"""Fetch probe 3: WHERE does MJWarp's disagreement live, and is it a quantisation rule?

Two questions, both answered by splitting the frame with MuJoCo's OWN segmentation buffer rather
than by guessing:

  1. per geom class (skybox / floor / table / robot meshes / gripper boxes / object / target), what
     fraction of the frame is it and what fraction of MJWarp's error does it carry?
  2. is the +-1 residual the quantisation RULE? MuJoCo's GL pipeline rounds float->uint8; MJWarp's
     ``pack_rgba_to_uint32`` calls ``int()``, which TRUNCATES. If that is the mechanism the signed
     difference on flat interior pixels is in {0, -1} and nothing else.

Every renderer built here is kept alive for the life of the process. An earlier version rebound
``rend`` per scene, which released an EGL context and turned every LATER official render black
(mean 0.0, one distinct colour) -- exactly the failure conformance.py warns about, reproduced.
"""
from __future__ import annotations

import collections
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
from stable_envs.testing.conformance import pixel_agreement, interior_mask   # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
SCENES = {"Reach": ("FetchReach-v4", "reach.xml"),
          "PickAndPlace": ("FetchPickAndPlace-v4", "pick_and_place.xml")}
SIZE = int(os.environ.get("FETCH_SIZE", 480))
N = int(os.environ.get("FETCH_N", 16))
KEEP = []                       # every EGL-owning object, kept alive on purpose


def asset_dir():
    import gymnasium_robotics as gr
    return os.path.join(os.path.dirname(gr.__file__), "envs", "assets", "fetch")


def rollout_states(env, n, seed=0, max_steps=25):
    """Reset, then a random number of random actions: the distribution a dataset actually holds.

    Reset alone is not enough here and that is a measurement, not an opinion: ``_reset_sim`` restores
    ``initial_state`` and randomises only the object, so EVERY reset state has the identical arm pose
    and a parity number computed from them says nothing about the 11 joints that make up most of the
    picture.
    """
    u = env.unwrapped
    rng = np.random.default_rng(seed)
    q, mp, mq, g = [], [], [], []
    for i in range(n):
        env.reset(seed=seed * 1000 + i)
        for _ in range(int(rng.integers(0, max_steps + 1))):
            env.step(env.action_space.sample())
        q.append(u.data.qpos.copy())
        mp.append(u.data.mocap_pos.copy().reshape(-1))
        mq.append(u.data.mocap_quat.copy().reshape(-1))
        g.append(np.asarray(u.goal, np.float64).copy())
    return np.stack(q), np.stack(mp), np.stack(mq), np.stack(g)


def place(m, d, info, qpos, mp, mq, goal):
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    d.mocap_pos[:] = mp.reshape(d.mocap_pos.shape)
    d.mocap_quat[:] = mq.reshape(d.mocap_quat.shape)
    m.geom_pos[info["target_geom"]] = goal
    mujoco.mj_forward(m, d)


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
        seg = mujoco.Renderer(m, width=SIZE, height=SIZE)
        seg.enable_segmentation_rendering()
        KEEP += [rend, seg]

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

        m.geom_group[info["translucent"]] = FM.HIDDEN_GROUP
        mine, segs = [], []
        for i in range(N):
            place(m, d, info, qpos[i], mp[i], mq[i], goal[i])
            rend.update_scene(d, camera=info["cam"])
            mine.append(rend.render().copy())
            seg.update_scene(d, camera=info["cam"])
            segs.append(seg.render()[..., 0].copy())        # geom id, -1 = background
        mine = np.stack(mine)
        segs = np.stack(segs)

        # ---- MJWarp on the same model
        md = mujoco.MjData(m)
        mujoco.mj_forward(m, md)
        wm = mjw.put_model(m, batch_sizes={"geom_pos": N})
        wd = mjw.put_data(m, md, nworld=N)
        wp.to_torch(wm.geom_pos)[:, info["target_geom"]] = torch.as_tensor(
            goal, dtype=torch.float32).cuda()
        wp.to_torch(wd.qpos).copy_(torch.as_tensor(qpos, dtype=torch.float32).cuda())
        wp.to_torch(wd.mocap_pos).copy_(
            torch.as_tensor(mp.reshape(N, -1, 3), dtype=torch.float32).cuda())
        wp.to_torch(wd.mocap_quat).copy_(
            torch.as_tensor(mq.reshape(N, -1, 4), dtype=torch.float32).cuda())
        rc = mjw.create_render_context(
            m, nworld=N, cam_res=(SIZE, SIZE), render_rgb=True,
            cam_active=[i == info["cam"] for i in range(m.ncam)],
            enabled_geom_groups=[0, 1, 2], use_textures=True, use_shadows=True,
            render_skybox=True, enable_specular=True)
        mjw.kinematics(wm, wd)
        mjw.camlight(wm, wd)
        mjw.refit_bvh(wm, wd, rc)
        mjw.render(wm, wd, rc)
        wp.synchronize()
        warpf = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(
            N, SIZE, SIZE, 4)[..., [2, 1, 0]].cpu().numpy()

        rep["overall_vs_official"] = pixel_agreement(warpf, official)
        rep["overall_vs_mujoco"] = pixel_agreement(warpf, mine)
        rep["mujoco_surgical_vs_official"] = pixel_agreement(mine, official)

        # ---- class split
        names = info["geom_names"]
        cls = np.full(segs.shape, "background", dtype=object)
        for g, nm in names.items():
            if nm == "floor0":
                c = "floor"
            elif nm == "se:target":
                c = "target"
            elif nm == "object0":
                c = "object"
            elif nm.startswith("robot0:") and int(m.geom_type[g]) == 7:
                c = "robot_mesh"
            elif nm.startswith("robot0:"):
                c = "robot_box"
            elif int(m.geom_type[g]) == 6:
                c = "table"
            else:
                c = f"other:{nm}"
            cls[segs == g] = c
        dmj = np.abs(warpf.astype(np.int16) - mine.astype(np.int16)).max(-1)
        doff = np.abs(warpf.astype(np.int16) - official.astype(np.int16)).max(-1)
        tot_err = float(doff.sum())
        split = {}
        for c in sorted(set(cls.reshape(-1).tolist())):
            msk = cls == c
            split[c] = dict(frac=float(msk.mean()),
                            exact_vs_official=float((doff[msk] == 0).mean()),
                            mean_vs_official=float(doff[msk].mean()),
                            within1=float((doff[msk] <= 1).mean()),
                            within8=float((doff[msk] <= 8).mean()),
                            share_of_error=float(doff[msk].sum() / max(tot_err, 1)),
                            exact_vs_mujoco=float((dmj[msk] == 0).mean()))
        rep["class_split"] = split
        rep["static_frac"] = float(((cls == "floor") | (cls == "table")
                                    | (cls == "background")).mean())

        # ---- quantisation-rule test on FLAT interior pixels
        im = interior_mask(mine)
        sd = (warpf[:, :-1, :-1].astype(np.int16) - mine[:, :-1, :-1].astype(np.int16))[im]
        h = collections.Counter(sd.reshape(-1).tolist())
        n_tot = int(sd.size)
        rep["signed_interior_hist"] = {str(k): [int(v), round(100 * v / n_tot, 4)]
                                       for k, v in sorted(h.items()) if abs(k) <= 3}
        rep["interior_frac_le1"] = float((np.abs(sd) <= 1).mean())
        rep["interior_frac_eq0"] = float((sd == 0).mean())
        rep["interior_frac_eqm1"] = float((sd == -1).mean())
        rep["interior_frac_eqp1"] = float((sd == 1).mean())

        np.save(f"{OUT}/p3_official_{name}.npy", official[:4])
        np.save(f"{OUT}/p3_mine_{name}.npy", mine[:4])
        np.save(f"{OUT}/p3_warp_{name}.npy", warpf[:4])
        np.save(f"{OUT}/p3_seg_{name}.npy", segs[:4])
        np.savez(f"{OUT}/p3_states_{name}.npz", qpos=qpos, mocap_pos=mp, mocap_quat=mq, goal=goal)

        report[name] = rep
        print(f"\n=== {name}", flush=True)
        print("  MuJoCo(surgical) vs official:", rep["mujoco_surgical_vs_official"], flush=True)
        print("  MJWarp vs official:", rep["overall_vs_official"], flush=True)
        print("  static frac:", round(rep["static_frac"], 4), flush=True)
        for c, v in split.items():
            print(f"    {c:14s} frac {v['frac']:.4f} exact {v['exact_vs_official']:.4f} "
                  f"mean {v['mean_vs_official']:.3f} w1 {v['within1']:.4f} "
                  f"err_share {v['share_of_error']:.3f}", flush=True)
        print("  signed interior hist:", rep["signed_interior_hist"], flush=True)

    with open(f"{OUT}/probe3.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("WROTE", f"{OUT}/probe3.json")


if __name__ == "__main__":
    main()
