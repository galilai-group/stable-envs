"""Fetch probe 6: the plate is a constant on Reach and is NOT on the three object scenes.

probe 5 reported max 255 over ~1100 pixels of 1.84 M (0.06%) with every mover hidden. A max is not a
diagnosis; this asks WHICH pixels and WHICH geom they belong to, because the plate design is only
sound if the answer is "none".
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

import fetchmodel as FM

gymnasium.register_envs(gymnasium_robotics)
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from scratch.fetch.probe3_split import rollout_states, asset_dir      # noqa: E402
from scratch.fetch.probe4_plate import static_geoms                   # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
SIZE = 480
KEEP = []


def main():
    global KEEP
    rep = {}
    for name, env_id, xml in (("Push", "FetchPush-v4", "push.xml"),
                              ("Reach", "FetchReach-v4", "reach.xml")):
        env = gymnasium.make(env_id, render_mode="rgb_array", width=SIZE, height=SIZE)
        KEEP.append(env)
        u = env.unwrapped
        env.reset(seed=0)
        env.render()
        pos, fwd, up = FM.free_camera_pose(u.model, u.data, u.mujoco_renderer.viewer.cam)
        quat = FM.quat_from_forward_up(fwd, up)
        m, info = FM.build(os.path.join(asset_dir(), xml), pos, quat, float(u.model.vis.global_.fovy))
        m.vis.quality.offsamples = int(u.model.vis.quality.offsamples)
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, width=SIZE, height=SIZE)
        sg = mujoco.Renderer(m, width=SIZE, height=SIZE)
        sg.enable_segmentation_rendering()
        KEEP += [r, sg]
        m.geom_group[info["translucent"]] = FM.HIDDEN_GROUP
        qpos, mp, mq, goal = rollout_states(env, 8)

        stat = static_geoms(m, info)
        movers = [g for g in range(m.ngeom) if g not in stat]
        m.geom_group[movers] = FM.HIDDEN_GROUP
        plates, segs = [], []
        for i in range(8):
            d.qpos[:] = qpos[i]
            d.qvel[:] = 0.0
            d.mocap_pos[:] = mp[i].reshape(d.mocap_pos.shape)
            d.mocap_quat[:] = mq[i].reshape(d.mocap_quat.shape)
            m.geom_pos[info["target_geom"]] = goal[i]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=info["cam"])
            plates.append(r.render().copy())
            sg.update_scene(d, camera=info["cam"])
            segs.append(sg.render()[..., 0].copy())
        plates, segs = np.stack(plates), np.stack(segs)
        dd = np.abs(plates.astype(np.int16) - plates[:1].astype(np.int16)).max(-1)
        bad = dd > 0
        info_out = dict(n_bad=int(bad.sum()), frac=float(bad.mean()), max=int(dd.max()),
                        static=[info["geom_names"][g] for g in stat],
                        per_frame=[int(x) for x in bad.reshape(8, -1).sum(1)],
                        seg_of_bad={})
        if bad.any():
            ids, cnt = np.unique(segs[bad], return_counts=True)
            info_out["seg_of_bad"] = {str(int(i)): [info["geom_names"].get(int(i), "background"),
                                                    int(c)] for i, c in zip(ids, cnt)}
            ys, xs = np.where(bad[1])
            if len(ys):
                info_out["bbox_frame1"] = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
            # is the plate the same when qpos is the ONLY thing that changes?
            d.qpos[:] = qpos[0]
            d.mocap_pos[:] = mp[1].reshape(d.mocap_pos.shape)
            d.mocap_quat[:] = mq[1].reshape(d.mocap_quat.shape)
            m.geom_pos[info["target_geom"]] = goal[0]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=info["cam"])
            only_mocap = r.render().copy()
            d.mocap_pos[:] = mp[0].reshape(d.mocap_pos.shape)
            d.mocap_quat[:] = mq[0].reshape(d.mocap_quat.shape)
            m.geom_pos[info["target_geom"]] = goal[1]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=info["cam"])
            only_goal = r.render().copy()
            d.qpos[:] = qpos[1]
            m.geom_pos[info["target_geom"]] = goal[0]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=info["cam"])
            only_qpos = r.render().copy()
            base = plates[0].astype(np.int16)
            info_out["isolate"] = dict(
                mocap=int(np.abs(only_mocap.astype(np.int16) - base).max()),
                goal=int(np.abs(only_goal.astype(np.int16) - base).max()),
                qpos=int(np.abs(only_qpos.astype(np.int16) - base).max()))
            # which qpos entry?
            which = {}
            for k in range(m.nq):
                q = qpos[0].copy()
                q[k] = qpos[1][k]
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=info["cam"])
                e = int(np.abs(r.render().astype(np.int16) - base).max())
                if e:
                    which[str(k)] = e
            info_out["qpos_entries_that_move_the_plate"] = which
        m.geom_group[movers] = 0
        rep[name] = info_out
        print(name, json.dumps(info_out, indent=1)[:2500], flush=True)
    with open(f"{OUT}/probe6.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
