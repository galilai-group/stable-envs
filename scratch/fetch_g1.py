"""G1 for the Fetch family, with the COMPLETE render-relevant state.

The first pass restored ``qpos`` and ``qvel`` and measured 29-138/255, which reads like a purity
failure and is not one: MujocoFetchEnv drives the gripper through a MOCAP body and draws the target
as a site whose position it stores outside the physics state. So the probe was missing state, and
the honest way to say so is to add it and re-measure rather than to report the first number.
"""
import json, os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, gymnasium, stable_worldmodel  # noqa: F401

out = {}
for env_id in ("swm/FetchReach-v3", "swm/FetchPush-v3", "swm/FetchSlide-v3",
               "swm/FetchPickAndPlace-v3"):
    e = gymnasium.make(env_id, render_mode="rgb_array")
    u = e.unwrapped
    e.reset(seed=0)
    snap = {"qpos": u.data.qpos.copy(), "qvel": u.data.qvel.copy(),
            "mocap_pos": u.data.mocap_pos.copy(), "mocap_quat": u.data.mocap_quat.copy(),
            "goal": np.asarray(u.goal).copy()}
    f0 = np.ascontiguousarray(e.render())
    for _ in range(15):
        e.step(e.action_space.sample())
    u.goal = snap["goal"].copy()
    if hasattr(u, "_utils") or True:
        try:
            sid = mujoco.mj_name2id(u.model, mujoco.mjtObj.mjOBJ_SITE, "target0")
            u.model.site_pos[sid] = snap["goal"] - u.model.body_pos[u.model.site_bodyid[sid]]
        except Exception:
            pass
    u.data.qpos[:] = snap["qpos"]; u.data.qvel[:] = snap["qvel"]
    u.data.mocap_pos[:] = snap["mocap_pos"]; u.data.mocap_quat[:] = snap["mocap_quat"]
    mujoco.mj_forward(u.model, u.data)
    f1 = np.ascontiguousarray(e.render())
    d = np.abs(f0.astype(np.int16) - f1.astype(np.int16))
    out[env_id] = {"max_abs": int(d.max()), "exact_frac": float((d.max(-1) == 0).mean()),
                   "state_floats": int(u.model.nq + u.model.nv + u.data.mocap_pos.size
                                       + u.data.mocap_quat.size + 3),
                   "frame_bytes": int(np.prod(f0.shape))}
    out[env_id]["G2_ratio"] = out[env_id]["state_floats"] * 8 / out[env_id]["frame_bytes"]
    print(env_id, out[env_id], flush=True)
    e.close()
print(json.dumps(out, indent=1))
json.dump(out, open("/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/fetch_g1.json","w"), indent=1)
