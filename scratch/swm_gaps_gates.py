"""Gates for the swm families stable-envs does not yet render: gymnasium_robotics Fetch, and a
check that swm/PushT-Discrete-v1 draws the same picture as swm/PushT-v1.

Cost order from docs/research/when_does_this_pay_off.md: G3 (decoder stability) first, because two
resets and a shape comparison settle a family in seconds; then G1 (purity), G2 (compression) and G4
(economics).
"""
import json, os, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import gymnasium, stable_worldmodel  # noqa: F401

out = {}

# ---------------------------------------------------------------- PushT-Discrete
try:
    import stable_envs as se
    from stable_envs.testing import conformance as C
    from stable_envs.compat import worldmodel as wm
    rec = {}
    e = gymnasium.make("swm/PushT-Discrete-v1", resolution=224)
    u = e.unwrapped
    e.reset(seed=0, options={"variation": []})
    rng = np.random.default_rng(0)
    frames, states = [], []
    for _ in range(20):
        e.step(e.action_space.sample())
        states.append(wm.pusht_state(u)); frames.append(np.ascontiguousarray(u.render()))
    env = se.make("PushT", batch=20, size=224, device="cuda")
    ours = env.render(np.stack(states).astype(np.float32)).cpu().numpy()
    rec["agreement"] = C.pixel_agreement(ours, np.stack(frames))
    rec["action_space"] = str(e.action_space)
    rec["base_class"] = [c.__name__ for c in type(u).__mro__[:3]]
    e.close()
    out["PushT-Discrete"] = rec
except Exception as ex:
    import traceback; out["PushT-Discrete"] = traceback.format_exc()
print("PushT-Discrete done", flush=True)

# ---------------------------------------------------------------- gymnasium_robotics Fetch
for env_id in ("swm/FetchReach-v3", "swm/FetchPush-v3", "swm/FetchSlide-v3",
               "swm/FetchPickAndPlace-v3"):
    rec = {}
    try:
        e = gymnasium.make(env_id, render_mode="rgb_array")
        u = e.unwrapped
        shapes = []
        for s in range(3):
            e.reset(seed=s)
            m = u.model
            shapes.append({k: int(getattr(m, k)) for k in
                           ("nq", "nv", "ngeom", "nmesh", "nbody", "ntex", "nmat", "nlight",
                            "ncam", "nsite")})
        rec["G3_model_shapes"] = shapes
        rec["G3_shape_constant"] = all(s == shapes[0] for s in shapes)
        rec["G3_textures"] = shapes[0]["ntex"]

        # G1: force two histories to the same (qpos, qvel) and diff the frame
        e.reset(seed=0)
        q0 = u.data.qpos.copy(); v0 = u.data.qvel.copy()
        f0 = np.ascontiguousarray(e.render())
        for _ in range(15):
            e.step(e.action_space.sample())
        u.data.qpos[:] = q0; u.data.qvel[:] = v0
        import mujoco
        mujoco.mj_forward(u.model, u.data)
        f1 = np.ascontiguousarray(e.render())
        rec["G1_max_abs_diff"] = int(np.abs(f0.astype(np.int16) - f1.astype(np.int16)).max())
        rec["frame_shape"] = list(f0.shape)

        # G2: the render-relevant slice is qpos (+ mocap), against the frame
        nq = int(u.model.nq)
        rec["G2_state_bytes"] = nq * 8
        rec["G2_frame_bytes"] = int(np.prod(f0.shape))
        rec["G2_ratio"] = rec["G2_state_bytes"] / rec["G2_frame_bytes"]

        # G4
        t0 = time.perf_counter()
        for _ in range(30):
            e.render()
        rec["G4_official_fps"] = 30 / (time.perf_counter() - t0)
        e.close()
    except Exception:
        import traceback; rec["error"] = traceback.format_exc()
    out[env_id] = rec
    print(env_id, rec.get("G3_shape_constant"), rec.get("G1_max_abs_diff"),
          rec.get("G3_textures"), rec.get("G4_official_fps"), flush=True)

print(json.dumps(out, indent=1, default=str))
with open("/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/gaps_gates.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
