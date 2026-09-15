"""Step 7 / W5 phase 0 -- G1 purity probes, state/frame byte accounting, and the two frozen bars.

Three questions, measured on the registered swm defaults (no kwargs, exactly as
`stable_worldmodel.envs.__init__` registers them):

* **G1 purity.** Is the frame a function of the state vector?  Two probes: the same
  (qpos, qvel) reached twice through different histories, and a re-render with no state
  change.  Anything that differs is a hidden bit; the probe then names it by diffing the
  MjModel/MjData fields the renderer reads.
* **Bytes.** state bytes vs frame bytes at the registered resolution.
* **The bars.** Official `mujoco.Renderer` throughput at the registered default
  resolution on one thread, and the one-core JPEG-q90 decode fps of its own frames --
  the Step-6 `profile_note` protocol.  `S_cpu >= 5` and `cores_cpu >= 1` are then
  absolute fps targets for any stable-envs renderer of this scene.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

RENDER_FIELDS = (
    "geom_rgba", "mat_rgba", "light_diffuse", "light_pos", "light_dir",
    "cam_pos", "cam_quat", "cam_fovy", "geom_size", "geom_matid",
)


def set_state(u, qpos, qvel):
    """`set_state` through the public signature the env declares.

    `SceneEnv.set_state` takes two extra REQUIRED keyword arguments, `button_state_0`
    and `button_state_1`, and raises `KeyError` without them -- two discrete bits of
    appearance that are not in `qpos`.  This helper supplies whatever the env is
    currently holding, so the probe measures the frame's dependence on (qpos, qvel)
    alone."""
    n = getattr(u, "_num_buttons", 0)
    if n:
        cur = list(getattr(u, "_cur_button_states", [0] * n))
        kw = {f"button_state_{i}": cur[i] for i in range(n)}
        u.set_state(qpos, qvel, **kw)
    else:
        u.set_state(qpos, qvel)


def make(env_id):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    return gym.make(env_id)


def snap_model(m):
    return {f: np.array(getattr(m, f)).copy() for f in RENDER_FIELDS}


def probe(env_id, out_dir, reps):
    import mujoco
    env = make(env_id)
    u = env.unwrapped
    env.reset(seed=0)
    m, d = u.model, u.data
    h, w = u._render_height, u._render_width

    rec = {"env_id": env_id, "h": h, "w": w,
           "host": socket.gethostname(), "platform": platform.platform(),
           "mujoco": mujoco.__version__, "mujoco_gl": os.environ.get("MUJOCO_GL"),
           "default_camera_swm_render": "front_pixels",
           "nq": int(m.nq), "nv": int(m.nv), "nmocap": int(m.nmocap)}

    # ---- G1a: re-render with no state change --------------------------------
    f0 = u.render().copy()
    f0b = u.render().copy()
    rec["G1a_rerender_identical"] = bool(np.array_equal(f0, f0b))
    rec["frame_shape"] = list(f0.shape)
    rec["frame_dtype"] = str(f0.dtype)

    # ---- G1b: same (qpos,qvel), different history ---------------------------
    qpos_a, qvel_a = d.qpos.copy(), d.qvel.copy()
    mocap_a = (d.mocap_pos.copy(), d.mocap_quat.copy())
    model_a = snap_model(m)
    # drive the sim away, then restore only (qpos, qvel) -- the documented set_state API
    for _ in range(25):
        env.step(env.action_space.sample())
    set_state(u, qpos_a, qvel_a)
    f1 = u.render().copy()
    rec["G1b_same_qpos_qvel_identical"] = bool(np.array_equal(f0, f1))
    rec["G1b_max_abs_diff"] = int(np.abs(f0.astype(np.int16) - f1.astype(np.int16)).max())
    rec["G1b_frac_pixels_differing"] = float((f0 != f1).any(-1).mean())
    # what moved that is not (qpos, qvel)?
    rec["G1b_mocap_pos_changed"] = bool(not np.array_equal(mocap_a[0], d.mocap_pos))
    rec["G1b_mocap_quat_changed"] = bool(not np.array_equal(mocap_a[1], d.mocap_quat))
    model_b = snap_model(m)
    rec["G1b_model_fields_changed"] = sorted(
        f for f in RENDER_FIELDS if not np.array_equal(model_a[f], model_b[f]))

    # restoring mocap too: is the frame then pure in (qpos, qvel, mocap, model)?
    d.mocap_pos[:] = mocap_a[0]; d.mocap_quat[:] = mocap_a[1]
    for f in RENDER_FIELDS:
        getattr(m, f)[:] = model_a[f]
    mujoco.mj_forward(m, d)
    f2 = u.render().copy()
    rec["G1c_with_mocap_and_model_restored_identical"] = bool(np.array_equal(f0, f2))
    rec["G1c_max_abs_diff"] = int(np.abs(f0.astype(np.int16) - f2.astype(np.int16)).max())
    rec["G1c_frac_pixels_differing"] = float((f0 != f2).any(-1).mean())

    # ---- G1d: two resets with the SAME seed ---------------------------------
    # `compile_model_and_data` REBINDS `_model`/`_data` on every dirty reset, so `m`/`d`
    # must be re-fetched after each reset; caching them silently compares a struct with
    # itself.
    env.reset(seed=7); m, d = u.model, u.data; fr1 = u.render().copy()
    q1, v1 = d.qpos.copy(), d.qvel.copy()
    mp1, mq1 = d.mocap_pos.copy(), d.mocap_quat.copy()
    g1 = snap_model(m)
    b1 = list(getattr(u, "_cur_button_states", []))
    env.reset(seed=7); m, d = u.model, u.data; fr2 = u.render().copy()
    rec["G1d_same_seed_reset_identical"] = bool(np.array_equal(fr1, fr2))
    rec["G1d_same_seed_frac_pixels_differing"] = float((fr1 != fr2).any(-1).mean())
    rec["G1d_same_seed_qpos_identical"] = bool(np.array_equal(q1, d.qpos))
    rec["G1d_same_seed_qvel_identical"] = bool(np.array_equal(v1, d.qvel))
    rec["G1d_same_seed_mocap_pos_identical"] = bool(np.array_equal(mp1, d.mocap_pos))
    rec["G1d_same_seed_mocap_quat_identical"] = bool(np.array_equal(mq1, d.mocap_quat))
    rec["G1d_same_seed_mocap_pos_delta"] = float(np.abs(mp1 - d.mocap_pos).max())
    rec["G1d_same_seed_button_states_identical"] = bool(b1 == list(getattr(u, "_cur_button_states", [])))
    rec["G1d_same_seed_model_fields_changed"] = sorted(
        f for f in RENDER_FIELDS if not np.array_equal(g1[f], snap_model(m)[f]))
    # and with mocap forced back: is the residual zero?
    d.mocap_pos[:] = mp1; d.mocap_quat[:] = mq1
    for f in RENDER_FIELDS:
        getattr(m, f)[:] = g1[f]
    mujoco.mj_forward(m, d)
    d.qpos[:] = q1; d.qvel[:] = v1
    mujoco.mj_forward(m, d)
    fr3 = u.render().copy()
    rec["G1d_same_seed_identical_after_forcing_mocap_model_qpos"] = bool(np.array_equal(fr1, fr3))

    # ---- G1f: the public swm reset-from-state path --------------------------
    try:
        st = np.concatenate([q1, v1])
        env.reset(seed=0, options={"state": st})
        m, d = u.model, u.data
        fr4 = u.render().copy()
        rec["G1f_reset_options_state_works"] = True
        rec["G1f_reset_options_state_matches_original_frame"] = bool(np.array_equal(fr1, fr4))
        rec["G1f_reset_options_state_frac_pixels_differing"] = float((fr1 != fr4).any(-1).mean())
    except Exception as e:
        rec["G1f_reset_options_state_works"] = False
        rec["G1f_reset_options_state_error"] = f"{type(e).__name__}: {e}"[:300]
    rec["G1f_state_vector_len_nq_plus_nv"] = int(m.nq + m.nv)

    # ---- G1e: two resets with DIFFERENT seeds -- what appearance bits move? --
    env.reset(seed=11); m, d = u.model, u.data; a = snap_model(m); na = int(m.ngeom), int(m.nmesh), int(m.nq)
    env.reset(seed=12); m, d = u.model, u.data; b = snap_model(m); nb = int(m.ngeom), int(m.nmesh), int(m.nq)
    rec["G1e_ngeom_nmesh_nq_stable_across_resets"] = bool(na == nb)
    rec["G1e_ngeom_nmesh_nq"] = list(na)
    rec["G1e_model_fields_changed_across_seeds"] = sorted(
        f for f in RENDER_FIELDS if not np.array_equal(a[f], b[f]))

    # ---- bytes ---------------------------------------------------------------
    state_bytes = int(m.nq * 8 + m.nv * 8)
    full_bytes = state_bytes + int(m.nmocap * 7 * 8)
    frame_bytes = int(np.prod(f0.shape))
    rec["bytes"] = {"qpos_qvel_f64": state_bytes,
                    "qpos_qvel_mocap_f64": full_bytes,
                    "frame_uint8": frame_bytes,
                    "frame_over_state": frame_bytes / state_bytes}

    # ---- the bars -----------------------------------------------------------
    env.reset(seed=0)
    m, d = u.model, u.data
    for _ in range(10):
        u.render()
    t0 = time.perf_counter()
    for _ in range(reps):
        u.render()
    rec["official_fps_render_only"] = reps / (time.perf_counter() - t0)

    qs = []
    for s in range(8):
        env.reset(seed=100 + s)
        m, d = u.model, u.data
        qs.append((d.qpos.copy(), d.qvel.copy()))
    t0 = time.perf_counter()
    n = 0
    for _ in range(max(1, reps // 8)):
        for q, v in qs:
            set_state(u, q, v); u.render(); n += 1
    rec["official_fps_from_state"] = n / (time.perf_counter() - t0)

    frame = u.render().copy()
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(frame).save(buf, format="JPEG", quality=90)
    blob = buf.getvalue()
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    nn = 300
    t0 = time.perf_counter()
    for _ in range(nn):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    rec["decode_fps_per_core"] = nn / (time.perf_counter() - t0)
    rec["jpeg_q90_bytes"] = len(blob)

    rec["bar_S_cpu_ge_5_abs_fps"] = 5.0 * rec["official_fps_render_only"]
    rec["bar_cores_cpu_ge_1_abs_fps"] = rec["decode_fps_per_core"]
    rec["binding_bar_abs_fps"] = max(rec["bar_S_cpu_ge_5_abs_fps"], rec["bar_cores_cpu_ge_1_abs_fps"])

    # ---- does every reset recompile the MjModel? ----------------------------
    env.reset(seed=1); m0 = u.model
    t0 = time.perf_counter(); env.reset(seed=2); rec["reset_s"] = time.perf_counter() - t0
    rec["model_rebound_every_reset"] = bool(u.model is not m0)
    rec["dirty_after_reset"] = bool(getattr(u, "_dirty", None))

    try:
        from OpenGL import GL
        rec["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        rec["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
    except Exception as e:
        rec["gl_renderer"] = f"unreadable: {type(e).__name__}"

    tag = env_id.split("/")[-1].replace("-v0", "")
    np.save(Path(out_dir) / f"official_{tag}_{h}.npy", f0)
    env.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=120)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for eid in a.envs:
        rec = probe(eid, out, a.reps)
        tag = eid.split("/")[-1].replace("-v0", "")
        (out / f"purity_fps_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{eid} {rec['h']}px: pure(qpos,qvel)={rec['G1b_same_qpos_qvel_identical']} "
              f"pure(+mocap+model)={rec['G1c_with_mocap_and_model_restored_identical']} "
              f"seed-stable={rec['G1d_same_seed_reset_identical']} "
              f"official {rec['official_fps_render_only']:.1f} f/s  decode {rec['decode_fps_per_core']:.0f} f/s  "
              f"S>=5 bar {rec['bar_S_cpu_ge_5_abs_fps']:.0f}  cores>=1 bar {rec['bar_cores_cpu_ge_1_abs_fps']:.0f}", flush=True)


if __name__ == "__main__":
    main()
