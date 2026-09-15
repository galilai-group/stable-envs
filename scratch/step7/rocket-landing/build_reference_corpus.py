"""M1: the reference corpus for PFRocketLanding — STEPPED rollouts, states as the env observes them.

Convention (coordinator ruling 3): the official picture is the one swm observes after ``step``, whose
links and camera are one 1/240 s substep behind the base. So every state here was recorded right
after an ``env.step`` and its frame is the frame that step produced; the renderer reproduces the
lagged pose from (pose, v, w). The state is ``obs[:13]`` = ang_vel (body frame), quaternion (xyzw),
lin_vel (body frame), lin_pos -- exactly the leading 13 entries of the env's own observation.

Validations recorded per state, in meta.json:
  * the cached link-0 pose (what TinyRenderer draws / where the camera sits) equals
    (pos - v dt, exp(-w dt) q) with WORLD velocities recovered from the obs;
  * the obs quaternion equals PyBullet's (it round-trips through Euler in PyFlyt);
  * the per-link rgba PyBullet holds equals the style -> colour law the renderer will use;
  * the "reference trick" -- teleport to the previous substep, one raw ``stepSimulation`` -- reproduces
    the recorded frame (so ``reference_render`` can be driven at ARBITRARY states later).

Usage: /Users/samib/se7-rocket-venv/bin/python scratch/step7/rocket-landing/build_reference_corpus.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version

import numpy as np
from scipy.spatial.transform import Rotation as Rt

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
os.makedirs(OUT, exist_ok=True)
DT = 1.0 / 240.0
G = np.array([0.0, 0.0, -9.81])
IGNITION_RGBA = np.array([1.0, 0.3, 0.3, 0.8])
FLAME_RGBA = np.array([1.0, 0.8, 0.0, 0.1])
N_EPISODES = int(os.environ.get("N_EPISODES", "24"))
UNIFORM_PER_EP = 6
HEIGHT_TRIGGERS = [200.0, 100.0, 50.0, 25.0, 12.0, 6.0, 3.0]
SEED0 = 1000

import gymnasium as gym  # noqa: E402
import pybullet as p  # noqa: E402
import stable_worldmodel  # noqa: E402,F401


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def style_to_rgba(style_rgb_uint8: np.ndarray, flame_bit: int):
    """The 13-geom colour table the renderer draws with, from the 5 RGBs + flame bit.

    Order: ground, body, fueltank, booster, fin_pos_x, fin_neg_x, fin_pos_y, fin_neg_y, leg1, leg2, leg3,
    flame, pad. Rocket links take ``value / 255`` as floats (``changeVisualShape``); the pad takes the
    3-decimal rounding the URDF rewrite applies (``f"{v:.3f}"``); the fuel tank is PyBullet's white
    default for a link without a <visual>; the flame is its URDF colour until the first ignition
    transition and the ignition colour ever after.
    """
    body, fin, leg, booster, pad = [np.asarray(style_rgb_uint8[i * 3:(i + 1) * 3], np.float64) / 255.0
                                    for i in range(5)]
    pad_q = np.array([float(f"{x:.3f}") for x in pad])
    flame = IGNITION_RGBA if flame_bit else FLAME_RGBA
    one = lambda rgb: np.array([*rgb, 1.0])   # noqa: E731
    return np.stack([one([1, 1, 1]), one(body), one([1, 1, 1]), one(booster), one(fin), one(fin), one(fin),
                     one(fin), one(leg), one(leg), one(leg), flame, one(pad_q)])


def main():
    env = gym.make("swm/PFRocketLanding-v0", render_mode="rgb_array")
    u = env.unwrapped
    rng = np.random.default_rng(7)
    rec = {k: [] for k in ["state13", "obs30", "style", "pose", "world_vel", "cached_link0", "episode", "step",
                           "height", "frame", "sha", "px_by_body"]}
    val = {"lag_pos_err": [], "lag_q_err": [], "obs_q_err": [], "obs_vel_err": [], "rgba_err": [],
           "trick_diff_px": [], "trick_pos_err": [], "trick_q_err": []}
    pol = {"policy": "per-episode: ignition sticky Bernoulli(0.5) held 10 steps, throttle U(0.3,1) when "
                     "ignited, fins/gimbal U(-0.3,0.3); samples: 6 uniform step indices in [1,650] plus one "
                     "trigger per height threshold", "height_triggers": HEIGHT_TRIGGERS,
           "variation": ["all"], "seeds": []}
    for ep in range(N_EPISODES):
        seed = SEED0 + ep
        pol["seeds"].append(seed)
        obs, info = env.reset(seed=seed, options={"variation": ["all"]})
        c, rid, pad, plane = u.env._client, u.env.drones[0].Id, u.landing_pad_id, u.env.planeId
        vs = u.variation_space
        style_rgb = np.concatenate([np.asarray(vs["rocket"]["body_color"].value, np.uint8),
                                    np.asarray(vs["rocket"]["fin_color"].value, np.uint8),
                                    np.asarray(vs["rocket"]["leg_color"].value, np.uint8),
                                    np.asarray(vs["rocket"]["booster_color"].value, np.uint8),
                                    np.asarray(vs["pad"]["color"].value, np.uint8)])
        want_steps = set(int(x) for x in rng.integers(1, 650, UNIFORM_PER_EP))
        triggers = list(HEIGHT_TRIGGERS)
        ignite, hold = 0.0, 0
        n_ep = 0
        for step in range(1, 1201):
            if hold <= 0:
                ignite, hold = float(rng.random() < 0.5), 10
            hold -= 1
            a = rng.uniform(-0.3, 0.3, 7)
            a[3] = ignite
            a[4] = rng.uniform(0.3, 1.0) if ignite else 0.0
            obs, r, term, trunc, info = env.step(a)
            h = float(obs[12])
            take = step in want_steps
            if triggers and h < triggers[0]:
                take = True
                while triggers and h < triggers[0]:
                    triggers.pop(0)
            if take and not (term or trunc):
                frame = u.render()
                pos, q = map(np.array, p.getBasePositionAndOrientation(rid, physicsClientId=c))
                lv, av = map(np.array, p.getBaseVelocity(rid, physicsClientId=c))
                ls = p.getLinkState(rid, 0, computeForwardKinematics=0, physicsClientId=c)
                cpos, cq = np.array(ls[0]), np.array(ls[1])
                shapes = p.getVisualShapeData(rid, physicsClientId=c)
                flame_bit = int(np.allclose(np.array(shapes[10][7]), IGNITION_RGBA))
                # --- validations
                Rq = Rt.from_quat(obs[3:7])
                v_world = Rq.apply(obs[7:10])
                w_world = Rq.apply(obs[0:3])
                pred_pos = pos - lv * DT
                pred_q = (Rt.from_rotvec(-av * DT) * Rt.from_quat(q)).as_quat()
                qerr = lambda a_, b_: float(min(np.abs(a_ - b_).max(), np.abs(a_ + b_).max()))  # noqa: E731
                val["lag_pos_err"].append(float(np.abs(cpos - pred_pos).max()))
                val["lag_q_err"].append(qerr(cq, pred_q))
                val["obs_q_err"].append(qerr(np.array(obs[3:7]), q))
                val["obs_vel_err"].append(float(max(np.abs(v_world - lv).max(), np.abs(w_world - av).max())))
                held = np.array([s[7] for s in shapes] + [p.getVisualShapeData(pad, physicsClientId=c)[0][7]])
                pred_rgba = style_to_rgba(style_rgb, flame_bit)[1:]
                val["rgba_err"].append(float(np.abs(held - pred_rgba).max()))
                _, _, _, _, seg = p.getCameraImage(480, 480, viewMatrix=u.env.drones[0].camera.view_mat,
                                                   projectionMatrix=u.env.drones[0].camera.proj_mat,
                                                   physicsClientId=c)
                seg = np.asarray(seg).reshape(480, 480)
                pxb = {"rocket": int((seg == rid).sum()), "pad": int((seg == pad).sum()),
                       "ground": int((seg == plane).sum()), "bg": int((seg < 0).sum())}
                # --- the reference trick, on a saved copy of the dynamics
                sid = p.saveState(physicsClientId=c)
                p.resetBasePositionAndOrientation(rid, list(pred_pos), list(pred_q), physicsClientId=c)
                p.resetBaseVelocity(rid, list(lv - G * DT), list(av), physicsClientId=c)
                p.stepSimulation(physicsClientId=c)
                tpos, tq = map(np.array, p.getBasePositionAndOrientation(rid, physicsClientId=c))
                tframe = u.render()
                val["trick_pos_err"].append(float(np.abs(tpos - pos).max()))
                val["trick_q_err"].append(qerr(tq, q))
                val["trick_diff_px"].append(int(np.any(tframe != frame, axis=-1).sum()))
                p.restoreState(stateId=sid, physicsClientId=c)
                p.removeState(sid, physicsClientId=c)
                # --- record
                rec["state13"].append(np.asarray(obs[:13], np.float32))
                rec["obs30"].append(np.asarray(obs, np.float64))
                rec["style"].append(np.concatenate([style_rgb.astype(np.float32), [float(flame_bit)]]))
                rec["pose"].append(np.concatenate([pos, q]))
                rec["world_vel"].append(np.concatenate([lv, av]))
                rec["cached_link0"].append(np.concatenate([cpos, cq]))
                rec["episode"].append(ep)
                rec["step"].append(step)
                rec["height"].append(h)
                rec["frame"].append(frame)
                rec["sha"].append(hashlib.sha256(frame.tobytes()).hexdigest())
                rec["px_by_body"].append(pxb)
                n_ep += 1
            if term or trunc:
                break
        log(f"episode {ep} seed {seed}: {step} steps, {n_ep} states, final h={h:.1f}, "
            f"term={term} trunc={trunc} info={ {k: v for k, v in info.items() if k != 'goal'} }")
    N = len(rec["state13"])
    frames = np.stack(rec["frame"]).astype(np.uint8)
    np.savez_compressed(os.path.join(OUT, "states.npz"),
                        state13=np.stack(rec["state13"]), obs30=np.stack(rec["obs30"]),
                        style=np.stack(rec["style"]), pose=np.stack(rec["pose"]),
                        world_vel=np.stack(rec["world_vel"]), cached_link0=np.stack(rec["cached_link0"]),
                        episode=np.array(rec["episode"]), step=np.array(rec["step"]),
                        height=np.array(rec["height"], np.float32),
                        frame_sha256=np.array(rec["sha"]))
    np.savez_compressed(os.path.join(OUT, "frames.npz"), frames=frames)
    meta = {
        "n_states": N, "n_episodes": N_EPISODES, "frame_shape": list(frames.shape[1:]),
        "state_layout": "obs[:13] = ang_vel_body(3), quat_xyzw(4), lin_vel_body(3), lin_pos(3)",
        "style_layout": "body_rgb, fin_rgb, leg_rgb, booster_rgb, pad_rgb (uint8 0-255 as float), flame_ignited",
        "convention": "frames recorded right after env.step (lagged links/camera); see build script docstring",
        "policy": pol,
        "validation": {k: {"max": float(np.max(v)), "mean": float(np.mean(v)), "n": len(v)} for k, v in val.items()},
        "trick_diff_px_hist": {str(k): int(n) for k, n in zip(*np.unique(val["trick_diff_px"], return_counts=True))},
        "height_hist": np.histogram(rec["height"], bins=[0, 3, 6, 12, 25, 50, 100, 200, 300, 500])[0].tolist(),
        "height_bins": [0, 3, 6, 12, 25, 50, 100, 200, 300, 500],
        "flame_ignited_frac": float(np.mean([s[-1] for s in rec["style"]])),
        "px_by_body_mean": {k: float(np.mean([d[k] for d in rec["px_by_body"]])) for k in ["rocket", "pad", "ground", "bg"]},
        "versions": {k: pkg_version(k) for k in ["PyFlyt", "pybullet", "stable-worldmodel", "gymnasium", "numpy", "numba"]},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "renderer": "PyBullet DIRECT / TinyRenderer, 480x480, env.render() -> [..., :3]",
        "files": {"states.npz": os.path.getsize(os.path.join(OUT, "states.npz")),
                  "frames.npz": os.path.getsize(os.path.join(OUT, "frames.npz"))},
        "build_s": round(time.time() - T0, 1),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    log("done", json.dumps({k: meta[k] for k in ["n_states", "validation", "trick_diff_px_hist", "height_hist",
                                                 "flame_ignited_frac", "files"]}, indent=1))
    env.close()


if __name__ == "__main__":
    main()
