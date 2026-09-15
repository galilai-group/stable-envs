"""Can the official (lagged) frame be produced for an ARBITRARY state? The zero-gravity substep trick.

Recipe: apply the style's colours, teleport the base to the PREVIOUS substep pose
(pos - v dt, exp(-w dt) q), set the recorded world velocities, run one raw ``stepSimulation`` with
gravity OFF (no PyFlyt forces, no gravity, damping already zero) so the base integrates to exactly
(pos, q) while the links and the camera link keep the cached previous pose, then render. Compared
pixel-for-pixel against the corpus frames, which were recorded from stepped rollouts.

Usage: /Users/samib/se7-rocket-venv/bin/python scratch/step7/rocket-landing/validate_reference_trick.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
REF = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
DT = 1.0 / 240.0
IGNITION_RGBA = [1.0, 0.3, 0.3, 0.8]
FLAME_RGBA = [1.0, 0.8, 0.0, 0.1]
LINK_OF = {"body": -1, "fueltank": 0, "booster": 1, "fin": [2, 3, 4, 5], "leg": [6, 7, 8], "flame": 9}

import gymnasium as gym  # noqa: E402
import pybullet as p  # noqa: E402
import stable_worldmodel  # noqa: E402,F401


def apply_style(c, rid, pad, style):
    body, fin, leg, booster, padc = [np.asarray(style[i * 3:(i + 1) * 3], np.float64) / 255.0 for i in range(5)]
    flame = IGNITION_RGBA if style[15] > 0.5 else FLAME_RGBA
    p.changeVisualShape(rid, -1, rgbaColor=[*body, 1.0], physicsClientId=c)
    p.changeVisualShape(rid, 1, rgbaColor=[*booster, 1.0], physicsClientId=c)
    for i in (2, 3, 4, 5):
        p.changeVisualShape(rid, i, rgbaColor=[*fin, 1.0], physicsClientId=c)
    for i in (6, 7, 8):
        p.changeVisualShape(rid, i, rgbaColor=[*leg, 1.0], physicsClientId=c)
    p.changeVisualShape(rid, 9, rgbaColor=flame, physicsClientId=c)
    p.changeVisualShape(pad, -1, rgbaColor=[*[float(f"{x:.3f}") for x in padc], 1.0], physicsClientId=c)


def lagged_render(u, state13, style, zero_gravity=True):
    c, rid, pad = u.env._client, u.env.drones[0].Id, u.landing_pad_id
    apply_style(c, rid, pad, style)
    q = np.asarray(state13[3:7], np.float64)
    R = Rt.from_quat(q)
    w = R.apply(np.asarray(state13[0:3], np.float64))
    v = R.apply(np.asarray(state13[7:10], np.float64))
    pos = np.asarray(state13[10:13], np.float64)
    pos_prev = pos - v * DT
    q_prev = (Rt.from_rotvec(-w * DT) * R).as_quat()
    if zero_gravity:
        p.setGravity(0, 0, 0, physicsClientId=c)
    # Bullet's raw substep is semi-implicit -- pos' = pos_prev + v1 dt, q' = exp(w1 dt) q_prev EXACTLY --
    # but v1 != v_set for a spinning composite multibody (gyroscopic and offset-mass terms). So iterate
    # on the SET velocity until the POST-step velocity equals the recorded one; the base then lands on
    # (pos, q) by construction while the links and the camera link keep the cached previous pose.
    v_set, w_set = v.copy(), w.copy()
    for _ in range(4):
        p.resetBasePositionAndOrientation(rid, list(pos_prev), list(q_prev), physicsClientId=c)
        p.resetBaseVelocity(rid, list(v_set), list(w_set), physicsClientId=c)
        p.stepSimulation(physicsClientId=c)
        v1, w1 = map(np.array, p.getBaseVelocity(rid, physicsClientId=c))
        if np.abs(v1 - v).max() < 1e-13 and np.abs(w1 - w).max() < 1e-13:
            break
        v_set, w_set = v_set + (v - v1), w_set + (w - w1)
    if zero_gravity:
        p.setGravity(0, 0, -9.81, physicsClientId=c)
    got_pos, got_q = map(np.array, p.getBasePositionAndOrientation(rid, physicsClientId=c))
    frame = u.render()
    return frame, float(np.abs(got_pos - pos).max()), float(min(np.abs(got_q - q).max(), np.abs(got_q + q).max()))


def main():
    t0 = time.time()
    S = np.load(os.path.join(REF, "states.npz"))
    F = np.load(os.path.join(REF, "frames.npz"))["frames"]
    st, sty = S["obs30"][:, :13].astype(np.float64), S["style"]
    env = gym.make("swm/PFRocketLanding-v0", render_mode="rgb_array")
    u = env.unwrapped
    env.reset(seed=0, options={"variation": ["all"]})
    rows = []
    idx = list(range(len(st)))
    # the MODULE's recipe (float64 renormalisation, float32-aware pose check), not this file's older copy
    from stable_envs.envs import rocket_landing as RL
    sty_env = RL.corpus_style_to_env(S["style"])
    for i in idx:
        frame = RL._reference_frame(env, st[i], sty_env[i])
        pos_, q_, *_ = (a_[0] for a_ in RL.lagged_pose(st[i][None]))
        gp, gq = (np.array(a_) for a_ in p.getBasePositionAndOrientation(u.env.drones[0].Id, physicsClientId=u.env._client))
        pe, qe = float(np.abs(gp - pos_).max()), float(1.0 - abs(np.dot(gq / np.linalg.norm(gq), q_)))
        d = int(np.any(frame != F[i], axis=-1).sum())
        rows.append({"i": i, "h": float(st[i][12]), "diff_px": d, "pos_err": pe, "q_err": qe})
    diffs = np.array([r["diff_px"] for r in rows])
    out = {"n": len(rows), "exact": int((diffs == 0).sum()), "max_diff_px": int(diffs.max()),
           "mean_diff_px": float(diffs.mean()), "pos_err_max": max(r["pos_err"] for r in rows),
           "q_err_max": max(r["q_err"] for r in rows), "recipe": "stable_envs.envs.rocket_landing._reference_frame (renormalised, float32-aware)",
           "nonzero": [r for r in rows if r["diff_px"]][:20], "s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    with open(os.path.join(REF, "reference_trick_validation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    env.close()


if __name__ == "__main__":
    main()
