"""Phase 0 follow-up: mechanism behind (a) post-step frame != post-teleport frame at the same base pose
and (b) explicit getCameraImage light kwargs vs the default picture. Writes phase0_b.json."""
from __future__ import annotations

import json
import os
import time

import numpy as np

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "phase0")
import gymnasium as gym  # noqa: E402
import pybullet as p  # noqa: E402
from PIL import Image  # noqa: E402
import stable_worldmodel  # noqa: E402,F401

R: dict = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def ndiff(a, b):
    return int(np.any(a != b, axis=-1).sum())


g = gym.make("swm/PFRocketLanding-v0", render_mode="rgb_array")
u = g.unwrapped
g.reset(seed=1, options={"variation": ["all"]})
c = u.env._client
rid = u.env.drones[0].Id
cam = u.env.drones[0].camera


def raw(res=480, seg_links=False, **kw):
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if seg_links else 0
    _, _, rgba, depth, seg = p.getCameraImage(res, res, viewMatrix=cam.view_mat, projectionMatrix=cam.proj_mat,
                                              flags=flags, physicsClientId=c, **kw)
    return np.asarray(rgba, np.uint8).reshape(res, res, -1)[..., :3], np.asarray(seg).reshape(res, res)


def links(fk):
    out = []
    for i in range(10):
        s = p.getLinkState(rid, i, computeForwardKinematics=fk, physicsClientId=c)
        out.append(np.array(s[4] if fk else s[0]))
    return np.array(out)


def pose():
    pos, q = p.getBasePositionAndOrientation(rid, physicsClientId=c)
    return np.array(pos), np.array(q)


def seg_links_count(seg):
    """pixels per (body, link) in a link-indexed segmentation mask"""
    d = {}
    for v in np.unique(seg):
        if v < 0:
            continue
        body, link = int(v & ((1 << 24) - 1)), int(v >> 24) - 1
        d[f"{body}:{link}"] = int((seg == v).sum())
    return d


# ---------------------------------------------------------------- (a) teleport to the SAME pose
log("(a) step x12 then teleport to the same reported pose")
for _ in range(12):
    g.step(np.zeros(7))
pos, q = pose()
R1, S1 = raw(seg_links=True)
L_cached_1, L_fk_1 = links(0), links(1)
base_1 = pos.copy()
p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
pos2, q2 = pose()
R3, S3 = raw(seg_links=True)
L_cached_3, L_fk_3 = links(0), links(1)
A = {"base_pose_roundtrip_max_abs": float(max(np.abs(pos2 - pos).max(), np.abs(q2 - q).max())),
     "post_step_vs_post_teleport_same_pose_diff_px": ndiff(R1, R3),
     "link_cached_pos_change_after_teleport_max_abs": float(np.abs(L_cached_3 - L_cached_1).max()),
     "link_fk_pos_change_after_teleport_max_abs": float(np.abs(L_fk_3 - L_fk_1).max()),
     "post_step_cached_vs_fk_max_abs": float(np.abs(L_cached_1 - L_fk_1).max()),
     "post_teleport_cached_vs_fk_max_abs": float(np.abs(L_cached_3 - L_fk_3).max()),
     "post_step_link0_minus_base": (L_cached_1[0] - base_1).round(5).tolist(),
     "post_teleport_link0_minus_base": (L_cached_3[0] - pos2).round(5).tolist(),
     "seg_px_post_step": seg_links_count(S1), "seg_px_post_teleport": seg_links_count(S3)}
# per-link: which links' pixels moved?
d = np.any(R1 != R3, axis=-1)
A["diff_px_by_link_post_step_mask"] = seg_links_count(np.where(d, S1, -1))
A["diff_px_by_link_post_teleport_mask"] = seg_links_count(np.where(d, S3, -1))
# does a physics step after the teleport (zero velocity, zero gravity effect is impossible -> use tiny step) restore it?
p.resetBaseVelocity(rid, [0, 0, 0], [0, 0, 0], physicsClientId=c)
p.stepSimulation(physicsClientId=c)
pos4, q4 = pose()
p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)  # back to the reported pose
R5, S5 = raw(seg_links=True)
A["teleport_after_a_step_vs_post_step_diff_px"] = ndiff(R1, R5)
Image.fromarray(R1).save(os.path.join(OUT, "preview_b_post_step.png"))
Image.fromarray(R3).save(os.path.join(OUT, "preview_b_post_teleport_same_pose.png"))
vis = np.zeros_like(R1)
vis[d] = 255
Image.fromarray(vis).save(os.path.join(OUT, "preview_b_diff_mask.png"))
R["a_teleport"] = A
log("  ", {k: v for k, v in A.items() if "diff_px" in k or "max_abs" in k})

# ---------------------------------------------------------------- (b) light kwargs: persistence + true default
log("(b) light persistence and default direction")
g.reset(seed=1, options={"variation": ["all"]})
c = u.env._client
rid = u.env.drones[0].Id
cam = u.env.drones[0].camera
D0, _ = raw()
D0b, _ = raw()
B = {"default_twice_diff_px": ndiff(D0, D0b)}
E1, _ = raw(lightDirection=[-5, 200, -40])
D1, _ = raw()
B["explicit_-5_200_-40_vs_default_diff_px"] = ndiff(D0, E1)
B["default_after_explicit_vs_default_before_diff_px"] = ndiff(D0, D1)
B["default_after_explicit_vs_explicit_diff_px"] = ndiff(E1, D1)
# fresh client for a clean sweep: each candidate in its own reset so nothing persists
pos, q = pose()
Rm = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
eye = pos + Rm @ np.array([-1.0, 0.0, 3.0])
view = np.array(cam.view_mat).reshape(4, 4).T
fwd = -view[2, :3]
cands = {"eye_pos": eye.tolist(), "eye_minus_target": (eye - pos).tolist(), "minus_forward": (-fwd).tolist(),
         "forward": fwd.tolist(), "z_up": [0, 0, 1], "z_down": [0, 0, -1], "y": [0, 1, 0], "-5_200_-40": [-5, 200, -40],
         "1_1_1": [1, 1, 1], "x": [1, 0, 0]}
sweep = {}
for name, ld in cands.items():
    g.reset(seed=1, options={"variation": ["all"]})
    c = u.env._client
    rid = u.env.drones[0].Id
    cam = u.env.drones[0].camera
    p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
    d0, _ = raw()
    e, _ = raw(lightDirection=[float(x) for x in ld])
    sweep[name] = ndiff(d0, e)
B["candidate_default_light_direction_diff_px_vs_default"] = sweep
# ambient-only relation: is default == int(c*(0.6 + 0.35*ndl + 0.05*spec))? measure top colours at both
g.reset(seed=1, options={"variation": ["all"]})
c = u.env._client
rid = u.env.drones[0].Id
cam = u.env.drones[0].camera
p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
d0, s0 = raw(seg_links=True)
amb, _ = raw(lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
body = (s0 & ((1 << 24) - 1)) == rid
body = body & ((s0 >> 24) - 1 == -1)
cb = np.array(p.getVisualShapeData(rid, physicsClientId=c)[0][7][:3])
B["body_rgba"] = cb.tolist()
B["ambient_only_body_unique"] = np.unique(amb[body].reshape(-1, 3), axis=0).tolist()[:6]
B["int_255c"] = [int(255 * x) for x in cb]
vals, cnt = np.unique(d0[body].reshape(-1, 3), axis=0, return_counts=True)
B["default_body_max_rgb"] = vals.max(0).tolist()
B["default_body_min_rgb"] = vals.min(0).tolist()
B["default_body_mode_rgb"] = vals[np.argmax(cnt)].tolist()
B["int_255c_x0.6"] = [int(int(255 * x) * 0.6) for x in cb]
B["int_255c_x1.0"] = [int(int(255 * x) * 1.0) for x in cb]
B["int_255c_x0.65"] = [int(int(255 * x) * 0.65) for x in cb]
R["b_light"] = B
log("  ", B)

# ---------------------------------------------------------------- (c) hand-recolour with EXACT rgba
log("(c) exact-rgba hand recolour of a default reset")
g.reset(seed=1, options={"variation": ["all"]})
c = u.env._client
rid = u.env.drones[0].Id
pad = u.landing_pad_id
cam = u.env.drones[0].camera
p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
ref, _ = raw()
cols = [v[7] for v in p.getVisualShapeData(rid, physicsClientId=c)]
padcol = p.getVisualShapeData(pad, physicsClientId=c)[0][7]
g.reset(seed=9, options={})
c = u.env._client
rid = u.env.drones[0].Id
pad = u.landing_pad_id
cam = u.env.drones[0].camera
for i, rgba in enumerate(cols):
    p.changeVisualShape(rid, i - 1, rgbaColor=list(rgba), physicsClientId=c)
p.changeVisualShape(pad, -1, rgbaColor=list(padcol), physicsClientId=c)
p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
hand, _ = raw()
R["c_hand_recolour_exact_diff_px"] = ndiff(ref, hand)
# and low, where the pad and ground are in frame
low_pos = np.array([0.4, -0.3, 6.0])
p.resetBasePositionAndOrientation(rid, list(low_pos), list(q), physicsClientId=c)
hand_low, _ = raw()
g.reset(seed=1, options={"variation": ["all"]})
c = u.env._client
rid = u.env.drones[0].Id
cam = u.env.drones[0].camera
p.resetBasePositionAndOrientation(rid, list(low_pos), list(q), physicsClientId=c)
ref_low, _ = raw()
R["c_hand_recolour_exact_low_altitude_diff_px"] = ndiff(ref_low, hand_low)
log("  ", R["c_hand_recolour_exact_diff_px"], R["c_hand_recolour_exact_low_altitude_diff_px"])
g.close()
with open(os.path.join(OUT, "phase0_b.json"), "w") as fh:
    json.dump(R, fh, indent=1)
log("done")
