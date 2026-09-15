"""Check ``stable_envs.envs.rocket_landing``'s pure math against PyBullet itself (no PyFlyt, no JIT).

Euler conventions against ``getEulerFromQuaternion`` / ``getQuaternionFromEuler``; the camera
against PyFlyt's ``Camera.view_mat`` recipe evaluated with PyBullet's own ``computeViewMatrix``; the
link frames against ``getLinkState`` on the loaded rocket URDF; the hull cylinder's silhouette against
TinyRenderer's. Writes examples/results/step7/rocket-landing/phase0/env_math_validation.json.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pybullet as p
import PyFlyt
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from stable_envs.envs import rocket_landing as RL  # noqa: E402
from stable_envs.render import nvdiff as ND  # noqa: E402

rng = np.random.default_rng(3)
R: dict = {}

# ---------------------------------------------------------------- euler
qs = rng.normal(size=(500, 4))
qs /= np.linalg.norm(qs, axis=1, keepdims=True)
e_err = max(np.abs(np.array(p.getEulerFromQuaternion(list(q))) - RL.euler_from_quat(q)).max() for q in qs)
rpys = rng.uniform(-3, 3, size=(500, 3))
q_err = 0.0
for r in rpys:
    a, b = np.array(p.getQuaternionFromEuler(list(r))), RL.quat_from_euler(r)
    q_err = max(q_err, float(min(np.abs(a - b).max(), np.abs(a + b).max())))
m_err = max(np.abs(np.array(p.getMatrixFromQuaternion(list(q))).reshape(3, 3) - RL.quat_to_mat(q)).max() for q in qs)
R["euler_from_quat_max_err"] = float(e_err)
R["quat_from_euler_max_err"] = float(q_err)
R["quat_to_mat_max_err"] = float(m_err)

# ---------------------------------------------------------------- camera: PyFlyt's recipe with PyBullet's functions
cam_err = 0.0
for q in qs[:200]:
    pos = rng.uniform(-5, 5, 3) + np.array([0, 0, 50.0])
    Rm = np.array(p.getMatrixFromQuaternion(list(q))).reshape(3, 3)
    position = pos + Rm @ RL.CAM_OFFSET
    rot = np.array(p.getEulerFromQuaternion(list(q)))
    rot[1] += RL.CAM_PITCH_DEG / 180 * math.pi
    Rc = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(list(rot)))).reshape(3, 3)
    up = Rc @ np.array([0.0, 0.0, 1.0])
    V = np.array(p.computeViewMatrix(list(position), list(pos), list(up))).reshape(4, 4).T
    eye, mat = RL.camera_pose(pos[None], q[None])
    view, _ = ND.Camera(eye, mat, RL.CAM_FOV, RL.CAM_NEAR, RL.CAM_FAR).matrices()
    cam_err = max(cam_err, float(np.abs(view[0] - V).max()))
    P = np.array(p.computeProjectionMatrixFOV(RL.CAM_FOV, 1.0, RL.CAM_NEAR, RL.CAM_FAR)).reshape(4, 4).T
    _, proj = ND.Camera(eye, mat, RL.CAM_FOV, RL.CAM_NEAR, RL.CAM_FAR).matrices()
    cam_err = max(cam_err, float(np.abs(proj - P).max()))
R["camera_view_and_proj_max_err"] = cam_err

# ---------------------------------------------------------------- link frames from the loaded URDF
c = p.connect(p.DIRECT)
rp = os.path.join(os.path.dirname(PyFlyt.__file__), "models/vehicles/rocket/rocket.urdf")
rid = p.loadURDF(rp, [0, 0, 10], useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c)
lerr = 0.0
for q in qs[:50]:
    pos = np.array([1.0, -2.0, 30.0]) + rng.uniform(-1, 1, 3)
    p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
    # zero velocity => t and t - dt coincide, so geom_poses gives the synced link frames
    state = np.concatenate([[0, 0, 0], q, [0, 0, 0], pos])
    xp, xm = RL.geom_poses(state[None])
    for i in range(10):
        ls = p.getLinkState(rid, i, computeForwardKinematics=1, physicsClientId=c)
        lp, lq = np.array(ls[4]), np.array(ls[5])
        g = 2 + i                                  # link i is geom 2+i (geom 1 is the base)
        lerr = max(lerr, float(np.abs(xp[0, g] - lp).max()),
                   float(np.abs(xm[0, g] - np.array(p.getMatrixFromQuaternion(list(lq))).reshape(3, 3)).max()))
    lerr = max(lerr, float(np.abs(xp[0, 1] - pos).max()))
R["link_frames_max_err"] = lerr
vis = p.getVisualShapeData(rid, physicsClientId=c)
R["visual_local_frames_identity"] = bool(all(np.allclose(v[5], 0) and np.allclose(v[6], [0, 0, 0, 1]) for v in vis))
p.disconnect(c)

# ---------------------------------------------------------------- hull silhouette vs TinyRenderer, top-down
c = p.connect(p.DIRECT)
vs = p.createVisualShape(p.GEOM_CYLINDER, radius=1.0, length=2.0, rgbaColor=[1, 1, 1, 1], physicsClientId=c)
p.createMultiBody(0, baseVisualShapeIndex=vs, basePosition=[0, 0, 0], physicsClientId=c)
V = p.computeViewMatrix([0, 0, 8], [0, 0, 0], [0, 1, 0])
P = p.computeProjectionMatrixFOV(20, 1, 0.1, 100)
res = 512
_, _, rgba, _, sg = p.getCameraImage(res, res, viewMatrix=V, projectionMatrix=P, physicsClientId=c)
tiny = np.asarray(sg).reshape(res, res) >= 0
p.disconnect(c)
vert, norm, face = RL.tiny_cylinder(2.0, 1.0)
Vm = np.array(V).reshape(4, 4).T
Pm = np.array(P).reshape(4, 4).T
clip = (Pm @ Vm @ np.concatenate([vert, np.ones((len(vert), 1))], 1).T).T
ndc = clip[:, :2] / clip[:, 3:4]
px = np.stack([(ndc[:, 0] + 1) * res / 2, (1 - ndc[:, 1]) * res / 2], 1)
img = Image.new("L", (res, res), 0)
d = ImageDraw.Draw(img)
for tri in face:
    d.polygon([tuple(px[i]) for i in tri], fill=255)
ours = np.asarray(img) > 0
inter, union = (tiny & ours).sum(), (tiny | ours).sum()
R["hull_silhouette_iou_vs_tinyrenderer"] = float(inter / union)
R["hull_silhouette_px"] = {"tiny": int(tiny.sum()), "ours": int(ours.sum()), "xor": int((tiny ^ ours).sum())}
R["hull_vertices"] = int(len(vert))
R["hull_triangles"] = int(len(face))

# ---------------------------------------------------------------- material law on the corpus meta (already validated) -> formula spot checks
cs = RL.corpus_style_to_env
R["material_examples"] = {
    "pad_200": float(RL.material_table(cs(np.array([0] * 12 + [200, 200, 200, 0])))[0, 12, 0]),
    "body_204": float(RL.material_table(cs(np.array([255, 204, 0] + [0] * 13)))[0, 1, 1]),
    "flame_bit": RL.material_table(cs(np.array([0] * 15 + [1])))[0, 11].tolist(),
    "style_layout": RL.style_layout(),
}
out = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "phase0", "env_math_validation.json")
with open(out, "w") as fh:
    json.dump(R, fh, indent=1)
print(json.dumps(R, indent=1))
