"""TinyRenderer laws, second pass: lighting FRAME by construction, specular exponent, nested cylinders,
row-offset sign. Writes phase0/tiny_law2.json."""
from __future__ import annotations

import json
import os

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image
from scipy.spatial.transform import Rotation as Rt

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "results",
                                   "step7", "rocket-landing", "phase0"))
L = np.array([-50.0, 30.0, 100.0])
L /= np.linalg.norm(L)
R: dict = {}


def client():
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    return c


def view(eye, target=(0, 0, 0), up=(0, 0, 1)):
    return p.computeViewMatrix(list(map(float, eye)), list(map(float, target)), list(map(float, up)))


def proj(fov=60.0, near=0.1, far=100.0):
    return p.computeProjectionMatrixFOV(fov, 1.0, near, far)


def shot(c, V, P, res=256, seg=True, **kw):
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if seg else 0
    _, _, rgba, depth, sg = p.getCameraImage(res, res, viewMatrix=V, projectionMatrix=P, flags=flags,
                                             physicsClientId=c, **kw)
    return (np.asarray(rgba, np.uint8).reshape(res, res, 4), np.asarray(depth, np.float32).reshape(res, res),
            np.asarray(sg).reshape(res, res))


def rot_of(V):
    return np.array(V, np.float64).reshape(4, 4).T[:3, :3]


def quat_z_to(n):
    """quaternion (xyzw) rotating +z onto unit n."""
    n = n / np.linalg.norm(n)
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    if s < 1e-9:
        return [0, 0, 0, 1] if n[2] > 0 else [1, 0, 0, 0]
    ax = v / s
    ang = np.arctan2(s, z @ n)
    return Rt.from_rotvec(ax * ang).as_quat().tolist()


def mode_of(img, mask):
    vals = img[mask][:, 0].astype(int)
    return int(np.bincount(vals).argmax()) if len(vals) else None


def quad_body(c, n, pos, rgba=(1, 1, 1, 1), half=0.4):
    vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, 0.02], rgbaColor=list(rgba), physicsClientId=c)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=list(pos),
                             baseOrientation=quat_z_to(np.asarray(n, float)), physicsClientId=c)


# ------------------------------------------------------------ 1. frame: a quad facing l in WORLD vs in VIEW
F = {}
for name, eye in {"eye_+x+y+z": (4.0, 3.0, 2.5), "eye_-x-y+z": (-4.0, -3.0, 2.5), "eye_+y-z": (0.5, 5.0, -1.5)}.items():
    c = client()
    V, P = view(eye), proj(45.0)
    Rv = rot_of(V)                                    # world -> view
    n_world = L.copy()                                # faces the light if the light is world-fixed
    n_viewhyp = Rv.T @ L                              # faces the light if the light is view-fixed
    # place the two quads side by side, both visible, offset perpendicular to the view axis
    right = Rv[0]
    b1 = quad_body(c, n_world, -0.6 * right)
    b2 = quad_body(c, n_viewhyp, +0.6 * right)
    img, _, sg = shot(c, V, P, 256, lightAmbientCoeff=0.0, lightDiffuseCoeff=1.0, lightSpecularCoeff=0.0)
    body = sg & ((1 << 24) - 1)
    F[name] = {"quad_normal_along_l_world_px": mode_of(img, body == b1),
               "quad_normal_along_l_view_px": mode_of(img, body == b2),
               "n_world_dot_l": round(float(n_world @ L), 4),
               "n_viewhyp_in_world_dot_l_world": round(float(n_viewhyp @ L), 4),
               "pred_if_world": [255, int(255 * max(0.0, n_viewhyp @ L))],
               "pred_if_view": [int(255 * max(0.0, (Rv @ n_world) @ L)), 255]}
    Image.fromarray(img[..., :3]).save(os.path.join(OUT, f"tiny_law2_frame_{name}.png"))
    p.disconnect(c)
R["frame"] = F

# ------------------------------------------------------------ 2. specular exponent: tilt sweep, A=D=0, S=1
S = {"tilts": []}
c = client()
V, P = view((0, 0, 6), (0, 0, 0), (0, 1, 0)), proj(30.0)
Rv = rot_of(V)
for deg in [0, 5, 10, 15, 20, 25, 30, 40, 50]:
    for axis_name, axis in (("about_x", np.array([1.0, 0, 0])), ("about_y", np.array([0, 1.0, 0]))):
        n = Rt.from_rotvec(axis * np.radians(deg)).apply([0, 0, 1.0])
        b = quad_body(c, n, (0, 0, 0), half=0.5)
        img, _, sg = shot(c, V, P, 128, lightAmbientCoeff=0.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=1.0)
        dif, _, _ = shot(c, V, P, 128, lightAmbientCoeff=0.0, lightDiffuseCoeff=1.0, lightSpecularCoeff=0.0)
        m = (sg & ((1 << 24) - 1)) == b
        r_w = 2 * (n @ L) * n - L                                   # world-frame reflection
        n_v = Rv @ n
        r_v = 2 * (n_v @ L) * n_v - L                               # view-frame reflection (same numbers here up to R)
        S["tilts"].append({"deg": deg, "axis": axis_name, "spec_px": mode_of(img, m), "diffuse_px": mode_of(dif, m),
                           "ndl_world": round(float(n @ L), 4), "r_z_world": round(float(r_w[2]), 4),
                           "r_z_view": round(float(r_v[2]), 4), "ndl_view": round(float(n_v @ L), 4)})
        p.removeBody(b, physicsClientId=c)
p.disconnect(c)
R["specular"] = S

# ------------------------------------------------------------ 3. nested cylinders, and the rocket URDF by link
c = client()
outer = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=4.0, rgbaColor=[1, 0.8, 0, 1], physicsClientId=c)
inner = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.0, rgbaColor=[1, 1, 1, 1], physicsClientId=c)
bo = p.createMultiBody(baseMass=0, baseVisualShapeIndex=outer, basePosition=[0, 0, 0], physicsClientId=c)
bi = p.createMultiBody(baseMass=0, baseVisualShapeIndex=inner, basePosition=[0, 0, 0], physicsClientId=c)
V, P = view((-1.0, 0.0, 3.0), (0, 0, 0), (0, 0, 1)), proj(90.0)     # the rocket camera geometry
img, dep, sg = shot(c, V, P, 480)
body = sg & ((1 << 24) - 1)
N3 = {"outer_px": int((body == bo).sum()), "inner_px_visible": int((body == bi).sum())}
Image.fromarray(img[..., :3]).save(os.path.join(OUT, "tiny_law2_nested.png"))
p.disconnect(c)
# the rocket urdf itself, upright at the origin, seen from its own camera geometry, segmented by link
import PyFlyt  # noqa: E402
c = client()
rp = os.path.join(os.path.dirname(PyFlyt.__file__), "models/vehicles/rocket/rocket.urdf")
rid = p.loadURDF(rp, [0, 0, 10.0], useFixedBase=False, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c)
V, P = view((-1.0, 0.0, 13.0), (0, 0, 10.0), (0, 0, 1)), proj(90.0)
img, dep, sg = shot(c, V, P, 480)
per_link = {}
for v in np.unique(sg):
    if v < 0:
        continue
    per_link[str(int(v >> 24) - 1)] = int((sg == v).sum())
N3["rocket_px_by_link"] = per_link
N3["rocket_visual_shapes"] = [(int(x[1]), int(x[2]), [round(float(y), 3) for y in x[3]])
                              for x in p.getVisualShapeData(rid, physicsClientId=c)]
Image.fromarray(img[..., :3]).save(os.path.join(OUT, "tiny_law2_rocket_links.png"))
p.disconnect(c)
R["nested"] = N3

# ------------------------------------------------------------ 4. row-offset sign of the corner sampling rule
c = client()
box = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.1], rgbaColor=[1, 0, 0, 1], physicsClientId=c)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0.37, 0.61, 0], physicsClientId=c)
res = 128
rows_ok = {"corner_top": 0, "corner_bottom": 0, "centre": 0, "n": 0, "detail": []}
for d in np.linspace(3.0, 3.6, 13):
    V, P = view((0, 0, d), (0, 0, 0), (0, 1, 0)), proj(60.0)
    img, _, sg = shot(c, V, P, res)
    f = 1.0 / np.tan(np.radians(30.0))
    z = d - 0.1
    # image x: +x right; image y: +y UP in NDC, row 0 at the TOP of the returned array
    x_lo, x_hi = ((0.37 - 1) / z * f + 1) * res / 2, ((0.37 + 1) / z * f + 1) * res / 2
    y_lo_ndc, y_hi_ndc = (0.61 - 1) / z * f, (0.61 + 1) / z * f
    # rows from the top: r = (1 - y_ndc) * res / 2
    r_lo, r_hi = (1 - y_hi_ndc) * res / 2, (1 - y_lo_ndc) * res / 2
    cols = np.nonzero((sg >= 0).any(0))[0]
    rws = np.nonzero((sg >= 0).any(1))[0]
    pred_cols = (int(np.ceil(x_lo)), int(np.floor(x_hi)))
    # candidate row rules
    rule_top = (int(np.ceil(r_lo)), int(np.floor(r_hi)))                     # integer row index inside
    rule_bot = (int(np.ceil(r_lo - 1)), int(np.floor(r_hi - 1)))             # if rows are indexed from the bottom edge
    rule_c = (int(np.ceil(r_lo - 0.5)), int(np.floor(r_hi - 0.5)))
    got = (int(rws[0]), int(rws[-1]))
    rows_ok["corner_top"] += got == rule_top
    rows_ok["corner_bottom"] += got == rule_bot
    rows_ok["centre"] += got == rule_c
    rows_ok["n"] += 1
    rows_ok["detail"].append({"d": round(float(d), 3), "cols": [int(cols[0]), int(cols[-1])], "pred_cols": pred_cols,
                              "rows": got, "rule_top": rule_top, "rule_bottom": rule_bot, "rule_centre": rule_c})
rows_ok["detail"] = rows_ok["detail"][:5]
p.disconnect(c)
R["row_rule"] = rows_ok

with open(os.path.join(OUT, "tiny_law2.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print(json.dumps(R, indent=1))
