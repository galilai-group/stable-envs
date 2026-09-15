"""TinyRenderer's laws, measured on a synthetic PyBullet scene (no PyFlyt, no numba JIT).

What the port needs and cannot take on faith: the frame the light direction lives in (view or world),
the specular exponent and frame, the alpha rule, the cylinder tessellation (phase, caps, normals),
the texture sampling rule on pybullet_data's plane, and the pixel-coverage rule. Writes
examples/results/step7/rocket-landing/phase0/tiny_law.json and a few PNGs.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "results", "step7",
                   "rocket-landing", "phase0")
OUT = os.path.abspath(OUT)
LDEF = np.array([-50.0, 30.0, 100.0])
LDEF = LDEF / np.linalg.norm(LDEF)
R: dict = {}


def client():
    c = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
    return c


def view(eye, target=(0, 0, 0), up=(0, 0, 1)):
    return p.computeViewMatrix(list(map(float, eye)), list(map(float, target)), list(map(float, up)))


def proj(fov=60.0, near=0.1, far=100.0):
    return p.computeProjectionMatrixFOV(fov, 1.0, near, far)


def shot(c, V, P, res=256, seg=False, **kw):
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if seg else 0
    _, _, rgba, depth, sg = p.getCameraImage(res, res, viewMatrix=V, projectionMatrix=P, flags=flags,
                                             physicsClientId=c, **kw)
    return (np.asarray(rgba, np.uint8).reshape(res, res, 4), np.asarray(depth, np.float32).reshape(res, res),
            np.asarray(sg).reshape(res, res))


def rot_of(V):
    """world->view rotation (rows = right, up, back) from a PyBullet view matrix (column-major)."""
    M = np.array(V, np.float64).reshape(4, 4).T
    return M[:3, :3]


# ------------------------------------------------------------------ 1. light frame + specular, on box faces
c = client()
box = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 1], rgbaColor=[1, 1, 1, 1], physicsClientId=c)
bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, 0, 0], physicsClientId=c)
faces = {"+x": np.array([1, 0, 0.]), "-x": np.array([-1, 0, 0.]), "+y": np.array([0, 1, 0.]),
         "-y": np.array([0, -1, 0.]), "+z": np.array([0, 0, 1.]), "-z": np.array([0, 0, -1.])}
eyes = {"from_+x+y+z": (3.0, 2.5, 2.0), "from_-x-y+z": (-3.0, -2.5, 2.0), "from_+x-y-z": (3.0, -2.5, -2.0),
        "from_-x+y+z_far": (-6.0, 4.0, 5.0)}
L1 = {"faces": {}}
for name, eye in eyes.items():
    V, P = view(eye), proj(50.0)
    Rv = rot_of(V)
    img, dep, sg = shot(c, V, P, 256, lightAmbientCoeff=0.0, lightDiffuseCoeff=1.0, lightSpecularCoeff=0.0)
    spec_img, _, _ = shot(c, V, P, 256, lightAmbientCoeff=0.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=1.0)
    amb_img, _, _ = shot(c, V, P, 256, lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
    # classify pixels by face using the depth-derived normal is overkill: the box is axis aligned, so
    # use the world point of each pixel from depth and pick the face whose plane it lies on.
    Rvi = Rv.T
    M = np.array(V, np.float64).reshape(4, 4).T
    t = M[:3, 3]
    res = 256
    near, far = 0.1, 100.0
    zn = dep.astype(np.float64)
    zeye = far * near / (far - (far - near) * zn)                   # linear depth along -z
    f = 1.0 / np.tan(np.radians(50.0) / 2)
    ys = (2.0 * (np.arange(res) + 0.5) / res - 1.0)[::-1]          # row 0 = top
    xs = (2.0 * (np.arange(res) + 0.5) / res - 1.0)
    X, Y = np.meshgrid(xs, ys)
    pv = np.stack([X * zeye / f, Y * zeye / f, -zeye], -1)          # view-space point
    pw = (pv - t) @ Rvi.T if False else (Rvi @ (pv.reshape(-1, 3) - t).T).T.reshape(res, res, 3)
    hit = sg >= 0
    rec = {}
    for fname, n in faces.items():
        on = hit & (np.abs(pw @ n - 1.0) < 2e-3)
        if on.sum() < 50:
            continue
        # drop a 3-px border of the face to stay clear of silhouettes
        vals = img[on][:, 0].astype(int)
        v = np.bincount(vals).argmax()
        sv = np.bincount(spec_img[on][:, 0].astype(int)).argmax()
        av = np.bincount(amb_img[on][:, 0].astype(int)).argmax()
        n_view = Rv @ n
        rec[fname] = {"px_diffuse_mode": int(v), "px_spec_mode": int(sv), "px_ambient_mode": int(av),
                      "n_world": n.tolist(), "n_view": n_view.round(4).tolist(),
                      "pred_world_int": int(255 * max(0.0, n @ LDEF)),
                      "pred_view_int": int(255 * max(0.0, n_view @ LDEF)),
                      "ndl_world": round(float(n @ LDEF), 4), "ndl_view": round(float(n_view @ LDEF), 4),
                      "npx": int(on.sum())}
    L1["faces"][name] = rec
    Image.fromarray(img[..., :3]).save(os.path.join(OUT, f"tiny_law_box_{name}.png"))
p.disconnect(c)
R["light_frame"] = L1

# ------------------------------------------------------------------ 2. alpha rule
c = client()
back = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 3, 0.1], rgbaColor=[0.0, 0.0, 1.0, 1], physicsClientId=c)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=back, basePosition=[0, 0, -1], physicsClientId=c)
A2 = {}
for a in [0.0, 0.1, 0.5, 0.8, 1.0]:
    front = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.1], rgbaColor=[1.0, 0.8, 0.0, a],
                                physicsClientId=c)
    fid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=front, basePosition=[0, 0, 0], physicsClientId=c)
    V, P = view((0, 0, 6), (0, 0, 0), (0, 1, 0)), proj(40.0)
    img, _, sg = shot(c, V, P, 128, seg=True)
    amb, _, _ = shot(c, V, P, 128, lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
    centre = img[64, 64, :3].tolist()
    centre_amb = amb[64, 64, :3].tolist()
    off = img[10, 10, :3].tolist()
    A2[f"alpha_{a}"] = {"centre_px_default_light": centre, "centre_px_ambient_only": centre_amb,
                        "blue_backdrop_px": off, "centre_seg": int(sg[64, 64]) if sg[64, 64] >= 0 else -1}
    p.removeBody(fid, physicsClientId=c)
p.disconnect(c)
R["alpha"] = A2

# ------------------------------------------------------------------ 3. cylinder tessellation: phase, caps
c = client()
cyl = p.createVisualShape(p.GEOM_CYLINDER, radius=1.0, length=2.0, rgbaColor=[1, 1, 1, 1], physicsClientId=c)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=cyl, basePosition=[0, 0, 0], physicsClientId=c)
V, P = view((0, 0, 8), (0, 0, 0), (0, 1, 0)), proj(20.0)              # straight down: silhouette = 32-gon
img, dep, sg = shot(c, V, P, 512, seg=True, lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
hit = sg >= 0
ys, xs = np.nonzero(hit)
cx, cy = 255.5, 255.5
ang = np.degrees(np.arctan2(-(ys - cy), xs - cx))                  # image y down -> world y up
rad = np.hypot(xs - cx, ys - cy)
# radius as a function of angle: a 32-gon has maxima at its vertices
bins = np.round(ang).astype(int) % 360
rmax = np.zeros(360)
for b_, r_ in zip(bins, rad):
    rmax[b_] = max(rmax[b_], r_)
peaks = [int(i) for i in range(360) if rmax[i] >= rmax[(i - 1) % 360] and rmax[i] >= rmax[(i + 1) % 360]
         and rmax[i] > 0.99 * rmax.max()]
C3 = {"topdown_silhouette_px": int(hit.sum()), "vertex_angle_candidates_deg": peaks[:40],
      "rmax_px": float(rmax.max()), "expected_disc_px_r": None}
# open or capped? look INTO the cylinder from above at an angle; with caps the top is one flat shade,
# without, the far inner wall (lit by outward normals) is visible through the opening.
V2 = view((0, -3.0, 4.0), (0, 0, 0), (0, 0, 1))
img2, dep2, sg2 = shot(c, V2, proj(45.0), 256, seg=True, lightAmbientCoeff=0.0, lightDiffuseCoeff=1.0,
                       lightSpecularCoeff=0.0)
vals = img2[sg2 >= 0][:, 0]
C3["oblique_distinct_grey_levels"] = int(len(np.unique(vals)))
C3["oblique_px"] = int((sg2 >= 0).sum())
# depth at the axis point seen from above: capped -> depth of the top cap (z=+1); open -> the bottom
# cap or the far wall
d_axis = float(dep[256, 256])
zeye = 100.0 * 0.1 / (100.0 - (100.0 - 0.1) * d_axis)
C3["depth_on_axis_from_z8_m"] = round(zeye, 4)
C3["depth_if_top_cap_m"] = 7.0
C3["depth_if_open_to_bottom_cap_m"] = 9.0
C3["depth_if_open_and_no_caps_at_all_m"] = "background (far plane)"
Image.fromarray(img[..., :3]).save(os.path.join(OUT, "tiny_law_cyl_top.png"))
Image.fromarray(img2[..., :3]).save(os.path.join(OUT, "tiny_law_cyl_oblique.png"))
p.disconnect(c)
R["cylinder"] = C3

# ------------------------------------------------------------------ 4. box tessellation: flat or averaged normals?
c = client()
box = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 1], rgbaColor=[1, 1, 1, 1], physicsClientId=c)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, 0, 0], physicsClientId=c)
V, P = view((3, 2.5, 2.0)), proj(50.0)
img, _, sg = shot(c, V, P, 256, seg=True, lightAmbientCoeff=0.0, lightDiffuseCoeff=1.0, lightSpecularCoeff=0.0)
R["box_distinct_levels_3_faces_visible"] = int(len(np.unique(img[sg >= 0][:, 0])))
p.disconnect(c)

# ------------------------------------------------------------------ 5. texture sampling on plane100.obj
c = client()
p.loadURDF("plane.urdf", useFixedBase=True, physicsClientId=c)
tex = np.asarray(Image.open(os.path.join(pybullet_data.getDataPath(), "checker_blue.png")).convert("RGBA"))
T5 = {"tex_shape": list(tex.shape)}
# checker period: run-length along row 0 and column 0
row = tex[0, :, 0].astype(int)
changes = np.nonzero(np.diff(row))[0]
T5["tex_row0_change_cols"] = changes[:12].tolist()
T5["tex_unique_rgba"] = np.unique(tex.reshape(-1, 4), axis=0).tolist()[:6]
# a camera 1.5 m above (0.37, 0.61) looking straight down with a narrow fov: ~0.8 m across the frame
eye = (0.37, 0.61, 1.5)
V, P = view(eye, (0.37, 0.61, 0.0), (0, 1, 0)), proj(30.0)
res = 256
img, dep, sg = shot(c, V, P, res, seg=True, lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
# predicted nearest sample: world (x,y) -> u=(x+100)/2, v=(y+100)/2 -> texel
f = 1.0 / np.tan(np.radians(30.0) / 2)
ys = (2.0 * (np.arange(res) + 0.5) / res - 1.0)[::-1]
xs = (2.0 * (np.arange(res) + 0.5) / res - 1.0)
X, Y = np.meshgrid(xs, ys)
wx = eye[0] + X * 1.5 / f
wy = eye[1] + Y * 1.5 / f
u = (wx + 100.0) / 2.0
v = (wy + 100.0) / 2.0
best = None
for vflip in (False, True):
    for uflip in (False, True):
        for off in (0.0, 0.5):
            uu = (1 - (u % 1.0)) if uflip else (u % 1.0)
            vv = (1 - (v % 1.0)) if vflip else (v % 1.0)
            ti = np.clip(np.floor(uu * 512 + off).astype(int), 0, 511)
            tj = np.clip(np.floor(vv * 512 + off).astype(int), 0, 511)
            for rowsel in ("tj", "ti"):
                pred = tex[tj, ti, :3] if rowsel == "tj" else tex[ti, tj, :3]
                # ambient-only: out = int(texel * white * 1.0) = texel (Kd of the mtl is NOT applied?)
                agree = float((np.all(pred == img[..., :3], axis=-1) & (sg >= 0)).sum() / max(1, (sg >= 0).sum()))
                cand = {"vflip": vflip, "uflip": uflip, "half_texel_offset": off, "index": rowsel, "agree": agree}
                if best is None or agree > best["agree"]:
                    best = cand
T5["best_nearest_hypothesis"] = best
T5["plane_px_unique_colours"] = np.unique(img[sg >= 0][:, :3], axis=0).tolist()[:6]
T5["mtl_Kd_0.588_applied?"] = "compare plane_px_unique_colours with tex_unique_rgba: equal => Kd ignored"
Image.fromarray(img[..., :3]).save(os.path.join(OUT, "tiny_law_plane_tex.png"))
p.disconnect(c)
R["texture"] = T5

# ------------------------------------------------------------------ 6. pixel coverage rule (sub-pixel edge sweep)
c = client()
box = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 0.1], rgbaColor=[1, 0, 0, 1], physicsClientId=c)
p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, 0, 0], physicsClientId=c)
res = 128
rules = {"centre": 0, "floor": 0, "ceil": 0, "n": 0}
detail = []
for d in np.linspace(3.0, 3.6, 25):
    V, P = view((0, 0, d), (0, 0, 0), (0, 1, 0)), proj(60.0)
    img, _, sg = shot(c, V, P, res, seg=True)
    cols = np.nonzero((sg >= 0).any(0))[0]
    filled = int(len(cols))
    f = 1.0 / np.tan(np.radians(30.0))
    half_px = (1.0 / (d - 0.1)) * f * res / 2                        # projected half-width in px
    lo, hi = res / 2 - half_px, res / 2 + half_px                     # edge positions in pixel units
    centre = int(np.floor(hi - 0.5) - np.ceil(lo - 0.5) + 1)          # pixel i covered iff lo <= i+0.5 < hi
    fl = int(np.floor(hi) - np.floor(lo))
    ce = int(np.ceil(hi) - np.ceil(lo))
    rules["centre"] += filled == centre
    rules["floor"] += filled == fl
    rules["ceil"] += filled == ce
    rules["n"] += 1
    detail.append({"d": round(float(d), 4), "filled": filled, "centre": centre, "floor": fl, "ceil": ce,
                   "first_col": int(cols[0]), "last_col": int(cols[-1])})
rules["detail"] = detail[:6]
p.disconnect(c)
R["coverage_rule"] = rules

with open(os.path.join(OUT, "tiny_law.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "light_frame"}, indent=1))
for name, rec in R["light_frame"]["faces"].items():
    print(name)
    for fn, r in rec.items():
        print("  ", fn, "diff px", r["px_diffuse_mode"], "pred world", r["pred_world_int"], "pred view",
              r["pred_view_int"], "spec px", r["px_spec_mode"], "amb", r["px_ambient_mode"], "npx", r["npx"])
