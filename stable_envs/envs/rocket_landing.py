"""PyFlyt rocket landing, rendered by nvdiffrast. VISUAL TIER.

THESE ARE NOT THE OFFICIAL PIXELS. The official frame is PyBullet's TinyRenderer at 480x480 through
PyFlyt's rocket-mounted camera; this cell draws the same scene -- the same 13 primitives, the same
camera, the same one-light fixed-function law, the same checker ground -- through a different
rasterizer, and declares what that costs in ``visual_fidelity``. Step 7 workstream W6; the gate that
admitted it is ``tests/fixtures/regression/join/step7/rocket-landing/findings.md`` and every number below is from
the probes it cites.

What the scene is (the frozen measurement record, census of ``getVisualShapeData``
over six resets: geometry identical, only ``rgba`` changes):

* the rocket: a body cylinder (l 4.77, r 0.185), a fuel-tank cylinder (l 1, r 0.1; PyBullet
  synthesises it from the collision shape, it sits inside the body and draws 0 px), a booster cylinder
  (l 0.5, r 0.25, at z -2), four fin boxes, three leg boxes and a "flame" cylinder (l 2, r 0.15, at
  z -3.4) whose rgba alpha is 0.1 -- and TinyRenderer has NO transparency: alpha 0 is not drawn and
  anything else is drawn opaque (``tiny_law.json: alpha``);
* the pad: one cylinder (l 0.1, r 2) fixed at (0, 0, 0.1);
* the ground: pybullet_data's ``plane100.obj``, a 200 m quad of two triangles under a 512x512 checker
  tiled 100x (2 m tiles), sampled NEAREST with v flipped and the mtl's ``Kd`` ignored
  (``tiny_law.json: texture``, 100 % of pixels reproduced in ``tiny_law3.json``).

How TinyRenderer draws it (``tiny_law_probe.py``, ``tiny_law_probe2.py``, ``tiny_law4_backface.json``;
every rule below was measured on synthetic scenes before it was written down):

* one WORLD-fixed directional light, direction ``(-50, 30, 100)`` normalised, ambient 0.6, diffuse
  0.35, specular 0.05 with exponent 2 on the world z of the reflection built from the UNCLAMPED
  ``n.l`` (a face turned away from the light reads 162, not the ambient-only 153); output truncated
  to uint8; no shadows, no gamma; ``nvdiff.TinyShading`` at its defaults;
* cylinders are the CONVEX HULL of 2 x 32 ring points as ``btShapeHull`` tessellates it: the support
  points of the hull in Bullet's 42 fixed unit-sphere directions, pushed out by the shape's 1 mm
  margin, with the vertex normal being the NORMALISED POSITION (``TinyRendererVisualShapeConverter.cpp``
  lines 211-290 and 655-705). That is why the body's brightest pixel reaches 0.991 of its colour on a
  vertical tube: its rim normals lean toward the light. ``tiny_cylinder`` reproduces the vertex set;
  the triangulation of coplanar caps is scipy's rather than Bullet's HullLibrary's, a declared residual;
* boxes are flat-shaded cubes; pixels are tested at INTEGER coordinates from the bottom-left, not at
  centres (25/25 columns, 13/13 rows), which ``sample_offset=(0.5, 0.5)`` reproduces.

What the state is, and why it carries velocities (coordinator ruling 3, ``findings.md`` G1): the frame
after ``env.step`` draws the base cylinder at pose *t* but the ten links AND the camera
link one 1/240 s substep behind, and that lagged pose is exactly ``(pos - v dt, exp(-w dt) q)`` in
world velocities (position error 0.0, quaternion 1e-16 over 40 steps; 1.8e-15 / 5.6e-16 over the
200-state corpus). So the state is the first 13 entries of the env's own observation --
``ang_vel`` (body frame), quaternion ``xyzw``, ``lin_vel`` (body frame), ``lin_pos`` -- and nothing
else: ``state = obs[:13]``. The five RGB colours and the flame's ignition bit are STYLE, because they
are per-episode appearance: they are sampled per reset, and PyFlyt recolours the flame on the first
ignition transition and never restores it.

Reference bridge: for an arbitrary state the official picture cannot be produced by a teleport (a
teleport syncs the links to the base), so ``reference_render`` teleports the base to the PREVIOUS
substep, sets velocities and runs one raw ``stepSimulation`` with gravity off, iterating the set
velocity until the post-step velocity equals the recorded one (Bullet's substep is semi-implicit and a
spinning composite multibody changes its own velocity by gyroscopic terms). Validated against the
corpus: 199/200 frames byte-identical, the last off by 1 px, base landing error 7e-13 m
(``tests/fixtures/regression/join/step7/rocket-landing/reference/reference_trick_validation.json``).

CPU: ``device="cpu"`` raises. No CPU-native backend in this repo rasterises an arbitrary textured
triangle soup, and the policy is that a CPU request is never served by CUDA; the feasibility of a CPU
triangle rasteriser for this ~600-triangle scene is a measured, separate question.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..spaces import Dict, Discrete, RGBBox

try:
    import nvdiffrast.torch as _nvdr

    _ = _nvdr.RasterizeCudaContext
    NVDIFF_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover
    NVDIFF_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

from ..render import _resources as _RES
from ..render import nvdiff as ND  # noqa: E402

#: PyBullet's physics substep in PyFlyt's Aviary (``physics_hz = 240``); the lag is exactly one of them.
DT = 1.0 / 240.0
#: The rocket's visual primitives in DRAW ORDER, after the ground (0) and before the pad (12). Each link's
#: frame is its URDF joint origin relative to the base (``PyFlyt/models/vehicles/rocket/rocket.urdf``); the
#: visuals themselves sit at the link origin (``getVisualShapeData`` local pose is identity on all 14).
#: ``kind``: ("cyl", length, radius) or ("box", full size). ``rpy`` is URDF roll-pitch-yaw.
ROCKET_LINKS = (
    ("body", ("cyl", 4.77, 0.185), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("fueltank", ("cyl", 1.0, 0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("booster", ("cyl", 0.5, 0.25), (0.0, 0.0, -2.0), (0.0, 0.0, 0.0)),
    ("fin_pos_x", ("box", 0.3, 0.03, 0.3), (0.35, 0.0, 2.051), (0.0, 0.0, 0.0)),
    ("fin_neg_x", ("box", 0.3, 0.03, 0.3), (-0.35, 0.0, 2.051), (0.0, 0.0, 0.0)),
    ("fin_pos_y", ("box", 0.03, 0.3, 0.3), (0.0, 0.35, 2.051), (0.0, 0.0, 0.0)),
    ("fin_neg_y", ("box", 0.03, 0.3, 0.3), (0.0, -0.35, 2.051), (0.0, 0.0, 0.0)),
    ("leg_1", ("box", 0.05, 0.5, 0.05), (0.0, 0.35, -2.4), (0.0, 0.0, 0.0)),
    ("leg_2", ("box", 0.05, 0.5, 0.05), (0.3031, -0.175, -2.4), (0.0, 0.0, 4.188)),
    ("leg_3", ("box", 0.05, 0.5, 0.05), (-0.3031, -0.175, -2.4), (0.0, 0.0, -4.188)),
    ("flame", ("cyl", 2.0, 0.15), (0.0, 0.0, -3.4), (0.0, 0.0, 0.0)),
)
GEOM_NAMES = ("ground",) + tuple(n for n, *_ in ROCKET_LINKS) + ("pad",)
N_GEOMS = len(GEOM_NAMES)
PAD_POS = np.array([0.0, 0.0, 0.1])
PAD_LEN, PAD_RADIUS = 0.1, 2.0
GROUND_HALF = 100.0
#: The flame's two colours: its URDF material, and what ``RocketBaseEnv.step`` writes on the first
#: ignition transition (``self.ignition_color``) and never resets.
FLAME_RGBA = (1.0, 0.8, 0.0, 0.1)
IGNITION_RGBA = (1.0, 0.3, 0.3, 0.8)
FUELTANK_RGBA = (1.0, 1.0, 1.0, 1.0)  # PyBullet's default for a link with no <visual>
#: PyFlyt's rocket camera (``Rocket.__init__``): link 0, offset in the link frame, +30 deg of pitch, a
#: tracking target at the link origin, 90 deg fov, near 0.1, far 255.
CAM_OFFSET = np.array([-1.0, 0.0, 3.0])
CAM_PITCH_DEG = 30.0
CAM_FOV, CAM_NEAR, CAM_FAR = 90.0, 0.1, 255.0
#: Bullet's ``btShapeHull`` support directions (``src/BulletCollision/CollisionShapes/btShapeHull.cpp``,
#: ``sUnitSpherePoints``, the 42 used at ``highres=0``), verbatim.
UNIT_SPHERE_42 = np.array(
    [
        [0.000000, -0.000000, -1.000000],
        [0.723608, -0.525725, -0.447219],
        [-0.276388, -0.850649, -0.447219],
        [-0.894426, -0.000000, -0.447216],
        [-0.276388, 0.850649, -0.447220],
        [0.723608, 0.525725, -0.447219],
        [0.276388, -0.850649, 0.447220],
        [-0.723608, -0.525725, 0.447219],
        [-0.723608, 0.525725, 0.447219],
        [0.276388, 0.850649, 0.447219],
        [0.894426, 0.000000, 0.447216],
        [-0.000000, 0.000000, 1.000000],
        [0.425323, -0.309011, -0.850654],
        [-0.162456, -0.499995, -0.850654],
        [0.262869, -0.809012, -0.525738],
        [0.425323, 0.309011, -0.850654],
        [0.850648, -0.000000, -0.525736],
        [-0.525730, -0.000000, -0.850652],
        [-0.688190, -0.499997, -0.525736],
        [-0.162456, 0.499995, -0.850654],
        [-0.688190, 0.499997, -0.525736],
        [0.262869, 0.809012, -0.525738],
        [0.951058, 0.309013, 0.000000],
        [0.951058, -0.309013, 0.000000],
        [0.587786, -0.809017, 0.000000],
        [0.000000, -1.000000, 0.000000],
        [-0.587786, -0.809017, 0.000000],
        [-0.951058, -0.309013, -0.000000],
        [-0.951058, 0.309013, -0.000000],
        [-0.587786, 0.809017, -0.000000],
        [-0.000000, 1.000000, -0.000000],
        [0.587786, 0.809017, -0.000000],
        [0.688190, -0.499997, 0.525736],
        [-0.262869, -0.809012, 0.525738],
        [-0.850648, 0.000000, 0.525736],
        [-0.262869, 0.809012, 0.525738],
        [0.688190, 0.499997, 0.525736],
        [0.525730, 0.000000, 0.850652],
        [0.162456, -0.499995, 0.850654],
        [-0.425323, -0.309011, 0.850654],
        [-0.425323, 0.309011, 0.850654],
        [0.162456, 0.499995, 0.850654],
    ],
    np.float64,
)
HULL_MARGIN = 0.001  # ``cylZShape->setMargin(0.001)``
CYL_STEPS = 32
#: Device memory per world, the fit ``mujoco_locomotion.py`` carries from ``docs/research/nvdiffrast.md``.
NVD_BYTES_PER_PIXEL, NVD_BYTES_PER_VERTEX, NVD_BUDGET_BYTES = 239, 69, 6 << 30


# --------------------------------------------------------------------------- pure math (CPU-tested)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """``(...,4)`` xyzw -> ``(...,3,3)``. PyBullet's quaternion order."""
    q = np.asarray(q, np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n > 0, 2.0 / np.where(n > 0, n, 1.0), 0.0)
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - s * (y * y + z * z)
    m[..., 0, 1] = s * (x * y - z * w)
    m[..., 0, 2] = s * (x * z + y * w)
    m[..., 1, 0] = s * (x * y + z * w)
    m[..., 1, 1] = 1 - s * (x * x + z * z)
    m[..., 1, 2] = s * (y * z - x * w)
    m[..., 2, 0] = s * (x * z - y * w)
    m[..., 2, 1] = s * (y * z + x * w)
    m[..., 2, 2] = 1 - s * (x * x + y * y)
    return m


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a * b`` in xyzw, i.e. rotate by ``b`` then by ``a``."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        -1,
    )


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * ang
    k = np.where(ang > 1e-12, np.sin(half) / np.where(ang > 1e-12, ang, 1.0), 0.5 - ang * ang / 48.0)
    return np.concatenate([v * k, np.cos(half)], -1)


def quat_from_euler(rpy: np.ndarray) -> np.ndarray:
    """PyBullet's ``getQuaternionFromEuler``: roll about x, then pitch about y, then yaw about z (fixed axes)."""
    r, p, y = np.moveaxis(np.asarray(rpy, np.float64), -1, 0)
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        -1,
    )


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    """PyBullet's ``getEulerFromQuaternion`` (``btMatrix3x3::getEulerZYX``), roll-pitch-yaw."""
    m = quat_to_mat(q)
    sp = -m[..., 2, 0]
    pitch = np.arcsin(np.clip(sp, -1.0, 1.0))
    roll = np.arctan2(m[..., 2, 1], m[..., 2, 2])
    yaw = np.arctan2(m[..., 1, 0], m[..., 0, 0])
    return np.stack([roll, pitch, yaw], -1)


def lagged_pose(state: np.ndarray):
    """``(B,13)`` obs-layout state -> ``(pos_t, q_t, pos_prev, q_prev, v_world, w_world)``, all float64.

    The previous-substep pose is what TinyRenderer draws the links at and where the camera link is:
    ``pos - v dt`` and ``exp(-w dt) q`` with world velocities, measured exact to 1e-15 / 1e-16.
    """
    s = np.asarray(state, np.float64).reshape(-1, 13)
    # RENORMALISE in float64 first. A float32 state carries |q| = 1 +- 3e-8, Bullet renormalises what it is
    # handed, and the two must describe the same rotation to the last bit before either side draws it.
    q = s[:, 3:7]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    R = quat_to_mat(q)
    w = np.einsum("bij,bj->bi", R, s[:, 0:3])
    v = np.einsum("bij,bj->bi", R, s[:, 7:10])
    pos = s[:, 10:13]
    pos_prev = pos - v * DT
    q_prev = quat_mul(quat_from_rotvec(-w * DT), q)
    return pos, q, pos_prev, q_prev, v, w


def camera_pose(pos_prev: np.ndarray, q_prev: np.ndarray):
    """PyFlyt's rocket camera from the LAGGED link-0 pose: ``(eye (B,3), mat (B,3,3))`` in ``nvdiff.Camera`` terms.

    ``Camera.view_mat``: position = link + R_link @ offset; rotation = the link's Euler angles with 30
    degrees added to the pitch (not gimballed); up = that rotation's z; target = the link origin
    (``is_tracking_camera``); then ``computeViewMatrix(eye, target, up)`` = gluLookAt. ``mat`` columns
    are right / up / BACK, so ``nvdiff.Camera.matrices`` rebuilds the same view matrix.
    """
    R = quat_to_mat(q_prev)
    eye = pos_prev + np.einsum("bij,j->bi", R, CAM_OFFSET)
    rpy = euler_from_quat(q_prev)
    rpy[:, 1] += math.radians(CAM_PITCH_DEG)
    Rc = quat_to_mat(quat_from_euler(rpy))
    up = Rc[:, :, 2]
    f = pos_prev - eye
    f = f / np.linalg.norm(f, axis=-1, keepdims=True)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    u = np.cross(s, f)
    return eye, np.stack([s, u, -f], -1)


def tiny_cylinder(length: float, radius: float, steps: int = CYL_STEPS, margin: float = HULL_MARGIN):
    """TinyRenderer's cylinder: ``btShapeHull`` of the 2 x ``steps`` ring, normals = normalised positions.

    Returns ``(vert (V,3), norm (V,3), face (F,3))`` with the faces wound outward. The vertex SET is
    Bullet's exactly (the support point of the ring in each of the 42 directions, first-maximum on
    ties, plus ``margin`` along the direction); the triangulation of each planar cap is scipy's.
    """
    from scipy.spatial import ConvexHull

    i = np.arange(steps)
    ring = np.stack([radius * np.sin(2 * np.pi * i / steps), radius * np.cos(2 * np.pi * i / steps)], -1)
    pts = np.concatenate(
        [
            np.concatenate([ring, np.full((steps, 1), length / 2.0)], -1),
            np.concatenate([ring, np.full((steps, 1), -length / 2.0)], -1),
        ]
    )
    # btConvexHullShape stores the two rings interleaved (top_i, bottom_i); ties resolve to the FIRST
    # maximum, so keep Bullet's order.
    pts = pts.reshape(2, steps, 3).transpose(1, 0, 2).reshape(-1, 3)
    sup = []
    for d in UNIT_SPHERE_42:
        k = int(np.argmax(pts @ d))
        sup.append(pts[k] + margin * d)
    sup = np.unique(np.round(np.asarray(sup), 12), axis=0)
    hull = ConvexHull(sup)
    vert = sup[hull.vertices]
    remap = {int(v): i for i, v in enumerate(hull.vertices)}
    face = np.array([[remap[int(a)] for a in tri] for tri in hull.simplices], np.int64)
    # wind outward: scipy's simplex order is arbitrary, its facet equation is not
    for k, tri in enumerate(face):
        a, b, c = vert[tri]
        if np.dot(np.cross(b - a, c - a), hull.equations[k, :3]) < 0:
            face[k] = tri[[0, 2, 1]]
    norm = vert / np.linalg.norm(vert, axis=-1, keepdims=True)
    return vert, norm, face


def build_scene_arrays():
    """The 13 geoms as one triangle soup in each geom's own frame, ground UVs and the checker texture."""
    V, N, F, VG, UV = [], [], [], [], []
    nv = 0

    def add(g, v, n, f, uv=None):
        nonlocal nv
        V.append(v)
        N.append(n)
        F.append(f + nv)
        VG.append(np.full(len(v), g, np.int64))
        UV.append(np.zeros((len(v), 2)) if uv is None else uv)
        nv += len(v)

    gq = np.array(
        [
            [GROUND_HALF, -GROUND_HALF, 0.0],
            [GROUND_HALF, GROUND_HALF, 0.0],
            [-GROUND_HALF, GROUND_HALF, 0.0],
            [-GROUND_HALF, -GROUND_HALF, 0.0],
        ]
    )
    guv = np.array([[100.0, 0.0], [100.0, 100.0], [0.0, 100.0], [0.0, 0.0]])  # plane100.obj's vt
    add(0, gq, np.tile([0.0, 0.0, 1.0], (4, 1)), np.array([[0, 1, 2], [0, 2, 3]], np.int64), guv)
    for g, (name, kind, _pos, _rpy) in enumerate(ROCKET_LINKS, start=1):
        if kind[0] == "cyl":
            add(g, *tiny_cylinder(kind[1], kind[2]))
        else:
            add(g, *ND._box(np.asarray(kind[1:], np.float64) / 2.0))
    add(N_GEOMS - 1, *tiny_cylinder(PAD_LEN, PAD_RADIUS))
    return (np.concatenate(V), np.concatenate(N), np.concatenate(F), np.concatenate(VG), np.concatenate(UV))


def link_local_frames():
    """``(pos (11,3), quat (11,4))`` of each rocket link relative to the base, from the URDF joint origins."""
    pos = np.array([p for _, _, p, _ in ROCKET_LINKS], np.float64)
    q = quat_from_euler(np.array([r for _, _, _, r in ROCKET_LINKS], np.float64))
    return pos, q


def geom_poses(state: np.ndarray):
    """``(B,13)`` -> ``xpos (B,13,3)``, ``xmat (B,13,3,3)``: base at *t*, links at *t - dt*, ground and pad fixed."""
    pos, q, pos_prev, q_prev, _, _ = lagged_pose(state)
    B = len(pos)
    lp, lq = link_local_frames()
    xpos = np.zeros((B, N_GEOMS, 3))
    xq = np.zeros((B, N_GEOMS, 4))
    xq[:, :, 3] = 1.0
    xpos[:, 1], xq[:, 1] = pos, q  # the body: pose t
    Rp = quat_to_mat(q_prev)
    for i in range(1, len(ROCKET_LINKS)):  # links: pose t - dt
        xpos[:, 1 + i] = pos_prev + np.einsum("bij,j->bi", Rp, lp[i])
        xq[:, 1 + i] = quat_mul(q_prev, np.broadcast_to(lq[i], (B, 4)))
    xpos[:, N_GEOMS - 1] = PAD_POS
    return xpos, quat_to_mat(xq)


#: Column order of the CORPUS style rows (``reference/states.npz: style``, written before the space
#: existed): body, fin, leg, booster, pad RGBs then the flame bit. The ENV's style layout is the
#: variation space's ``flat_layout``, which ``gymnasium.spaces.Dict`` sorts alphabetically; nothing
#: below assumes either order -- leaves are addressed by path.
CORPUS_STYLE_PATHS = (
    "rocket.body_color",
    "rocket.fin_color",
    "rocket.leg_color",
    "rocket.booster_color",
    "pad.color",
    "flame.ignited",
)


def _make_variation_space() -> Dict:
    return Dict(
        {
            "rocket": Dict(
                {
                    "body_color": RGBBox(init_value=(255, 204, 0)),
                    "fin_color": RGBBox(init_value=(51, 51, 51)),
                    "leg_color": RGBBox(init_value=(0, 0, 0)),
                    "booster_color": RGBBox(init_value=(51, 51, 51)),
                }
            ),
            "pad": Dict({"color": RGBBox(init_value=(255, 255, 255))}),
            "flame": Dict({"ignited": Discrete(2, init_value=0)}),
        }
    )


def style_layout() -> dict:
    """``{path: (offset, width)}`` of the env's flat style tensor, from the space itself."""
    return {p: (o, w) for p, o, w in _make_variation_space().flat_layout()}


def style_dim() -> int:
    return sum(w for _, w in style_layout().values())


def corpus_style_to_env(rows: np.ndarray) -> np.ndarray:
    """``(B,16)`` corpus-order style rows -> ``(B, style_dim)`` rows in the env's own layout."""
    rows = np.asarray(rows, np.float64).reshape(-1, 16)
    lay = style_layout()
    out = np.zeros((len(rows), style_dim()))
    col = 0
    for path in CORPUS_STYLE_PATHS:
        o, w = lay[path]
        out[:, o : o + w] = rows[:, col : col + w]
        col += w
    return out


def material_table(style: np.ndarray) -> np.ndarray:
    """``(B, style_dim)`` env-layout style -> ``(B,13,3)`` material colours, the way the reference applies them.

    Rocket links take ``value / 255`` (``changeVisualShape``); the pad takes the 3-decimal rounding the
    URDF rewrite applies (``f"{v:.3f}"``); the fuel tank is white; the ground's material is white (the
    checker is the texture); the flame is its URDF colour or the ignition colour by the style bit.
    Verified against ``getVisualShapeData`` over the corpus to 3e-8 (``reference/meta.json``).
    """
    lay = style_layout()
    st = np.asarray(style, np.float64).reshape(-1, style_dim())
    B = len(st)
    leaf = lambda p: st[:, lay[p][0] : lay[p][0] + lay[p][1]]  # noqa: E731
    body, fin, leg, booster, pad = (leaf(p) / 255.0 for p in CORPUS_STYLE_PATHS[:5])
    pad_q = np.vectorize(lambda x: float(f"{x:.3f}"))(pad)
    ignited = leaf("flame.ignited")[:, 0] > 0.5
    out = np.ones((B, N_GEOMS, 3))
    out[:, 1] = body
    out[:, 2] = FUELTANK_RGBA[:3]
    out[:, 3] = booster
    out[:, 4:8] = fin[:, None]
    out[:, 8:11] = leg[:, None]
    out[:, 11] = np.where(ignited[:, None], np.asarray(IGNITION_RGBA[:3]), np.asarray(FLAME_RGBA[:3]))
    out[:, 12] = pad_q
    return out


def checker_texture() -> np.ndarray:
    """pybullet_data's ``checker_blue.png`` as an (H,W,3) uint8 array FLIPPED vertically.

    TinyRenderer indexes v from the bottom row of the image (``tiny_law.json``: ``vflip``); nvdiffrast
    indexes from row 0, so the flip is done once here and the UVs stay the OBJ's.
    """
    import pybullet_data
    from PIL import Image

    path = os.path.join(pybullet_data.getDataPath(), "checker_blue.png")
    a = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    return np.ascontiguousarray(a[::-1])


def _optional_import_errors() -> dict:
    """The optional dependencies a RENDER needs beyond nvdiffrast, probed by ``find_spec`` -- CHEAP, because
    this runs at import on every machine (the import-time probe cost 0.87 s in review), and the
    registration gate is the same either way: a package that is present is one this module can use."""
    import importlib.util

    out = {}
    for mod in ("scipy", "pybullet_data", "PIL"):
        try:
            found = importlib.util.find_spec(mod) is not None
        except Exception as e:  # pragma: no cover
            found, why = False, f"{type(e).__name__}: {e}"
        else:
            why = "not installed"
        if not found:
            out[mod] = why
    return out


def missing_backend() -> str | None:
    """Why ``PyFlytRocketLanding`` is absent or cannot render on this machine, or ``None``.

    nvdiffrast (the rasterizer), scipy (the hull tessellation), pybullet_data (the checker texture --
    its provenance rests on the pybullet 3.2.7 declaration, the package that ships ``checker_blue.png``)
    and Pillow (reading it). Same reporter as the locomotion tier's ``missing_visual_backend``.
    """
    parts = []
    if NVDIFF_IMPORT_ERROR is not None:
        parts.append(f"nvdiffrast: {NVDIFF_IMPORT_ERROR}")
    parts += [f"{k}: {v}" for k, v in _optional_import_errors().items()]
    return "; ".join(parts) if parts else None


# --------------------------------------------------------------------------- the reference bridge

_REF_CACHE: dict = {}


def _reference_make(size: int):
    """PyFlyt rocket-landing env at ``size``.

    Production has no gymnasium id for this cell. Tests install a certification factory on
    ``_reference_make`` (see ``tests.helpers.reference.certification``).
    """
    hit = _REF_CACHE.get(size)
    if hit is not None:
        return hit
    raise LookupError(
        "PyFlytRocketLanding has no production gymnasium id; the certification overlay in "
        "tests.helpers.reference.certification installs the historical reference factory"
    )


def _reference_apply_style(p, c, rid, pad, style_row: np.ndarray) -> None:
    """``style_row`` in the ENV layout (``style_layout``)."""
    mat = material_table(style_row)[0]
    p.changeVisualShape(rid, -1, rgbaColor=[*mat[1], 1.0], physicsClientId=c)
    for link, g in ((1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)):
        p.changeVisualShape(rid, link, rgbaColor=[*mat[g], 1.0], physicsClientId=c)
    o, _ = style_layout()["flame.ignited"]
    ignited = float(np.asarray(style_row).reshape(-1)[o]) > 0.5
    p.changeVisualShape(rid, 9, rgbaColor=list(IGNITION_RGBA if ignited else FLAME_RGBA), physicsClientId=c)
    p.changeVisualShape(pad, -1, rgbaColor=[*mat[12], 1.0], physicsClientId=c)


#: DOCUMENTED_LIMITATION (Step 7, CLOSURE-1). The shared ``nvdiff`` transform is bitwise independent of
#: the batch width on cuda -- the device every nvdiff cell renders on, and the one the mandatory
#: ``check_batch_size_invariance`` is measured on (0 at 1, 7, 64 and 4096 for this cell) -- and on
#: arm64 CPU. On x86-64 CPU the clip outputs differ from their full-batch values by ONE float32 ulp at
#: some narrow widths: POSITIONS at widths 1-6 (3.05e-05 at a 300 m coordinate scale) and NORMALS at
#: widths 4 and 8 (5.96e-08 near 1.0); world positions never. Bitwise from width 7 up for positions and
#: at every other width for normals. The width sweep is
#: ``closure1/probe_clip_cpu_batch_x86_widthsweep.json``; two earlier samples skipped widths 4-6 and so
#: reported a narrower limitation than this.
#:
#: It reaches no frame this cell produces: ``PyFlytRocketLanding`` raises on ``device="cpu"`` by design, so
#: its CPU transform path exists only so the geometry can be unit-tested without a GPU. The cause is
#: measured and is not this repo's arithmetic -- ``nvdiff.mvp_rows`` is bit-identical at every width on
#: every machine measured, and so are the normals and world positions; the residual is
#: ``torch.sparse.mm``'s freedom to reorder a four-term sum for a narrow batch
#: (``tests/fixtures/regression/join/step7/rocket-landing/closure1/probe_clip_cpu_batch_x86_threads1.json``, with
#: ``_threads2`` and ``_threads4``, and ``tests/fixtures/regression/join/step7/closure/nvdiff_cpu_batch/``).
#:
#: The fix is written, measured and NOT applied in Step 7:
#: the frozen measurement record writes the sum out and removes the
#: kernel's freedom, but on cuda it moves four of the six re-frozen locomotion cells by 2.7e-07 of
#: mean_abs and 1e-09 of SSIM (``closure1/digest_vs_refreeze.json``), and a re-freeze requires a
#: certification failure with a causal reason rather than an implementation that merely differs. It is
#: the nvdiff candidate, under its own clean re-verification pass. It addresses the POSITIONS only; the
#: normals share the same sparse stage and would need the same treatment.
DOCUMENTED_LIMITATIONS = (
    "nvdiff's CPU transform is batch-size invariant to within one float32 ulp at every width, and "
    "bitwise at every width on arm64; on x86-64 the clip positions move by one ulp at widths 1-6 and "
    "the normals at widths 4 and 8 (CLOSURE-1). Cuda, the certified device, is bitwise invariant at "
    "every width. No frame this cell produces is affected: device='cpu' raises. See findings.md "
    "section M7 and tests/fixtures/regression/join/step7/rocket-landing/closure1/."
)


#: The contact band the reference bridge cannot cross. The rocket's collision extent reaches 3.40 m
#: below the base origin (link AABBs, upright, measured in the reference env) and the pad's top face is
#: at z = 0.152; a state whose links sit within CONTACT_BAND_MARGIN of the pad top / ground puts a
#: ground contact into the recipe's one substep, which Bullet resolves with a split-impulse position
#: correction the velocity fixed point cannot absorb. Measured share of such states in the cell's own
#: ``sample_states`` distribution: REFERENCE_UNREACHABLE_SHARE (1 of 960 over 15 seeds x 64 states,
#: ``tests/fixtures/regression/join/step7/rocket-landing/tier/reference_rejections_local.json``; the rejected state:
#: height 2.07 m, tilt 3.4 deg). The sampled distribution is NOT narrowed for it.
PAD_TOP_Z = float(PAD_POS[2] + PAD_LEN / 2.0)
CONTACT_BAND_MARGIN = 0.05
REFERENCE_UNREACHABLE_SHARE = 1.0 / 960.0


def _lowest_link_z(p, c, rid) -> float:
    """The lowest z of the rocket's link collision AABBs at its CURRENT pose in the client."""
    n = p.getNumJoints(rid, physicsClientId=c)
    return min(p.getAABB(rid, link, physicsClientId=c)[0][2] for link in range(-1, n))


def _reference_frame(env, state: np.ndarray, style_row: np.ndarray) -> np.ndarray:
    """One official frame at an arbitrary state: the previous-substep teleport with a velocity fixed point.
    Raises when the recipe does not reach the requested pose (the strict form the validators use)."""
    frame, unreachable = _reference_frame_flagged(env, state, style_row, contact_fallback=False)
    return frame


def _reference_frame_flagged(env, state: np.ndarray, style_row: np.ndarray, contact_fallback: bool = True):
    """``(frame, unreachable)``. The LAGGED official picture (links and camera one substep behind the
    base) via the previous-substep teleport and the velocity fixed point. When the recipe does not reach
    the requested pose AND the pose is in the contact band (``PAD_TOP_Z`` + ``CONTACT_BAND_MARGIN``, any
    link AABB), the lagged picture is unreachable by this bridge: with ``contact_fallback`` the frame
    returned is the SYNCED picture of the requested pose (teleport, no substep -- the convention the
    reference goal image is rendered under) and ``unreachable`` is True, so a caller can record and exclude the
    row; without it, or when the failure is outside the band (a genuine bridge failure), it raises."""
    import pybullet as p

    u = env.unwrapped
    c, rid, pad = u.env._client, u.env.drones[0].Id, u.landing_pad_id
    _reference_apply_style(p, c, rid, pad, style_row)
    pos, q, pos_prev, q_prev, v, w = (a[0] for a in lagged_pose(state))
    p.setGravity(0, 0, 0, physicsClientId=c)
    v_set, w_set = v.copy(), w.copy()
    for _ in range(6):
        p.resetBasePositionAndOrientation(rid, list(pos_prev), list(q_prev), physicsClientId=c)
        p.resetBaseVelocity(rid, list(v_set), list(w_set), physicsClientId=c)
        p.stepSimulation(physicsClientId=c)
        v1, w1 = (np.array(a) for a in p.getBaseVelocity(rid, physicsClientId=c))
        if np.abs(v1 - v).max() < 1e-13 and np.abs(w1 - w).max() < 1e-13:
            break
        v_set, w_set = v_set + (v - v1), w_set + (w - w1)
    p.setGravity(0, 0, -9.81, physicsClientId=c)
    got_pos, got_q = (np.array(a) for a in p.getBasePositionAndOrientation(rid, physicsClientId=c))
    # ``q`` is unit (lagged_pose renormalised it); Bullet's is too. Compare ROTATIONS, sign-free:
    # 1 - |q . q_got| is 0 for the same rotation and ~theta^2/8 for a small angle theta.
    pos_err = float(np.abs(got_pos - pos).max())
    rot_err = float(1.0 - abs(np.dot(got_q / np.linalg.norm(got_q), q)))
    if pos_err > 1e-6 or rot_err > 1e-9:
        # the requested pose itself: is it in the contact band?
        p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
        p.resetBaseVelocity(rid, list(v), list(w), physicsClientId=c)
        in_band = _lowest_link_z(p, c, rid) <= PAD_TOP_Z + CONTACT_BAND_MARGIN
        if contact_fallback and in_band:
            return np.asarray(u.render(), np.uint8)[..., :3].copy(), True
        raise AssertionError(
            f"PyFlytRocketLanding: the reference did not reach the requested pose (pos {pos_err:.3e} m, "
            f"1-|q.q'| {rot_err:.3e}; {'in' if in_band else 'outside'} the contact band); every "
            f"fidelity number computed from it would compare two different scenes"
        )
    return np.asarray(u.render(), np.uint8)[..., :3].copy(), False


# --------------------------------------------------------------------------- the env


class _Nvd:
    __slots__ = ("br", "size", "ss", "chunk")


_NVD_CACHE = _RES.BoundedCache(8, name="rocket.nvd", on_evict=ND.close_cached_renderer)


class PyFlytRocketLanding(Env):
    """PyFlyt rocket landing, rendered by nvdiffrast. State is obs[:13].

    VISUAL TIER — these are NOT the official pixels. The frame is a different rasterization of the same
    scene from the same state: the same 13 primitives, the same rocket-mounted camera one substep behind
    the body, the same one-light TinyRenderer law, the same nearest-sampled checker ground. The
    residual is silhouettes, checker edges and the far ground where tiles are sub-pixel, and the
    triangulation of the cylinders' caps. See ``visual_fidelity`` (measured, job 6410105) and
    the module docstring for what was measured.

    Suitable for generating training frames live. NOT a substitute for a stored dataset when a policy
    will be evaluated on official PyFlyt pixels. See visual_note.
    """

    family = "pyflyt"
    domain = "rocket_landing"
    requires_cuda = True

    name = "pf-rocket-landing"
    reference_id = None
    #: Registered default: 480 x 480 x 3 = 691,200 B, the observation PyFlyt actually returns.
    default_size = 480
    default_supersample = 1  # TinyRenderer has no anti-aliasing
    state_dim = 13
    state_names = (
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
        "quat_x",
        "quat_y",
        "quat_z",
        "quat_w",
        "lin_vel_x",
        "lin_vel_y",
        "lin_vel_z",
        "pos_x",
        "pos_y",
        "pos_z",
    )
    fidelity_tier = "visual"
    parity_interior_min = None
    parity_note = ""
    #: MEASURED on an RTX 3090, worst of four independent 64-state draws, and copied here BY HAND from
    #: ``tests/fixtures/regression/join/step7/rocket-landing/tier/job6410105/gpu2106.json``: a threshold a script
    #: writes into the file it then tests is not a threshold.
    #:
    #: The three headline numbers are the worst draw's. The three thresholds are the DMC family's
    #: margin convention applied to that worst draw -- ceiling ~1.3x, floors ~0.93-0.95x, rounded to
    #: round numbers -- and not values chosen to clear it; the frozen measurement record
    #: computes them from the JSON and prints exactly 10.0 / 0.91 / 0.80
    #: (``tier/job6410105/declaration_block.json``). The same numbers came out of two jobs on two
    #: different 3090s (6408817/gpu2109 and 6410105/gpu2106) to every digit printed here.
    #:
    #: ``ssim_min`` 0.80 is 0.9368x the measurement, and that is the DMC convention rather than a
    #: looser one: the five DMC entries realise 0.9322-0.9533 (Walker 0.9322, Cheetah 0.9380, Hopper
    #: 0.9357, Humanoid 0.9447, Swimmer 0.9533), and this sits inside that range. The six nvdiff
    #: locomotion cells realise 0.9668-0.9786, but a floor is only fragile or generous relative to how
    #: reproducible the measurement is, and those cells' SSIM varies by 0.0011-0.0024 across draws
    #: while this one varies by 0.0139 -- an order of magnitude more, because a rocket at 500 m and a
    #: rocket at 2 m are different pictures. Measured in units of its own draw-to-draw spread, this
    #: floor is the TIGHTEST in the repo: margin/spread 3.86 here, against 4.12-22.8 across the DMC
    #: five and 11.1-28.7 across the nvdiff six. A 0.97x floor would be 0.829, i.e. 1.84 spreads below
    #: the worst draw, which is the fragility the convention exists to avoid.
    #:
    #: ``reference_unreachable_share`` is the honest part of the measurement: ``reference_render``
    #: cannot produce the LAGGED official picture for a state in the contact band (see
    #: ``_reference_frame_flagged`` and ``REFERENCE_UNREACHABLE_SHARE``), so those rows are flagged and
    #: excluded, and the share they were measured at is declared beside the numbers rather than left
    #: for a reader to discover. It is 1 state in 960 of this env's own ``sample_states`` distribution,
    #: and none of the four certification draws drew one.
    visual_fidelity = {
        "mean_abs": 7.71,
        "within8_frac": 0.961,
        "ssim": 0.854,
        "exact_frac": 0.959,
        "interior_exact_frac": 0.967,
        "draw_spread": {"mean_abs": [7.482, 7.709], "within8_frac": [0.961, 0.962], "ssim": [0.854, 0.868]},
        "under_variation": {"mean_abs": 7.646, "within8_frac": 0.958, "ssim": 0.896, "exact_frac": 0.903},
        "reference_unreachable_share": REFERENCE_UNREACHABLE_SHARE,
        "reference_unreachable_rows_in_draws": 0,
        "mean_abs_max": 10.0,
        "within8_min": 0.91,
        "ssim_min": 0.80,
        "measured_at": (
            "480 px, supersample 1, default appearance, worst of 4 independent 64-state draws of "
            "sample_states (default_rng(1000..1003)) over the rows the reference bridge reaches, "
            "against PyFlyt 0.29.0 / pybullet 3.2.7 TinyRenderer frames of the rocket-landing env "
            "at the lagged pose; NVIDIA GeForce RTX 3090 gpu2106, SLURM job 6410105, tree 726241e, "
            "torch 2.14.0+cu126, nvdiffrast 0.4.0 "
            "(tests/fixtures/regression/join/step7/rocket-landing/tier/job6410105/)"
        ),
        "reference_versions": {"PyFlyt": "0.29.0", "pybullet": "3.2.7"},
        "measured_with": {"nvdiffrast": "0.4.0", "torch": "2.14.0+cu126"},
    }
    #: The ONE cell in the repo where ``check_colors_apply_exactly``'s literal-RGB bound is declared
    #: inapplicable, and the declaration is narrow on purpose: these five leaves, named one by one,
    #: and nothing else. Sami's ruling of 2026-09-16, on the packet at
    #: ``tests/fixtures/regression/join/step8/certification/colour_check_decision.md``.
    #:
    #: The mechanism is the reference's own shading law, not ours. bullet3's TinyRenderer computes
    #: ``out_i = int(255 * c_i * (A + D*max(0, n.l) + S*max(0, r_z)**spec) * L_i)``, truncated, with
    #: ``A=0.6, D=0.35, S=0.05, spec=2`` -- and the specular MULTIPLIES the material colour rather
    #: than adding white the way MuJoCo's does, so the bracket never reaches 1. Maximised over a
    #: 2001x4001 grid of unit normals it peaks at 0.993567, which puts the check's own probe targets
    #: out of reach at their brightest: 200 -> 198, 33 -> 32, 17 -> 16. The exact triple therefore
    #: cannot appear in the frame for ANY correct implementation of this reference, and it does not
    #: appear in the OFFICIAL PyFlyt frames either (brightest body pixel 0.991*int(255*c)).
    #:
    #: What still carries the wiring claim -- which is what the check exists to establish -- is
    #: unaffected: ``check_variations_do_something``, the under-variation fidelity numbers, and the
    #: colour table's agreement with ``getVisualShapeData`` to 1.19e-08 on every geom. And the probe
    #: still runs: ``check_colors_apply_exactly`` records ``best_abs_diff`` per leaf, so a knob that
    #: comes off its primitive still moves a number even while the literal bound is not asserted.
    #:
    #: NOT a tolerance and NOT a threshold: the bound is unchanged everywhere it is asserted. A
    #: law-aware successor (compare our response to the knob against the OFFICIAL renderer's response
    #: to the same knob) is a PROPOSAL only -- see docs/research/step8_certification.md.
    color_exactness_applicability = {
        "leaves": ("pad.color", "rocket.body_color", "rocket.booster_color", "rocket.fin_color", "rocket.leg_color"),
        "status": "not_applicable: literal RGB equality is unreachable under the reference's own law",
        "mechanism": "bullet3 TinyRenderer's MULTIPLICATIVE one-light law "
        "out_i = int(255*c_i*(A + D*max(0,n.l) + S*max(0,r_z)**spec)*L_i), truncated, "
        "A=0.6 D=0.35 S=0.05 spec=2, light (-50,30,100) normalised, no shadow, no gamma",
        "max_shading_factor": 0.993567,
        "reachable_ceiling_for_the_probe_targets": {"200": 198, "33": 32, "17": 16},
        "official_renderer_fails_the_same_check": True,
        "official_renderer_evidence": "the official PyFlyt 0.29.0 / pybullet 3.2.7 frame's brightest "
        "body pixel is 0.991*int(255*c); the requested triple is absent "
        "from the OFFICIAL frames too",
        "evidence": (
            "tests/fixtures/regression/join/step8/certification/colour_law_ceiling.json",
            "tests/fixtures/regression/join/step8/certification/colour_check_decision.md",
            "tests/fixtures/regression/join/step7/rocket-landing/findings.md:251-257,1209-1220",
            "tests/fixtures/regression/join/step7/rocket-landing/review_phase1.md:885-894",
        ),
        "measured_against": "PyFlyt 0.29.0, pybullet 3.2.7 (bullet3 TinyRenderer.cpp Shader::fragment)",
        "ruled_by": "Sami, 2026-09-16, on tests/fixtures/regression/join/step8/certification/colour_check_decision.md",
        "scope_note": "scoped to these five lit leaves only; any other RGB leaf on this env stays "
        "judged against literal equality",
        "successor": "PROPOSAL only, not implemented and not mandatory: compare this renderer's "
        "response to the knob against the OFFICIAL renderer's response to the same knob",
    }
    visual_note = (
        "A different rasterizer (nvdiffrast) drawing TinyRenderer's scene under TinyRenderer's law. "
        "Residual: silhouette pixels (different edge rules and float paths), checker-tile edges "
        "(sub-pixel UV rounding under nearest sampling) and the far ground where tiles are "
        "sub-pixel, scipy's rather than Bullet's triangulation of the hull cylinders' planar caps "
        "(Gouraud interpolation across a cap differs by the diagonal chosen), and the 61-px flame. "
        "NOT a substitute for a stored dataset when a policy will be evaluated on official pixels. "
        + DOCUMENTED_LIMITATIONS
    )

    # ------------------------------------------------------------------ state
    #
    # The public :data:`State` is the 13-number observation grouped by what it describes: one rigid
    # body, its attitude, and the two velocities the camera lag reads. The offsets are the Step-7
    # packed columns unchanged, in ``state_names`` order, so ``pack`` writes the frozen row and the
    # corpus round-trips exactly. The quaternion is ONE field of width 4 rather than four scalars,
    # because it is one quantity with a norm constraint and splitting it invites a caller to write
    # three of its components.
    #
    # No field has offset ``None``: this cell ships a renderer and no simulator (PyFlyt's dynamics
    # stay in PyFlyt), so the packed row IS the whole state and ``step``/``transition`` raise the
    # contract's refusal from the base class.
    state_layout = (
        (("rocket", "angular_velocity"), 3, 0),
        (("rocket", "orientation"), 4, 3),
        (("rocket", "linear_velocity"), 3, 7),
        (("rocket", "position"), 3, 10),
    )

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str | None = None,
        backend: str = "auto",
        seed: int = 0,
    ):
        """``num_envs`` rockets on ``device``.

        The CPU refusal below is BY DESIGN and is checked before anything is built: a ``device="cpu"``
        request is never quietly served by CUDA (plan §2's device-honesty rule), and this cell has no
        CPU textured-triangle rasteriser to serve it with.
        """
        if missing_backend() is not None:  # pragma: no cover
            raise ImportError(f"PyFlytRocketLanding cannot render in this interpreter: {missing_backend()}")
        device = Env.resolve_device(device)
        if str(device).startswith("cpu"):
            raise NotImplementedError(
                "PyFlytRocketLanding has no CPU backend: it needs a CPU textured-triangle rasteriser, which this "
                "repo does not have (cpu3d/raycast3d cast MuJoCo primitives). A CPU request is never served "
                "by CUDA."
            )
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        self._nvd()

    def _close(self) -> None:
        """Drop this env's reference to the shared nvdiffrast renderer.

        ``_NVD_CACHE`` is keyed by ``(size, supersample, device)`` and deliberately SHARED between
        envs — building a ``BatchRenderer`` allocates a CUDA context and the scene's vertex buffers,
        and two envs at the same resolution should not pay for it twice. So ``close`` does not tear
        the cache down: another live env may be rendering through the same entry, and freeing it here
        would be a use-after-free in that env rather than a leak in this one. What ``close`` does own
        is this env's refusal to render afterwards, which ``Env.close`` installs.
        """
        return None

    # ---- renderer -----------------------------------------------------------

    def _nvd(self, size: int | None = None, ss: int | None = None) -> _Nvd:
        size = int(self.size if size is None else size)
        ss = int(self.ss if ss is None else ss)
        key = (size, ss, str(self.device))
        hit = _NVD_CACHE.get(key)
        if hit is not None:
            return hit
        vert, norm, face, vg, uv = build_scene_arrays()
        rgba = np.ones((N_GEOMS, 4))
        rgba[11] = FLAME_RGBA
        tex_of_geom = np.full(N_GEOMS, -1, np.int64)
        tex_of_geom[0] = 0
        scene = ND.Scene.from_triangles(
            vert,
            norm,
            face,
            vg,
            rgba,
            uv=uv,
            textures={0: checker_texture()},
            tex_of_geom=tex_of_geom,
            device=str(self.device),
            tex_policy="none",
        )
        c = _Nvd()
        c.size, c.ss = size, ss
        c.br = ND.BatchRenderer(
            None,
            res=size,
            device=str(self.device),
            supersample=ss,
            mipmap=False,
            skybox=False,
            quant="trunc",
            scene=scene,
            shading="tiny",
            tiny=ND.TinyShading(),
            tex_filter="nearest",
            sample_offset=(0.5, 0.5),
        )
        c.br.owns_scene = True
        c.br.background = np.array([1.0, 1.0, 1.0])  # TinyRenderer's clear colour
        per_world = NVD_BYTES_PER_PIXEL * (size * ss) ** 2 + NVD_BYTES_PER_VERTEX * scene.V
        c.chunk = max(1, int(NVD_BUDGET_BYTES // max(1, per_world)))
        _NVD_CACHE.put(key, c)
        return c

    # ---- variation ----------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """The five colours sampled per reset, plus the flame's ignition bit. Defaults = the official
        appearance of a reset WITHOUT a variation request: URDF yellow / grey / black, a white pad (no
        material), the flame unignited. ``environment.sky_color`` is deliberately absent: it is sampled and
        nothing reads it. ``start_*`` are dynamics, not appearance, and live in ``sample_states``.
        The flat layout is alphabetical (``gymnasium.spaces.Dict`` sorts); see ``style_layout``."""
        return _make_variation_space()

    # ---- states ---------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``(batch, 13)`` in the obs layout, covering the env's OWN visited distribution under a random
        policy as the 200-state corpus measured it (``reference/meta.json``): heights 1.9-489 m (median 311),
        lateral drift to 190 m, v_z from -74 to +62 m/s, |w| to 4.9 rad/s, attitudes tumbling (tilt up to
        174 deg) with a near-upright minority. Half the draws are the high-altitude "rocket on white" regime
        the far plane makes, half the log-uniform descent to the pad."""
        rng = rng or self.rng
        B = int(batch)
        h = np.where(
            rng.random(B) < 0.5, rng.uniform(300.0, 480.0, B), np.exp(rng.uniform(np.log(1.9), np.log(300.0), B))
        )
        ang = rng.uniform(0, 2 * np.pi, B)
        rad = rng.uniform(0.0, 1.0, B) ** 0.5 * 190.0
        rad = np.where(rng.random(B) < 0.3, rng.uniform(0.0, 3.0, B), rad)
        pos = np.stack([rad * np.cos(ang), rad * np.sin(ang), h], -1)
        q = rng.normal(size=(B, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        upright = rng.random(B) < 0.3
        tilt = quat_from_rotvec(rng.normal(size=(B, 3)) * 0.08)
        yaw = quat_from_euler(np.stack([np.zeros(B), np.zeros(B), rng.uniform(-np.pi, np.pi, B)], -1))
        q = np.where(upright[:, None], quat_mul(yaw, tilt), q)
        v_world = np.stack([rng.normal(0, 12, B), rng.normal(0, 12, B), rng.uniform(-70.0, 60.0, B)], -1)
        w_world = rng.normal(size=(B, 3)) * rng.uniform(0.02, 2.5, B)[:, None]
        Rt_ = np.transpose(quat_to_mat(q), (0, 2, 1))  # world -> body
        v_body = np.einsum("bij,bj->bi", Rt_, v_world)
        w_body = np.einsum("bij,bj->bi", Rt_, w_world)
        return np.concatenate([w_body, q, v_body, pos], -1).astype(np.float32)

    # ---- render ---------------------------------------------------------------

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,13)`` -> ``(...,H,W,3)`` uint8 on device. Style may differ per environment.

        THESE ARE NOT THE OFFICIAL PIXELS. See ``visual_fidelity`` for what they are instead.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)
        c = self._nvd(size, ss)
        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        # ONE device-to-host read each for the state and the style; the kinematics, the camera and the
        # colour table are a few thousand float64 ops on the host per row, as in mujoco_locomotion.py.
        st = s.detach().cpu().numpy().astype(np.float64)
        mat = torch.as_tensor(
            material_table(sty.values.detach().cpu().numpy()), dtype=torch.float32, device=self.device
        )
        xp, xm = geom_poses(st)
        _, _, pos_prev, q_prev, _, _ = lagged_pose(st)
        eye, cmat = camera_pose(pos_prev, q_prev)
        xp_t = torch.as_tensor(xp, dtype=torch.float32, device=self.device)
        xm_t = torch.as_tensor(xm, dtype=torch.float32, device=self.device)
        step = -(-B // max(1, -(-B // c.chunk)))
        for lo in range(0, B, step):
            hi = min(lo + step, B)
            cam = ND.Camera(eye[lo:hi], cmat[lo:hi], CAM_FOV, CAM_NEAR, CAM_FAR)
            buf[lo:hi] = c.br.render(xp_t[lo:hi], xm_t[lo:hi], cam, mat_rgb=mat[lo:hi])
        return buf.view(*lead, size, size, 3) if lead else buf

    # ---- reference bridge -------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 480):
        """The PyFlyt rocket-landing env at ``size`` (its own ``render_resolution``), reset once and never closed."""
        return _reference_make(int(size))

    #: Per-row flags of the LAST ``reference_render`` call: True where the bridge could not produce the
    #: lagged picture (contact band, see ``_reference_frame_flagged``) and returned the synced picture
    #: instead.
    #:
    #: **HAZARD, and why it exists anyway.** This is mutable class state, and mutable class state set
    #: by one call and read by the next is exactly the pattern this repo distrusts: it is last-call-wins,
    #: it is shared by every instance, and two threads or two interleaved harnesses would read each
    #: other's flags. Prefer ``reference_render(..., return_flags=True)``, which returns
    #: ``(frames, flags)`` and has none of those properties -- that is the supported path and it is what
    #: the measurement harness uses.
    #:
    #: The attribute survives because ``check_visual_fidelity_under_variation`` in
    #: ``stable_envs/testing/conformance.py`` calls ``cls.reference_render(ref, states, style=style)``
    #: with no way to ask for the flags, and ``conformance.py`` is shared and not this workstream's to
    #: change. So the variation path reads the attribute immediately after the call and nothing else
    #: does. Carryover: give the shared checks a way to receive per-row exclusions, and delete
    #: this attribute.
    last_reference_unreachable: np.ndarray = np.zeros(0, bool)

    @classmethod
    def reference_render(cls, env, states, style=None, return_flags: bool = False):
        """Official frames for ``(state, style)`` rows -- the LAGGED picture after ``step``.

        Rows the bridge cannot reach (the contact band, ``REFERENCE_UNREACHABLE_SHARE`` of the sampled
        distribution) get the SYNCED picture instead and are flagged. With ``return_flags`` the
        return is ``(frames, flags)``; without it the return is the frame stack alone -- the shape the
        shared conformance checks expect -- and the flags are also left in
        ``last_reference_unreachable``, whose hazard is documented above.

        ``return_flags`` is keyword-only in practice and defaulted off, so
        ``conformance.reference_accepts_style`` (which inspects this signature for a ``style``
        parameter) and every existing caller are unaffected.
        """
        st = np.asarray(states, np.float64).reshape(-1, 13)
        if style is None:
            sty = np.tile(_default_style_row(), (len(st), 1))
        else:
            sty = np.asarray(style, np.float64).reshape(len(st), -1)
        out = [_reference_frame_flagged(env, st[i], sty[i]) for i in range(len(st))]
        frames = np.stack([fr for fr, _ in out])
        flags = np.array([u for _, u in out], bool)
        cls.last_reference_unreachable = flags
        return (frames, flags) if return_flags else frames


def _default_style_row() -> np.ndarray:
    """The official appearance as a style row IN THE ENV LAYOUT, without constructing an env."""
    return np.asarray(_make_variation_space().pack(), np.float64)


if missing_backend() is None:
    register(PyFlytRocketLanding)
