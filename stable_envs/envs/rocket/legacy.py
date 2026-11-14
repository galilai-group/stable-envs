from __future__ import annotations

import functools
import math
import os

import numpy as np
import torch

from ... import tuning
from ...env import Env
from ...registry import register
from ...spaces import Dict, Discrete, RGBBox

try:
    import nvdiffrast.torch as _nvdr

    _ = _nvdr.RasterizeCudaContext
    NVDIFF_IMPORT_ERROR: str | None = None
except Exception as _e:  # pragma: no cover
    NVDIFF_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"

from ...render import _resources as _RES
from ...render import nvdiff as ND  # noqa: E402

# PyFlyt advances physics at 240 Hz; the camera lags by one substep.
DT = 1.0 / 240.0
# Visual primitives in draw order, with URDF joint frames and full box dimensions.
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
# TinyRenderer culls back faces, so the one-sided ground is not drawn from below.
GROUND_CULL_DROP = 1.0e4
# The ignition transition changes the flame color and leaves it changed.
FLAME_RGBA = (1.0, 0.8, 0.0, 0.1)
IGNITION_RGBA = (1.0, 0.3, 0.3, 0.8)
FUELTANK_RGBA = (1.0, 1.0, 1.0, 1.0)  # PyBullet's default for a link with no <visual>
# Camera offset is link-local; pitch changes the up vector while the target stays on the link.
CAM_OFFSET = np.array([-1.0, 0.0, 3.0])
CAM_PITCH_DEG = 30.0
CAM_FOV, CAM_NEAR, CAM_FAR = 90.0, 0.1, 255.0
# Bullet btShapeHull support directions at highres=0, in source order.
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
# Per-world memory estimates bound renderer chunk allocation.
NVD_BYTES_PER_PIXEL, NVD_BYTES_PER_VERTEX = 239, 69


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
    """Convert (B, 13) observation rows to current and previous poses and world velocities.

    Bullet's cached link frames, and so the camera, lag the base by one physics
    substep. Return float64 (pos_t, q_t, pos_prev, q_prev, v_world, w_world).
    """
    s = np.asarray(state, np.float64).reshape(-1, 13)
    # Normalize in float64 to match Bullet before computing either current or lagged rotations.
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
    """Return camera eye (B, 3) and right/up/back axes (B, 3, 3) from the lagged link pose.

    The camera offset rotates with the link. Its up vector uses the link Euler
    angles with CAM_PITCH_DEG added to pitch; the target is the link origin.
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


def tiny_cylinder(length: float, radius: float):
    """Bullet's hull of the 32-step cylinder as TinyRenderer meshes it: vertices, unit normals, outward triangles.

    The hull is ``btShapeHull::buildHull`` over the support points at the 42 unit-sphere directions, with Bullet's
    ``HullLibrary`` triangles. Normals are the normalized vertex positions after TinyRenderer's float32
    renormalization.
    """
    from .tiny_raster import bullet_cylinder

    vert, norm, face = bullet_cylinder(float(length), float(radius))
    return vert.copy(), norm.astype(np.float64), face.copy()


def build_scene_arrays():
    """The 13 geoms as TinyRenderer meshes them, each in its own frame, with the ground UVs (``tiny_raster``)."""
    from .tiny_raster import OBJECTS, object_mesh

    V, N, F, VG, UV = [], [], [], [], []
    nv = 0
    for g, (_, _, _, spec, _) in enumerate(OBJECTS):
        v, n, f, uv = object_mesh(spec)
        V.append(v)
        N.append(np.asarray(n, np.float64))
        F.append(f + nv)
        VG.append(np.full(len(v), g, np.int64))
        UV.append(uv)
        nv += len(v)
    return (np.concatenate(V), np.concatenate(N), np.concatenate(F), np.concatenate(VG), np.concatenate(UV))


def link_local_frames():
    """``(pos (11,3), quat (11,4))`` of each rocket link relative to the base, from the URDF joint origins."""
    pos = np.array([p for _, _, p, _ in ROCKET_LINKS], np.float64)
    q = quat_from_euler(np.array([r for _, _, _, r in ROCKET_LINKS], np.float64))
    return pos, q


def cull_ground(xpos: np.ndarray, xmat: np.ndarray, eye: np.ndarray) -> None:
    """Move the ground past the far plane in each world whose camera ``eye`` is behind it."""
    normal = xmat[:, 0, :, 2]
    behind = np.einsum("bi,bi->b", eye - xpos[:, 0], normal) < 0
    xpos[behind, 0] -= GROUND_CULL_DROP * normal[behind]


def drawn_poses(state: np.ndarray):
    """``(B,13)`` -> ``xpos (B,13,3)``, ``xmat (B,13,3,3)`` as TinyRenderer draws them.

    ``getCameraImage`` places link visuals by forward kinematics from the base at *t*,
    so the base and every link share pose *t*. Only the camera uses the lagged link frame.
    """
    pos, q, _, _, _, _ = lagged_pose(state)
    B = len(pos)
    lp, lq = link_local_frames()
    xpos = np.zeros((B, N_GEOMS, 3))
    xq = np.zeros((B, N_GEOMS, 4))
    xq[:, :, 3] = 1.0
    xpos[:, 1], xq[:, 1] = pos, q
    R = quat_to_mat(q)
    for i in range(1, len(ROCKET_LINKS)):
        xpos[:, 1 + i] = pos + np.einsum("bij,j->bi", R, lp[i])
        xq[:, 1 + i] = quat_mul(q, np.broadcast_to(lq[i], (B, 4)))
    xpos[:, N_GEOMS - 1] = PAD_POS
    return xpos, quat_to_mat(xq)


# Saved corpus colors use a fixed order; environment style rows use variation-space path order.
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
    return dict(_style_layout())


@functools.cache
def _style_layout() -> tuple:
    return tuple((p, (o, w)) for p, o, w in _make_variation_space().flat_layout())


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
    """Map (B, style_dim) appearance rows to (B, 13, 3) material colors.

    Rocket colors use value/255; pad colors use the URDF three-decimal rounding.
    The flame ignition bit selects its material, and the textured ground stays white.
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
    """Load the checker texture as vertically flipped (H, W, 3) uint8.

    TinyRenderer indexes v from the bottom; nvdiffrast indexes from row zero.
    """
    import pybullet_data
    from PIL import Image

    path = os.path.join(pybullet_data.getDataPath(), "checker_blue.png")
    a = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    return np.ascontiguousarray(a[::-1])


def _optional_import_errors() -> dict:
    """Find missing optional render packages without importing CUDA backends."""
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
    """Return the missing render dependency explanation, or None.

    Rendering requires nvdiffrast, SciPy, pybullet_data, and Pillow.
    """
    parts = []
    if NVDIFF_IMPORT_ERROR is not None:
        parts.append(f"nvdiffrast: {NVDIFF_IMPORT_ERROR}")
    parts += [f"{k}: {v}" for k, v in _optional_import_errors().items()]
    return "; ".join(parts) if parts else None


_REF_CACHE: dict = {}


def _reference_make(size: int):
    """Construct the reference Rocket task; tests supply the certification factory."""
    hit = _REF_CACHE.get(size)
    if hit is not None:
        return hit
    raise LookupError(
        "PyFlytRocketLanding has no registered gymnasium id; "
        "tests.helpers.reference.certification installs the reference factory"
    )


def _reference_apply_style(p, c, rid, pad, style_row: np.ndarray) -> None:
    """Apply a style row in environment layout (``style_layout``) to the reference bodies."""
    mat = material_table(style_row)[0]
    p.changeVisualShape(rid, -1, rgbaColor=[*mat[1], 1.0], physicsClientId=c)
    for link, g in ((1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)):
        p.changeVisualShape(rid, link, rgbaColor=[*mat[g], 1.0], physicsClientId=c)
    o, _ = style_layout()["flame.ignited"]
    ignited = float(np.asarray(style_row).reshape(-1)[o]) > 0.5
    p.changeVisualShape(rid, 9, rgbaColor=list(IGNITION_RGBA if ignited else FLAME_RGBA), physicsClientId=c)
    p.changeVisualShape(pad, -1, rgbaColor=[*mat[12], 1.0], physicsClientId=c)


# Sparse matrix multiplication on CPU can change rounding with batch width.
DOCUMENTED_LIMITATIONS = (
    "nvdiff's CPU transform is batch-size invariant to within one float32 ulp at every width, and "
    "bitwise at every width on arm64; on x86-64 the clip positions move by one ulp at widths 1-6 and "
    "the normals at widths 4 and 8. On CUDA it is bitwise invariant at every width. "
    "No frame this cell produces is affected: device='cpu' raises."
)


# Contact correction can prevent the reference velocity fixed point from reaching a pose.
# Flag those rows for exclusion instead of silently comparing synchronized frames.
PAD_TOP_Z = float(PAD_POS[2] + PAD_LEN / 2.0)
CONTACT_BAND_MARGIN = 0.05
REFERENCE_UNREACHABLE_SHARE = 1.0 / 960.0


def _lowest_link_z(p, c, rid) -> float:
    """The lowest z of the rocket's link collision AABBs at its current pose in the client."""
    n = p.getNumJoints(rid, physicsClientId=c)
    return min(p.getAABB(rid, link, physicsClientId=c)[0][2] for link in range(-1, n))


def _reference_frame_flagged(env, state: np.ndarray, style_row: np.ndarray, contact_fallback: bool = True):
    """Return (frame, unreachable) using the previous-substep velocity fixed point.

    When contact prevents reconstruction, contact_fallback returns a synchronized
    frame flagged for exclusion. Other reconstruction failures raise.
    """
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
    # Compare quaternion rotations without depending on their sign.
    pos_err = float(np.abs(got_pos - pos).max())
    rot_err = float(1.0 - abs(np.dot(got_q / np.linalg.norm(got_q), q)))
    if pos_err > 1e-6 or rot_err > 1e-9:
        # Check whether the requested pose itself lies in the contact band.
        p.resetBasePositionAndOrientation(rid, list(pos), list(q), physicsClientId=c)
        p.resetBaseVelocity(rid, list(v), list(w), physicsClientId=c)
        in_band = _lowest_link_z(p, c, rid) <= PAD_TOP_Z + CONTACT_BAND_MARGIN
        if contact_fallback and in_band:
            return np.asarray(u.render(), np.uint8)[..., :3].copy(), True
        raise AssertionError(
            f"PyFlytRocketLanding: the reference did not reach the requested pose (pos {pos_err:.3e} m, "
            f"1-|q.q'| {rot_err:.3e}; {'in' if in_band else 'outside'} the contact band)"
        )
    return np.asarray(u.render(), np.uint8)[..., :3].copy(), False


def nvd_renderer(size, ss, device, scene_type=ND.Scene, renderer_type=ND.BatchRenderer):
    """Build the textured-triangle renderer for one size and supersample; return it with its bytes per world."""
    vert, norm, face, vg, uv = build_scene_arrays()
    rgba = np.ones((N_GEOMS, 4))
    rgba[GEOM_NAMES.index("flame")] = FLAME_RGBA
    tex_of_geom = np.full(N_GEOMS, -1, np.int64)
    tex_of_geom[GEOM_NAMES.index("ground")] = 0
    scene = scene_type.from_triangles(
        vert,
        norm,
        face,
        vg,
        rgba,
        uv=uv,
        textures={0: checker_texture()},
        tex_of_geom=tex_of_geom,
        device=str(device),
        tex_policy="none",
    )
    renderer = renderer_type(
        None,
        res=size,
        device=str(device),
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
    renderer.owns_scene = True
    renderer.background = np.array([1.0, 1.0, 1.0])  # TinyRenderer's clear color
    return renderer, NVD_BYTES_PER_PIXEL * (size * ss) ** 2 + NVD_BYTES_PER_VERTEX * scene.V


def nvd_worlds(per_world: int, device, size: int, batch: int) -> int:
    """Return the worlds one renderer chunk holds within the ``nvd_budget_bytes`` knob."""
    budget = tuning.get("nvd_budget_bytes", path="rocket-nvdiff", size=size, batch=batch, device=device)
    return max(1, int(budget // per_world))


class _Nvd:
    __slots__ = ("br", "size", "ss", "per_world")


_NVD_CACHE = _RES.BoundedCache(8, name="rocket.nvd", on_evict=ND.close_cached_renderer)


class PyFlytRocketLanding(Env):
    """Rocket geometry renderer over observation rows obs[:13].

    This base renders stored poses with CUDA. The registered semantic subclass
    adds simulation and CPU rendering. Silhouette pixels can differ from official
    TinyRenderer pixels; see visual_fidelity.
    """

    family = "pyflyt"
    domain = "rocket_landing"
    requires_cuda = True

    name = "pf-rocket-landing"
    reference_id = None
    # Default reference resolution is 480 × 480 RGB.
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
    visual_fidelity = {
        "mse_mean": 0.0091,
        "mse_p95": 0.0772,
        "mse_max": 0.2354,
        "reference_unreachable_share": REFERENCE_UNREACHABLE_SHARE,
        "measured_at": (
            "480 px, supersample 1, default appearance, per-frame MSE over pixels and channels on the 0-255 "
            "scale, over the rows of sample_render_rows(default_rng(20260811)) (64 rows) that the reference "
            "bridge reaches, against PyFlyt 0.29.0 / pybullet 3.2.7 TinyRenderer frames of the rocket-landing "
            "env with the camera at the lagged link pose; NVIDIA L40S / driver 595.91.07, "
            "torch 2.14.1+cu126, nvdiffrast 0.4.0"
        ),
        "reference_versions": {"PyFlyt": "0.29.0", "pybullet": "3.2.7"},
        "measured_with": {"nvdiffrast": "0.4.0", "torch": "2.14.1+cu126"},
    }
    # TinyRenderer's lighting multiplies material color, so literal RGB probes can be unreachable.
    # The exception is restricted to the declared lit leaves; other color checks still apply.
    color_exactness_applicability = {
        "leaves": ("pad.color", "rocket.body_color", "rocket.booster_color", "rocket.fin_color", "rocket.leg_color"),
        "status": "not_applicable: literal RGB equality is unreachable under the reference's own law",
        "mechanism": "bullet3 TinyRenderer's multiplicative one-light law "
        "out_i = int(255*c_i*(A + D*max(0,n.l) + S*max(0,r_z)**spec)*L_i), truncated, "
        "A=0.6 D=0.35 S=0.05 spec=2, light (-50,30,100) normalised, no shadow, no gamma",
        "max_shading_factor": 0.993567,
        "reachable_ceiling_for_the_probe_targets": {"200": 198, "33": 32, "17": 16},
        "official_renderer_fails_the_same_check": True,
        "official_renderer_evidence": "the official PyFlyt 0.29.0 / pybullet 3.2.7 frame's brightest "
        "body pixel is 0.991*int(255*c); the requested triple is absent "
        "from the official frames too",
        "evidence": ("tests/fixtures/evidence/rocket_colour/colour_law_ceiling.json",),
        "measured_against": "PyFlyt 0.29.0, pybullet 3.2.7 (bullet3 TinyRenderer.cpp Shader::fragment)",
        "scope_note": "scoped to these five lit leaves only; any other RGB leaf on this env stays "
        "judged against literal equality",
        "successor": "not implemented: compare this renderer's response to the knob against the "
        "official renderer's response to the same knob",
    }
    visual_note = (
        "A different rasterizer (nvdiffrast) drawing TinyRenderer's scene under TinyRenderer's law. "
        "Residual: silhouette pixels (different edge rules and float paths); flat interiors match. "
        "Under sampled appearance, some lit colours differ by one level of rounding. " + DOCUMENTED_LIMITATIONS
    )

    # The base packs observation geometry only; the semantic subclass owns complete simulator state.
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
        """Construct the CUDA geometry renderer; reject CPU before allocating a context."""
        if missing_backend() is not None:  # pragma: no cover
            raise ImportError(f"PyFlytRocketLanding cannot render in this interpreter: {missing_backend()}")
        device = Env.resolve_device(device)
        if str(device).startswith("cpu"):
            raise NotImplementedError(
                "PyFlytRocketLanding has no CPU backend: it needs a CPU textured-triangle rasteriser "
                "(cpu3d/raycast3d cast MuJoCo primitives only). A CPU request is never served by CUDA."
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
        """Retain shared renderer contexts for other environments using the same cache key."""
        return None

    def _nvd(self, size: int | None = None, ss: int | None = None) -> _Nvd:
        size = int(self.size if size is None else size)
        ss = int(self.ss if ss is None else ss)
        key = (size, ss, str(self.device))
        hit = _NVD_CACHE.get(key)
        if hit is not None:
            return hit
        c = _Nvd()
        c.size, c.ss = size, ss
        c.br, c.per_world = nvd_renderer(size, ss, self.device)
        _NVD_CACHE.put(key, c)
        return c

    def make_variation_space(self) -> Dict:
        """Return body, fin, leg, booster, pad colors and the flame ignition bit.

        Defaults match reset appearance. Dynamics parameters are not render variation.
        """
        return _make_variation_space()

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample (batch, 13) observation rows with body-frame velocities.

        Half the samples cover high-altitude flight; half cover log-uniform descent.
        These are geometry samples, not complete simulator snapshots.
        """
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

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """Render (..., 13) pose rows and per-row appearance to uint8 RGB.

        Frames can differ from the upstream renderer; see ``visual_note``.
        """
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = int(s.shape[0])
        size = int(size or self.size)
        tuning.on_render("rocket-nvdiff", size, B, self)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)
        c = self._nvd(size, ss)
        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        # Read pose and appearance once per batch for float64 host geometry preparation.
        st = s.detach().cpu().numpy().astype(np.float64)
        mat = torch.as_tensor(
            material_table(sty.values.detach().cpu().numpy()), dtype=torch.float32, device=self.device
        )
        xp, xm = drawn_poses(st)
        _, _, pos_prev, q_prev, _, _ = lagged_pose(st)
        eye, cmat = camera_pose(pos_prev, q_prev)
        cull_ground(xp, xm, eye)
        xp_t = torch.as_tensor(xp, dtype=torch.float32, device=self.device)
        xm_t = torch.as_tensor(xm, dtype=torch.float32, device=self.device)
        chunk = nvd_worlds(c.per_world, self.device, size, B)
        step = -(-B // max(1, -(-B // chunk)))
        for lo in range(0, B, step):
            hi = min(lo + step, B)
            cam = ND.Camera(eye[lo:hi], cmat[lo:hi], CAM_FOV, CAM_NEAR, CAM_FAR)
            buf[lo:hi] = c.br.render(xp_t[lo:hi], xm_t[lo:hi], cam, mat_rgb=mat[lo:hi])
        return buf.view(*lead, size, size, 3) if lead else buf

    @classmethod
    def reference_env(cls, size: int = 480):
        """Construct and reset a reference environment; the caller owns its lifetime."""
        return _reference_make(int(size))

    # Last-call flags are shared across instances. Use return_flags=True for concurrent reference calls.
    last_reference_unreachable: np.ndarray = np.zeros(0, bool)

    @classmethod
    def reference_render(cls, env, states, style=None, return_flags: bool = False):
        """Render lagged upstream frames; optionally return per-row exclusion flags.

        Contact-band reconstruction failures return synchronized frames with True
        flags. The same flags are stored in last_reference_unreachable, which is
        shared mutable class state; prefer return_flags=True for concurrent callers.
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
    """Return the default appearance in environment style order without constructing an env."""
    return np.asarray(_make_variation_space().pack(), np.float64)


if missing_backend() is None:
    register(PyFlytRocketLanding)
