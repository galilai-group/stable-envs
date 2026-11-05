from __future__ import annotations

import numpy as np

from ..._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from ..._core.plate_texels import quantise as _quantise
from ...render import _resources as _RES
from ...render import mjwarp as MW

# mjvLight type code for directional lights.
LIGHT_DIRECTIONAL = 1

# Effective specular scale fitted to the reference cartpole face.
SPECULAR_SCALE = 0.988

# Default MuJoCo geom/site visibility groups.
VISIBLE_GROUPS = (0, 1, 2)
# Hide movers from plate bakes by assigning them to this disabled group.
HIDDEN_GROUP = 5

# Model arrays inspected for per-episode renderer changes.
EPISODE_ARRAYS = (
    "geom_pos",
    "geom_quat",
    "geom_size",
    "site_pos",
    "site_quat",
    "site_size",
    "body_pos",
    "body_quat",
    "light_pos",
    "light_dir",
)
# A fixed reset count makes the sampled census reproducible.
N_DRAWS = 6
_COMPONENT = {3: ("x", "y", "z"), 4: ("qw", "qx", "qy", "qz")}
_SCENE_CACHE_MAX = 96
_SCENE_CACHE = _RES.BoundedCache(_SCENE_CACHE_MAX, name="dmcontrol.scene")
# (width, height, channels) -> checker fill masks, excluding marks.
_GRID_MASKS: dict = {}
# (domain, task, size) -> grid slots with model-specific texture addresses.
_GRID_SLOTS: dict = {}
# (domain, task, size) -> env, model, data, pristine arrays, renderer, episode slots.
# These reference renderers stay alive for the process lifetime.
_ENV_CACHE: dict = {}
# Captured GPU contexts are bounded and released on eviction.
_TC_CACHE = _RES.BoundedCache(3, name="dmcontrol.tc", on_evict=MW.release_cached_ctx)


def _check_dm_control_works() -> None:
    """Compile an empty model at import to report incompatible dependency pairs before registration."""
    from dm_control import mujoco as dm_mujoco

    try:
        dm_mujoco.Physics.from_xml_string("<mujoco/>")
    except AttributeError as e:
        import mujoco

        raise ImportError(
            f"dm_control is installed but cannot compile a model with mujoco "
            f"{mujoco.__version__}: {e}. This is a version skew, not a missing package. Supported "
            f"pair: mujoco==3.17.0 with dm_control==1.0.50."
        ) from e


_check_dm_control_works()


def _suite():
    from dm_control import suite

    return suite


def _mat_rgb(m, matid: int, rgba) -> np.ndarray:
    """Resolve material RGB when element RGBA matches the compiler default; otherwise use its RGB."""
    rgba = np.asarray(rgba, np.float64)
    if matid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[matid], np.float64)[:3]
    return rgba[:3]


def _mat_spec_shin(m, matid: int) -> tuple[float, float]:
    if matid < 0:
        return 0.5, 0.5  # MuJoCo's material defaults
    return float(m.mat_specular[matid]), float(m.mat_shininess[matid])


def _grid_defaults() -> np.ndarray:
    """Read rgb1, rgb2, and markrgb from the installed suite materials XML."""
    import re

    from dm_control.suite import common

    xml = common.ASSETS["./common/materials.xml"].decode()
    mat = re.search(r'<texture[^>]*name="grid"[^>]*/>', xml)
    assert mat, "dm_control's common/materials.xml does not declare a texture named 'grid'"
    out = []
    for attr in ("rgb1", "rgb2", "markrgb"):
        got = re.search(attr + r'="([^"]+)"', mat.group(0))
        assert got, f"the suite's grid texture does not declare {attr}"
        out.append([float(x) for x in got.group(1).split()])
    return np.asarray(out, np.float64)


def _grid_masks(m, texid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fill-color masks for the suite checker, excluding edge marks of the same color.

    Compile distinct probe colors and require the masks to reconstruct the real
    default texture. Cache by texture dimensions/channel count for the shared builtin.
    """
    import mujoco

    key = (
        int(m.tex_width[texid]),
        int(m.tex_height[texid]),
        int(m.tex_nchannel[texid]) if hasattr(m, "tex_nchannel") else 3,
    )
    hit = _GRID_MASKS.get(key)
    if hit is not None:
        return hit
    w, h, nch = key
    probe = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><texture name="grid" type="2d" builtin="checker" rgb1="1 0 0" '
        f'rgb2="0 1 0" width="{w}" height="{h}" mark="edge" markrgb="0 0 1"/>'
        f'<material name="grid" texture="grid"/></asset><worldbody>'
        f'<geom type="plane" size="1 1 1" material="grid"/></worldbody></mujoco>'
    )
    pt = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
    pa = int(probe.tex_adr[pt])
    pn = w * h * (int(probe.tex_nchannel[pt]) if hasattr(probe, "tex_nchannel") else 3)
    pf = np.asarray(probe.tex_data[pa : pa + pn]).reshape(w * h, -1)[:, :3]
    m1 = np.all(pf == np.array([255, 0, 0], np.uint8)[None, :], axis=1)
    m2 = np.all(pf == np.array([0, 255, 0], np.uint8)[None, :], axis=1)
    mk = np.all(pf == np.array([0, 0, 255], np.uint8)[None, :], axis=1)
    assert (m1 | m2 | mk).all(), (
        "the probe checker compiled to a colour that is none of its three declared ones, so the "
        "masks do not describe MuJoCo's builtin and must not be used"
    )
    # Require the probe masks and declared colors to reproduce the compiled default texture.
    d = _grid_defaults()
    adr = int(m.tex_adr[texid])
    n = w * h * nch
    real = np.asarray(m.tex_data[adr : adr + n]).reshape(-1, nch)[:, :3]
    rebuilt = np.empty_like(real)
    rebuilt[m1] = _quantise(d[0])
    rebuilt[m2] = _quantise(d[1])
    rebuilt[mk] = _quantise(d[2])
    assert np.array_equal(rebuilt, real), (
        f"the probe masks plus the suite's declared grid colours rebuild {int((rebuilt != real).any(1).sum())} "
        f"of {w * h} texels differently from the compiled texture; a substitution through these "
        f"masks would paint the wrong texels"
    )
    _GRID_MASKS[key] = (m1, m2)
    return _GRID_MASKS[key]


def _tex_rgb(m, matid: int) -> np.ndarray:
    """Return a flat texture color, average of nonuniform texels, or ones when untextured.

    This is not spatial texture sampling.
    """
    if matid < 0:
        return np.ones(3)
    tid = m.mat_texid[matid]
    tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
    if tid < 0:
        return np.ones(3)
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    a0 = int(m.tex_adr[tid])
    h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
    reg = np.asarray(getattr(m, attr)[a0 : a0 + 3 * h * w], np.float64).reshape(-1, 3) / 255.0
    u = np.unique(reg, axis=0)
    return u[0] if len(u) == 1 else reg.mean(0)


def _row_width(m, arr: str) -> int:
    a = np.asarray(getattr(m, arr))
    return int(a.reshape(a.shape[0], -1).shape[1]) if a.ndim > 1 else 1


def _slot_element(m, arr: str, i: int) -> tuple[str, int]:
    """Return the element kind and ID owning a flat model-array component."""
    return arr.split("_")[0], int(i // _row_width(m, arr))


def _slot_name(m, arr: str, i: int) -> str:
    import mujoco

    kind, eid = _slot_element(m, arr, i)
    obj = {
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "light": mujoco.mjtObj.mjOBJ_LIGHT,
    }[kind]
    name = mujoco.mj_id2name(m, obj, eid) or f"{kind}{eid}"
    w = _row_width(m, arr)
    comp = _COMPONENT.get(w, tuple(str(k) for k in range(w)))[int(i % w)]
    return f"{name}_{arr.split('_', 1)[1]}_{comp}"


def _episode_slots(env, m) -> tuple[tuple[str, int], ...]:
    """Find model-array components that differ across a fixed number of resets.

    Leave the environment at the final draw, including constant reset-time edits.
    Reject varying sizes and light directions because those are shared per bake.
    Position and quaternion components flow through per-row forward kinematics.
    This sampled census is limited to EPISODE_ARRAYS.
    """
    have = [k for k in EPISODE_ARRAYS if hasattr(m, k)]
    draws = []
    for _ in range(N_DRAWS):
        env.reset()
        draws.append({k: np.array(getattr(m, k), copy=True).reshape(-1) for k in have})
    slots: list[tuple[str, int]] = []
    for k in have:
        stack = np.stack([dr[k] for dr in draws]).astype(np.float64)
        for i in np.flatnonzero(stack.max(0) - stack.min(0) > 0):
            slots.append((k, int(i)))
    bad = [s for s in slots if s[0].endswith(("_size", "_dir"))]
    if bad:
        # Per-row size or light-direction changes would invalidate shared bake data; reject them.
        raise RuntimeError(
            f"{env.__class__.__name__}: the episode draws {[_slot_name(m, a, i) for a, i in bad]}, "
            f"which change a primitive's size or a light's direction rather than a pose. Mover "
            f"primitives carry one size per batch and lights are captured once per bake, so these "
            f"need a per-row size/light path. Per-episode *_quat entries are supported."
        )
    return tuple(slots)
