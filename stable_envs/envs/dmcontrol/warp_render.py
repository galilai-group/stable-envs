"""MuJoCo Warp rendering of the DM Control cells on CUDA (``backend="mjwarp"``).

A context holds one render model, its MuJoCo Warp model, data and render context, and one captured frame.
The render model is the cell's model with the appearance of one look group: the field of view, the
visibility and shape leaves, and the floor colours when no reflection pass recolours the floor. Colours,
light scales, sphere radii and gravity change per world without a new context.

MuJoCo Warp draws geoms only. For the fixed-camera cells the render model adds one proxy geom for each site
in a visible group and one capsule for each segment of a drawn tendon. A kernel places them each frame.

With MuJoCo Warp physics, a live frame copies the physics poses on the device into the render data and
draws them. Otherwise each launch writes the compact rows and runs kinematics in the render data.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, fields, replace
from functools import cache

import numpy as np
import torch

from ... import tuning
from ..._core import environ as _environ
from ..._core.plate_texels import quantise as _quantise
from ..._core.warp_cuda import require_warp_cuda
from ...render import _resources as _RES
from ...render import mjwarp as MW
from ._model import HIDDEN_GROUP, VISIBLE_GROUPS, _suite

#: Values of :attr:`Options.shadows`.
SHADOWS = ("all", "one", "none")
#: Values of :attr:`Options.bvh`.
BVH_MODES = ("refit", "rebuild")
#: Most worlds per launch when neither the option nor the tuning knobs set it.
MAX_WORLDS = 256
#: Name of the body that carries the proxy geoms.
PROXY_BODY = "snv_render_proxies"
# Contexts for compact rows; contexts that read live physics belong to their environment.
_CACHE = _RES.BoundedCache(8, name="dmcontrol.warp", on_evict=MW.release_cached_ctx)
_UNSET = object()


@dataclass(frozen=True)
class Options:
    """Speed and fidelity options of the MuJoCo Warp renderer.

    - ``shadows``: ``"all"`` shadow-casting lights cast shadows, ``"one"`` only the first one,
      ``"none"`` no shadow rays.
    - ``samples``: sub-pixel samples per axis. None uses the environment's ``supersample``.
    - ``groups``: the geom groups that are drawn.
    - ``bvh``: ``"refit"`` updates the scene BVH bounds each frame, ``"rebuild"`` builds the BVH again.
    - ``graph``: replay one captured CUDA graph per launch instead of separate kernel launches.
    - ``worlds``: worlds per launch. None uses the ``dmc_launch_rays`` and ``dmc_min_worlds`` knobs.
    - ``reflection``: the floor reflection pass, which also recolours the floor per world.
    - ``sky``: the cached source sky for cameras with a fixed orientation. Otherwise MuJoCo Warp
      draws the skybox.
    - ``segmentation`` and ``depth``: keep the geom id and the ray hit distance of each pixel.
    """

    shadows: str = "all"
    samples: int | None = None
    groups: tuple[int, ...] = VISIBLE_GROUPS
    bvh: str = "refit"
    graph: bool = True
    worlds: int | None = None
    reflection: bool = True
    sky: bool = True
    segmentation: bool = False
    depth: bool = False

    @classmethod
    def parse(cls, value=None) -> Options:
        """Merge ``value`` (a mapping or Options) over ``STABLE_ENVS_DMC_WARP_OPTIONS`` and the defaults."""
        if isinstance(value, Options):
            return value
        merged = dict(_environ.read("STABLE_ENVS_DMC_WARP_OPTIONS") or {})
        merged.update(dict(value or {}))
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(merged) - names)
        if unknown:
            raise ValueError(f"unknown MuJoCo Warp render options {unknown}; the options are {sorted(names)}")
        if "groups" in merged:
            merged["groups"] = tuple(sorted({int(g) for g in merged["groups"]}))
        out = cls(**merged)
        out._check()
        return out

    def _check(self) -> None:
        if self.shadows not in SHADOWS:
            raise ValueError(f"shadows must be one of {SHADOWS}, got {self.shadows!r}")
        if self.bvh not in BVH_MODES:
            raise ValueError(f"bvh must be one of {BVH_MODES}, got {self.bvh!r}")
        for name in ("samples", "worlds"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
                raise ValueError(f"{name} must be None or an int >= 1, got {v!r}")
        if not self.groups or any(not 0 <= g < HIDDEN_GROUP for g in self.groups):
            raise ValueError(f"groups must name geom groups in [0, {HIDDEN_GROUP}), got {self.groups!r}")
        for name in ("graph", "reflection", "sky", "segmentation", "depth"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False, got {getattr(self, name)!r}")


def selected(env, backend=None) -> bool:
    """Whether ``env`` draws with this renderer.

    ``backend="mjwarp"`` selects it. ``backend="auto"`` selects it on CUDA when
    ``STABLE_ENVS_DMC_CUDA_RENDER`` is ``mjwarp``.
    """
    chosen = env.backend if backend is None else backend
    if chosen == "mjwarp":
        return True
    return chosen == "auto" and env.device.type == "cuda" and _environ.read("STABLE_ENVS_DMC_CUDA_RENDER") == "mjwarp"


# --------------------------------------------------------------------------- kernels


@cache
def _kernels():
    import warp as wp
    from mujoco_warp._src import math as mjmath

    @wp.kernel
    def welded_geoms(
        geom_bodyid: wp.array(dtype=int),
        geom_pos: wp.array2d(dtype=wp.vec3),
        geom_quat: wp.array2d(dtype=wp.quat),
        geoms: wp.array(dtype=int),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        geom_xpos: wp.array2d(dtype=wp.vec3),
        geom_xmat: wp.array2d(dtype=wp.mat33),
    ):
        # MuJoCo Warp's kinematics keeps the make_data pose of geoms welded to the world; these move per episode.
        w, k = wp.tid()
        g = geoms[k]
        b = geom_bodyid[g]
        q = xquat[w, b]
        geom_xpos[w, g] = xpos[w, b] + mjmath.rot_vec_quat(geom_pos[w % geom_pos.shape[0], g], q)
        geom_xmat[w, g] = mjmath.quat_to_mat(mjmath.mul_quat(q, geom_quat[w % geom_quat.shape[0], g]))

    @wp.kernel
    def copy_vec3(src: wp.array2d(dtype=wp.vec3), offset: wp.array(dtype=int), dst: wp.array2d(dtype=wp.vec3)):
        w, i = wp.tid()
        r = w + offset[0]
        if r < src.shape[0]:
            dst[w, i] = src[r, i]

    @wp.kernel
    def copy_mat33(src: wp.array2d(dtype=wp.mat33), offset: wp.array(dtype=int), dst: wp.array2d(dtype=wp.mat33)):
        w, i = wp.tid()
        r = w + offset[0]
        if r < src.shape[0]:
            dst[w, i] = src[r, i]

    @wp.kernel
    def world_sites(
        site_pos: wp.array2d(dtype=wp.vec3),
        sites: wp.array(dtype=int),
        offset: wp.array(dtype=int),
        site_xpos: wp.array2d(dtype=wp.vec3),
    ):
        # A site on the world body has its model position as its world position.
        w, k = wp.tid()
        r = w + offset[0]
        if r < site_pos.shape[0]:
            s = sites[k]
            site_xpos[w, s] = site_pos[r, s]

    @wp.kernel
    def place_sites(
        site_xpos: wp.array2d(dtype=wp.vec3),
        site_xmat: wp.array2d(dtype=wp.mat33),
        sites: wp.array(dtype=int),
        first: int,
        geom_xpos: wp.array2d(dtype=wp.vec3),
        geom_xmat: wp.array2d(dtype=wp.mat33),
    ):
        w, k = wp.tid()
        s = sites[k]
        geom_xpos[w, first + k] = site_xpos[w, s]
        geom_xmat[w, first + k] = site_xmat[w, s]

    @wp.func
    def cosh64(x: wp.float64) -> wp.float64:
        e = wp.exp(x)
        return wp.float64(0.5) * (e + wp.float64(1.0) / e)

    @wp.func
    def sinh64(x: wp.float64) -> wp.float64:
        e = wp.exp(x)
        return wp.float64(0.5) * (e - wp.float64(1.0) / e)

    @wp.func
    def residual(b: wp.float64, intercept: wp.float64) -> wp.float64:
        q = wp.float64(2.0) * b * sinh64(wp.float64(0.5) / b) - wp.float64(1.0)
        return wp.float64(1.0) / wp.sqrt(q) - intercept

    @wp.func
    def gradient(b: wp.float64) -> wp.float64:
        a = wp.float64(0.5) / b
        q = wp.float64(2.0) * b * sinh64(a) - wp.float64(1.0)
        return (a * cosh64(a) - sinh64(a)) * wp.pow(q, wp.float64(-1.5))

    @wp.func
    def solve(v: wp.float64, h: wp.float64, length: wp.float64) -> wp.float64:
        # MuJoCo's transformed catenary equation with its Newton and backtracking steps.
        intercept = wp.float64(1.0) / wp.sqrt(wp.sqrt(length * length - v * v) / h - wp.float64(1.0))
        b = intercept / wp.sqrt(wp.float64(24.0))
        outer = int(50)  # noqa: UP018 - a mutable Warp int variable
        inner = int(10)  # noqa: UP018 - a mutable Warp int variable
        for _i in range(outer):
            res = residual(b, intercept)
            if wp.abs(res) < wp.float64(1.0e-9):
                break
            step = -res / gradient(b)
            for _j in range(inner):
                if wp.abs(residual(b + step, intercept)) < wp.abs(res):
                    break
                step = step * wp.float64(0.5)
            b = b + step
        return b

    @wp.func
    def impedance(solimp: wp.array(dtype=wp.float64), pos: wp.float64, margin: wp.float64) -> float:
        d0 = solimp[0]
        d1 = solimp[1]
        width = solimp[2]
        mid = solimp[3]
        power = solimp[4]
        if d0 == d1 or width <= wp.float64(1.0e-15):
            return float(wp.float64(0.5) * (d0 + d1))
        x = (pos - margin) / width
        if x < wp.float64(0.0):
            x = -x
        if x >= wp.float64(1.0):
            return float(d1)
        if x <= wp.float64(0.0):
            return float(d0)
        y = x
        if power != wp.float64(1.0):
            if x <= mid:
                y = wp.pow(x, power) / wp.pow(mid, power - wp.float64(1.0))
            else:
                y = wp.float64(1.0) - wp.pow(wp.float64(1.0) - x, power) / wp.pow(
                    wp.float64(1.0) - mid, power - wp.float64(1.0)
                )
        return float(d0 + y * (d1 - d0))

    @wp.func
    def polyline(
        i: int,
        count: int,
        kind: int,
        npoints: int,
        x0: wp.vec3d,
        x1: wp.vec3d,
        up: wp.vec3d,
        ac: wp.vec3d,
        h: wp.float64,
        bh: wp.float64,
        h_off: wp.float64,
        v_off: wp.float64,
        mid: wp.vec3d,
    ) -> wp.vec3d:
        if i <= 0:
            return x0
        if i >= count:
            return x1
        if kind == 1:
            return mid
        # Interior spacing divides by the point count, as mjv_catenary does.
        hor = wp.float64(i) * h / wp.float64(npoints)
        return x0 + ac * hor + up * (bh * cosh64((hor - h_off) / bh) + v_off)

    @wp.kernel
    def place_tendon(
        site_xpos: wp.array2d(dtype=wp.vec3),
        s0: int,
        s1: int,
        gravity: wp.array(dtype=wp.vec3),
        upper: wp.float64,
        npoints: int,
        nseg: int,
        first: int,
        radius: float,
        grey: wp.vec3,
        constraint: wp.vec3,
        solimp: wp.array(dtype=wp.float64),
        margin: wp.float64,
        geom_xpos: wp.array2d(dtype=wp.vec3),
        geom_xmat: wp.array2d(dtype=wp.mat33),
        geom_size: wp.array2d(dtype=wp.vec3),
        geom_rgba: wp.array2d(dtype=wp.vec4),
    ):
        w = wp.tid()
        a = site_xpos[w, s0]
        c = site_xpos[w, s1]
        x0 = wp.vec3d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]))
        x1 = wp.vec3d(wp.float64(c[0]), wp.float64(c[1]), wp.float64(c[2]))
        zero = wp.float64(0.0)
        dist = wp.length(x0 - x1)
        up = wp.vec3d(zero, zero, zero)
        ac = wp.vec3d(zero, zero, zero)
        mid = wp.vec3d(zero, zero, zero)
        h = zero
        bh = wp.float64(1.0)
        h_off = zero
        v_off = zero
        kind = int(0)  # noqa: UP018 - a mutable Warp int variable
        count = int(1)  # noqa: UP018 - a mutable Warp int variable
        colour = grey
        if dist < upper:
            g = gravity[w]
            gu = wp.vec3d(-wp.float64(g[0]), -wp.float64(g[1]), -wp.float64(g[2]))
            n = wp.length(gu)
            if n < wp.float64(1.0e-15):
                up = wp.vec3d(wp.float64(1.0), zero, zero)
            else:
                up = gu * (wp.float64(1.0) / n)
            x01 = x1 - x0
            ac = x01 - up * (up[0] * x01[0] + up[1] * x01[1] + up[2] * x01[2])
            n = wp.length(ac)
            if n < wp.float64(1.0e-15):
                ac = wp.vec3d(zero, zero, zero)
            else:
                ac = ac * (wp.float64(1.0) / n)
            h = x01[0] * ac[0] + x01[1] * ac[1] + x01[2] * ac[2]
            v = x01[0] * up[0] + x01[1] * up[1] + x01[2] * up[2]
            if upper > wp.float64(100.0) * h:
                d_up = wp.float64(-0.5) * (wp.sqrt(upper * upper - h * h) - v)
                mid = x0 + up * d_up + ac * (h * d_up / (wp.float64(2.0) * d_up - v))
                kind = 1
                count = 2
            else:
                bh = solve(v, h, upper) * h
                h_off = wp.float64(-0.5) * (wp.log((upper + v) / (upper - v)) * bh - h)
                v_off = -cosh64(h_off / bh) * bh
                kind = 2
                count = npoints - 1
        else:
            pos = upper - dist
            if pos < margin:
                imp = impedance(solimp, pos, margin)
                colour = grey * (1.0 - imp) + constraint * imp
        even = nseg % count == 0
        per = nseg // count
        for j in range(nseg):
            k0 = int(0)  # noqa: UP018 - a mutable Warp int variable
            k1 = int(0)  # noqa: UP018 - a mutable Warp int variable
            f0 = zero
            f1 = zero
            if even:
                k0 = j // per
                k1 = k0
                f0 = wp.float64(j % per) / wp.float64(per)
                f1 = f0 + wp.float64(1.0) / wp.float64(per)
            else:
                t0 = wp.float64(j) * wp.float64(count) / wp.float64(nseg)
                t1 = wp.float64(j + 1) * wp.float64(count) / wp.float64(nseg)
                k0 = wp.min(int(wp.floor(t0)), count - 1)
                k1 = wp.min(int(wp.floor(t1)), count - 1)
                f0 = t0 - wp.float64(k0)
                f1 = t1 - wp.float64(k1)
            p0 = polyline(k0, count, kind, npoints, x0, x1, up, ac, h, bh, h_off, v_off, mid)
            p1 = polyline(k0 + 1, count, kind, npoints, x0, x1, up, ac, h, bh, h_off, v_off, mid)
            q0 = polyline(k1, count, kind, npoints, x0, x1, up, ac, h, bh, h_off, v_off, mid)
            q1 = polyline(k1 + 1, count, kind, npoints, x0, x1, up, ac, h, bh, h_off, v_off, mid)
            lo = p0 + (p1 - p0) * f0
            hi = q0 + (q1 - q0) * f1
            axis = hi - lo
            length = wp.length(axis)
            z = axis / wp.max(length, wp.float64(1.0e-15))
            # Complete the frame with the least-aligned world axis, the first one on a tie.
            ref = wp.vec3d(zero, zero, wp.float64(1.0))
            if wp.abs(z[0]) <= wp.abs(z[1]) and wp.abs(z[0]) <= wp.abs(z[2]):
                ref = wp.vec3d(wp.float64(1.0), zero, zero)
            elif wp.abs(z[1]) <= wp.abs(z[2]):
                ref = wp.vec3d(zero, wp.float64(1.0), zero)
            x = wp.cross(ref, z)
            x = x / wp.max(wp.length(x), wp.float64(1.0e-15))
            y = wp.cross(z, x)
            centre = (lo + hi) * wp.float64(0.5)
            gid = first + j
            geom_xpos[w, gid] = wp.vec3(float(centre[0]), float(centre[1]), float(centre[2]))
            geom_xmat[w, gid] = wp.mat33(
                float(x[0]),
                float(y[0]),
                float(z[0]),
                float(x[1]),
                float(y[1]),
                float(z[1]),
                float(x[2]),
                float(y[2]),
                float(z[2]),
            )
            geom_size[w, gid] = wp.vec3(radius, float(length * wp.float64(0.5)), 0.0)
            geom_rgba[w, gid] = wp.vec4(colour[0], colour[1], colour[2], 1.0)

    return {
        "copy_vec3": copy_vec3,
        "copy_mat33": copy_mat33,
        "world_sites": world_sites,
        "place_sites": place_sites,
        "welded_geoms": welded_geoms,
        "place_tendon": place_tendon,
    }


# --------------------------------------------------------------------------- render model


def _is_suite(env) -> bool:
    """Fixed-camera cells apply appearance through ``_apply_variation`` and draw sites and the tendon."""
    return hasattr(env, "_apply_variation")


def _floor_slots(env):
    slots = getattr(env, "_grid_slots", None) or getattr(env, "_floor_slots", None)
    return slots() if slots is not None else None


def _reflective_planes(m) -> list[int]:
    import mujoco

    return [
        g
        for g in range(m.ngeom)
        if int(m.geom_matid[g]) >= 0
        and float(m.mat_reflectance[int(m.geom_matid[g])]) > 0
        and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    ]


def _reflection_available() -> bool:
    try:
        from mujoco_warp._src import render as upstream
    except ImportError:
        return False
    return hasattr(upstream, "_make_sample_texture")


def _palette_by_reflection(env, opts: Options) -> bool:
    """Whether the reflection pass recolours the floor per world, so floor colours need no new context."""
    cls = type(env)
    if not opts.reflection or _floor_slots(env) is None:
        return False
    return len(_reflective_planes(cls._model())) == 1 and _reflection_available()


def _look_paths(env, opts: Options, paths) -> tuple[str, ...]:
    """Leaves that need their own render model: field of view, visibility, shapes and unrecoloured floors."""
    out = [
        p
        for p in paths
        if p == "camera.fovy_scale"
        or p.startswith("visibility.")
        or (p.endswith(".shape") and p.split(".")[0] in getattr(env, "SHAPE_KNOBS", {}))
    ]
    if "floor.color" in paths and not _palette_by_reflection(env, opts):
        out.append("floor.color")
    return tuple(out)


def _apply_look(env, m, pr, look) -> None:
    """Write the look into the shared model, as a plate bake does; trackcom cells use the canonical model."""
    if _is_suite(env):
        env._apply_variation(m, pr, look)
        return
    for arr in {a for a, _ in env._slots()}:
        getattr(m, arr)[:] = pr[arr]
    m.mat_rgba[:] = pr["mat_rgba"]
    m.light_diffuse[:] = pr["light_diffuse"]
    m.geom_group[:] = pr["geom_group"]
    m.site_group[:] = pr["site_group"]


def _welded_geoms(env, m) -> list[int]:
    """Geoms welded to the world whose pose an episode entry moves: their own pose or an ancestor body's."""
    from ._model import _slot_element

    bodies, geoms = set(), set()
    for arr, k in env._slots():
        kind, eid = _slot_element(m, arr, k)
        if kind == "body":
            bodies.add(int(eid))
        elif kind == "geom" and arr in ("geom_pos", "geom_quat"):
            geoms.add(int(eid))
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if int(m.body_weldid[b]) != 0:
            continue
        chain = set()
        while b > 0:
            chain.add(b)
            b = int(m.body_parentid[b])
        if g in geoms or chain & bodies:
            out.append(g)
    return out


def _drawn_sites(m, pr) -> list[int]:
    return [s for s in range(m.nsite) if int(pr["site_group"][s]) in VISIBLE_GROUPS]


def _source_xml(cls):
    module = getattr(_suite(), cls.DOMAIN)
    if cls.DOMAIN == "manipulator":
        return module.make_model(use_peg="peg" in cls.TASK, insert=cls.TASK.startswith("insert"))
    return module.get_model_and_assets()


# Arrays copied from the shared model into the render model: appearance, episode entries and task edits.
_ELEMENT_ARRAYS = ("geom_rgba", "geom_size", "geom_type", "geom_group", "geom_matid", "geom_pos", "geom_quat")
_BODY_ARRAYS = ("body_pos", "body_quat")
_SHARED_ARRAYS = (
    "site_rgba",
    "site_size",
    "site_type",
    "site_group",
    "site_matid",
    "site_pos",
    "site_quat",
    "mat_rgba",
    "mat_reflectance",
    "mat_texid",
    "mat_specular",
    "mat_shininess",
    "light_diffuse",
    "light_specular",
    "light_ambient",
    "light_pos",
    "light_dir",
    "light_castshadow",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "tex_data",
)


def _proxy_model(cls, m, pr, count: int):
    """Compile the cell's XML with ``count`` proxy geoms on one extra body, the last body of the tree."""
    import tempfile
    from pathlib import Path

    import mujoco

    xml, assets = _source_xml(cls)
    # MjSpec.from_string does not read <include> files from the assets dict; files on disk resolve.
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in {**assets, "model.xml": xml}.items():
            path = Path(tmp, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
        spec = mujoco.MjSpec.from_file(str(Path(tmp, "model.xml")))
        body = spec.worldbody.add_body(name=PROXY_BODY)
        for k in range(count):
            body.add_geom(
                name=f"{PROXY_BODY}_{k}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.0, 0.0],
                contype=0,
                conaffinity=0,
                group=0,
            )
        m2 = spec.compile()
    same = (
        m2.ngeom == m.ngeom + count
        and m2.nbody == m.nbody + 1
        and all(getattr(m2, n) == getattr(m, n) for n in ("nsite", "nmat", "ntex", "nlight", "ncam", "nq"))
        and np.array_equal(np.asarray(m2.geom_type[: m.ngeom]), pr["geom_type"])
        and np.array_equal(np.asarray(m2.geom_bodyid[: m.ngeom]), np.asarray(m.geom_bodyid))
        and len(m2.tex_data) == len(m.tex_data)
    )
    if not same:
        raise RuntimeError(
            f"{cls.__name__}: the suite XML with {count} proxy geoms does not compile to the cell's model "
            f"plus the proxies; the MuJoCo Warp renderer cannot map its geoms"
        )
    m2.vis.global_.offwidth = m.vis.global_.offwidth
    m2.vis.global_.offheight = m.vis.global_.offheight
    return m2


def _copy_appearance(m, m2) -> None:
    for name in _ELEMENT_ARRAYS:
        getattr(m2, name)[: m.ngeom] = getattr(m, name)
    for name in _BODY_ARRAYS:
        getattr(m2, name)[: m.nbody] = getattr(m, name)
    for name in _SHARED_ARRAYS:
        getattr(m2, name)[:] = getattr(m, name)


@dataclass
class _Proxies:
    first: int = 0
    sites: tuple[int, ...] = ()
    world_sites: tuple[int, ...] = ()
    tendon: object = None
    tendon_first: int = 0


def _render_model(env, m, pr, look, opts: Options):
    """Return the render model and its proxies for one look; the shared model carries the look already."""
    import mujoco

    from ._tendon import _find_tendon

    cls = type(env)
    sites = _drawn_sites(m, pr) if _is_suite(env) else []
    tendon = _find_tendon(m) if _is_suite(env) else None
    nseg = tendon.nseg if tendon is not None else 0
    if sites or nseg:
        m2 = _proxy_model(cls, m, pr, len(sites) + nseg)
        _copy_appearance(m, m2)
    else:
        m2 = copy.deepcopy(m)
    px = _Proxies(first=int(m.ngeom), sites=tuple(sites), tendon=tendon, tendon_first=int(m.ngeom) + len(sites))
    px.world_sites = tuple(s for s in sites if int(m.site_bodyid[s]) == 0)
    for k, s in enumerate(sites):
        g = px.first + k
        m2.geom_type[g] = m2.site_type[s]
        m2.geom_size[g] = m2.site_size[s]
        m2.geom_rgba[g] = m2.site_rgba[s]
        m2.geom_matid[g] = m2.site_matid[s]
        m2.geom_group[g] = m2.site_group[s]
    if tendon is not None:
        for j in range(nseg):
            g = px.tendon_first + j
            m2.geom_type[g] = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            m2.geom_size[g] = (tendon.radius, 0.0, 0.0)
            m2.geom_rgba[g] = (*tendon.grey, 1.0)
            m2.geom_matid[g] = int(m.tendon_matid[tendon.id])
            m2.geom_group[g] = int(m.tendon_group[tendon.id])
    if look and "floor.color" in look:
        slots = _floor_slots(env)
        flat = m2.tex_data[slots["adr"] : slots["adr"] + slots["n"]].reshape(-1, slots["nch"])
        for mask, colour in zip(slots["masks"], np.asarray(look["floor.color"], np.float64).reshape(-1, 3)):
            flat[mask, :3] = _quantise(colour)
    MW.prepare_model(m2, opts.groups)
    if opts.shadows == "one":
        casting = [i for i in range(m2.nlight) if int(m2.light_castshadow[i])]
        for i in casting[1:]:
            m2.light_castshadow[i] = 0
    return m2, px


def _fixed_view(m, camera: int) -> bool:
    """Whether the camera keeps one orientation, so one sky plate serves every frame."""
    import mujoco

    mode = int(m.cam_mode[camera])
    if mode == int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM):
        return True
    return mode == int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED) and int(m.body_weldid[int(m.cam_bodyid[camera])]) == 0


def _sky(cls, size: int) -> np.ndarray:
    """The source frame with no geom, site or tendon drawn, under the look the shared model carries."""
    from dm_control.mujoco import wrapper

    source = cls._bundle(cls.default_size)[0]
    option = wrapper.MjvOption()
    option.geomgroup[:] = 0
    option.sitegroup[:] = 0
    option.tendongroup[:] = 0
    return source.physics.render(height=size, width=size, camera_id=cls.CAMERA, scene_option=option).copy()


# --------------------------------------------------------------------------- contexts


class _Ctx:
    """One render model on the device, its render context, and the captured frame."""

    __slots__ = (
        "wp",
        "mjw",
        "wm",
        "wd",
        "rc",
        "graph",
        "nworld",
        "res",
        "opts",
        "bgra",
        "seg",
        "depth",
        "textures",
        "reflection",
        "proxies",
        "needs_com",
        "views",
        "gravity",
        "solimp",
        "offset",
        "offset_t",
        "sources",
        "site_pos",
        "names",
        "mats",
        "masses_varied",
        "floor_tilted",
        "static_rows",
        "wp_device",
        "site_ids",
        "world_site_ids",
        "tendon_sites",
        "tendon_margin",
        "keymap",
        "light_base",
        "colors",
        "spheres",
        "welded_ids",
    )


def _view(c, owner, name):
    key = f"{owner}.{name}"
    v = c.views.get(key)
    if v is None:
        v = c.views[key] = c.wp.to_torch(getattr(c.wm if owner == "m" else c.wd, name))
    return v


def _build(env, size: int, nworld: int, look, opts: Options, spp: int, batch=None) -> _Ctx:
    """Build a context for one look. ``batch`` (a ``WarpBatch``) makes it read live physics poses."""
    import mujoco
    import mujoco_warp as mjw

    from ._mass_rows import ARRAYS as _MASS_ARRAYS

    wp = require_warp_cuda("DM Control MuJoCo Warp rendering")
    cls = type(env)
    _, m, _, pr, _, slots = cls._bundle(cls.default_size)
    _apply_look(env, m, pr, look)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(size))
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(size))
    m2, px = _render_model(env, m, pr, look, opts)
    sky = _sky(cls, int(size)) if opts.sky and _fixed_view(m, cls.CAMERA) else None

    c = _Ctx()
    c.wp, c.mjw, c.nworld, c.res, c.opts, c.proxies = wp, mjw, int(nworld), int(size), opts, px
    c.views, c.masses_varied, c.floor_tilted = {}, False, False
    c.graph = c.textures = c.reflection = None
    c.needs_com = MW.needs_com_pos(m2)
    index = env.device.index if env.device.index is not None else torch.cuda.current_device()
    c.wp_device = f"cuda:{index}"
    mass = getattr(cls, "_mass_rows", None)
    batched = {"mat_rgba", "light_diffuse", "geom_rgba", "geom_size"} | {a for a, _ in slots}
    if mass is not None and mass() is not None:
        batched |= set(_MASS_ARRAYS)
    seg = opts.segmentation or sky is not None or opts.reflection
    with wp.ScopedDevice(c.wp_device):
        d0 = mujoco.MjData(m2)
        d0.qpos[:] = m2.qpos0
        mujoco.mj_forward(m2, d0)
        c.wm = mjw.put_model(m2, batch_sizes={k: int(nworld) for k in sorted(batched)})
        c.wd = mjw.put_data(m2, d0, nworld=int(nworld))
        k = _kernels()
        c.gravity = wp.array(np.tile(np.asarray(m2.opt.gravity, np.float32), (int(nworld), 1)), dtype=wp.vec3)
        c.offset = wp.zeros(1, dtype=int)
        c.offset_t = wp.to_torch(c.offset)
        c.site_ids = wp.array(np.asarray(px.sites, np.int32), dtype=int) if px.sites else None
        c.world_site_ids = wp.array(np.asarray(px.world_sites, np.int32), dtype=int) if px.world_sites else None
        welded = _welded_geoms(env, m)
        c.welded_ids = wp.array(np.asarray(welded, np.int32), dtype=int) if welded else None
        c.solimp, c.tendon_sites, c.tendon_margin = None, None, 0.0
        if px.tendon is not None:
            t = px.tendon.id
            adr = int(m2.tendon_adr[t])
            c.tendon_sites = (int(m2.wrap_objid[adr]), int(m2.wrap_objid[adr + 1]))
            c.tendon_margin = float(m2.tendon_margin[t])
            s = np.asarray(m2.tendon_solimp_lim[t], np.float64).copy()
            # MuJoCo's clamps of solimp before it computes an impedance (mjMINIMP, mjMAXIMP).
            s[[0, 1, 3]] = np.clip(s[[0, 1, 3]], 0.0001, 0.9999)
            s[2] = max(0.0, s[2])
            s[4] = max(1.0, s[4])
            c.solimp = wp.array(s, dtype=wp.float64)
        c.sources, c.site_pos = [], None
        if batch is not None:
            sources = [
                ("geom_xpos", wp.vec3, m.ngeom),
                ("geom_xmat", wp.mat33, m.ngeom),
                ("cam_xpos", wp.vec3, m.ncam),
            ]
            sources += [("cam_xmat", wp.mat33, m.ncam), ("light_xpos", wp.vec3, m.nlight)]
            sources += [("light_xdir", wp.vec3, m.nlight), ("site_xpos", wp.vec3, m.nsite)]
            sources += [("site_xmat", wp.mat33, m.nsite)]
            for name, dtype, count in sources:
                if count:
                    src = wp.from_torch(batch.data(name), dtype=dtype)
                    kernel = k["copy_vec3"] if dtype is wp.vec3 else k["copy_mat33"]
                    c.sources.append((kernel, src, getattr(c.wd, name), int(count)))
            if px.world_sites:
                try:
                    c.site_pos = wp.from_torch(batch.model_field("site_pos"), dtype=wp.vec3)
                except KeyError:
                    c.site_pos = None
        with MW.gl_rounding():
            c.rc = MW.create_render_context(
                m2,
                nworld=int(nworld),
                cam_res=(int(size), int(size)),
                render_rgb=True,
                render_seg=bool(seg),
                render_depth=bool(opts.depth),
                cam_active=[i == int(cls.CAMERA) for i in range(m2.ncam)],
                enabled_geom_groups=list(opts.groups),
                use_textures=True,
                use_shadows=opts.shadows != "none",
                render_skybox=sky is None,
                samples_per_pixel=int(spp),
                **dict(getattr(cls, "RENDER_FLAGS", {}) or {}),
            )
            if sky is not None:
                from .trackcom_texture import Textures

                c.textures = Textures(m2, c.rc, sky)
            if opts.reflection and len(_reflective_planes(m2)) == 1 and _reflection_available():
                from .trackcom_reflection import Reflection

                palette = _floor_slots(env) if _palette_by_reflection(env, opts) else None
                c.reflection = Reflection(c.wm, c.wd, c.rc, m2, palette)
            _describe(env, c, m2)
            _frame(c)
            wp.synchronize()
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(size), int(size), 4)
        c.seg = wp.to_torch(c.rc.seg_data).reshape(int(nworld), -1, 2) if seg else None
        c.depth = wp.to_torch(c.rc.depth_data) if opts.depth else None
        if opts.graph:
            try:
                with wp.ScopedCapture() as capture:
                    _frame(c)
                c.graph = capture.graph
            except Exception as error:  # noqa: BLE001 - a graph is a speed option; frames stay the same
                warnings.warn(
                    f"{cls.__name__}: MuJoCo Warp frame capture failed ({error}); launching kernels without a graph",
                    RuntimeWarning,
                    stacklevel=3,
                )
                c.graph = None
    return c


def _describe(env, c: _Ctx, m2) -> None:
    """Record per-world material indices, static rows and the names of the render model's geoms."""
    import mujoco

    px = c.proxies
    mats = []
    for name in env._used_materials():
        i = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if i >= 0:
            mats.append((int(i), f"material.{name}"))
    c.mats = (torch.as_tensor([i for i, _ in mats], dtype=torch.long, device=env.device), [p for _, p in mats])
    c.light_base = torch.as_tensor(env._pristine()["light_diffuse"], device=env.device, dtype=torch.float32)
    gid = lambda name: int(mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, name))  # noqa: E731
    c.colors = [(gid(name), f"{group}.color") for group, geoms in env.GEOM_COLORS.items() for name in geoms]
    c.spheres = [(gid(geom), path) for path, geom in env.SPHERE_SIZES.items()]
    c.static_rows = [
        (kind, eid, torch.as_tensor(base, device=env.device, dtype=torch.float32), comps)
        for (kind, eid), (base, comps) in env._static_slot_rows().items()
    ]
    names = [mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(px.first)]
    names += [f"site:{mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_SITE, s) or s}" for s in px.sites]
    if px.tendon is not None:
        tendon = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_TENDON, px.tendon.id) or f"tendon{px.tendon.id}"
        names += [f"tendon:{tendon}"] * px.tendon.nseg
    c.names = tuple(names)
    # Render-model geom -> (objid, objtype) of the cell's model, which is the model of reference_env.
    keymap = [(g, int(mujoco.mjtObj.mjOBJ_GEOM)) for g in range(px.first)]
    keymap += [(int(s), int(mujoco.mjtObj.mjOBJ_SITE)) for s in px.sites]
    if px.tendon is not None:
        keymap += [(int(px.tendon.id), int(mujoco.mjtObj.mjOBJ_TENDON))] * px.tendon.nseg
    c.keymap = torch.as_tensor(np.asarray(keymap, np.int64).reshape(-1, 2), device=env.device)


def _bvh(c: _Ctx) -> None:
    if c.opts.bvh == "refit":
        c.mjw.refit_bvh(c.wm, c.wd, c.rc)
        return
    from mujoco_warp._src import bvh as upstream

    rc = c.rc
    c.wp.launch(
        upstream._compute_bvh_bounds,
        dim=(c.nworld, rc.bvh_ngeom),
        inputs=[
            c.wm.geom_type,
            c.wm.geom_dataid,
            c.wm.geom_size,
            c.wd.geom_xpos,
            c.wd.geom_xmat,
            rc.bvh_ngeom + rc.bvh_nflexgeom,
            rc.enabled_geom_ids,
            rc.mesh_bounds_size,
            rc.hfield_bounds_size,
            rc.lower,
            rc.upper,
            rc.group,
        ],
    )
    # The in-place CUDA LBVH rebuild reuses its buffers, so a captured graph stays valid. A new topology
    # moves each world's root node.
    rc.bvh.rebuild("lbvh")
    c.wp.launch(upstream.compute_bvh_group_roots, dim=c.nworld, inputs=[rc.bvh.id], outputs=[rc.group_root])


def _frame(c: _Ctx) -> None:
    """Poses, proxies, BVH, the ray tracer, then the sky and reflection passes."""
    wp, k, px = c.wp, _kernels(), c.proxies
    if c.sources:
        for kernel, src, dst, count in c.sources:
            wp.launch(kernel, dim=(c.nworld, count), inputs=[src, c.offset, dst])
        if c.site_pos is not None:
            wp.launch(
                k["world_sites"],
                dim=(c.nworld, len(px.world_sites)),
                inputs=[c.site_pos, c.world_site_ids, c.offset, c.wd.site_xpos],
            )
    else:
        c.mjw.kinematics(c.wm, c.wd)
        if c.needs_com:
            # Kinematics leaves subtree_com stale; tracking cameras and lights read it.
            c.mjw.com_pos(c.wm, c.wd)
        c.mjw.camlight(c.wm, c.wd)
        if c.welded_ids is not None:
            wp.launch(
                k["welded_geoms"],
                dim=(c.nworld, c.welded_ids.shape[0]),
                inputs=[c.wm.geom_bodyid, c.wm.geom_pos, c.wm.geom_quat, c.welded_ids, c.wd.xpos, c.wd.xquat],
                outputs=[c.wd.geom_xpos, c.wd.geom_xmat],
            )
    if px.sites:
        wp.launch(
            k["place_sites"],
            dim=(c.nworld, len(px.sites)),
            inputs=[c.wd.site_xpos, c.wd.site_xmat, c.site_ids, px.first, c.wd.geom_xpos, c.wd.geom_xmat],
        )
    if px.tendon is not None:
        t = px.tendon
        s0, s1 = c.tendon_sites
        wp.launch(
            k["place_tendon"],
            dim=c.nworld,
            inputs=[
                c.wd.site_xpos,
                s0,
                s1,
                c.gravity,
                wp.float64(t.upper),
                int(t.ncat),
                int(t.nseg),
                px.tendon_first,
                float(t.radius),
                wp.vec3(*map(float, t.grey)),
                wp.vec3(*map(float, t.constraint)),
                c.solimp,
                wp.float64(c.tendon_margin),
                c.wd.geom_xpos,
                c.wd.geom_xmat,
                c.wm.geom_size,
                c.wm.geom_rgba,
            ],
        )
    _bvh(c)
    c.mjw.render(c.wm, c.wd, c.rc)
    if c.textures is not None:
        c.textures.render()
    if c.reflection is not None:
        c.reflection.render()


# --------------------------------------------------------------------------- per-call uploads


@dataclass
class _Plan:
    """Look groups of one style and whether any row recolours the floor."""

    groups: list
    palette: bool


def _plan(env, sty, B: int, opts: Options) -> _Plan:
    """Group rows by look. The host reads the style once per style tensor version."""
    values = sty.values
    # The cache holds the tensor itself, so a later style cannot reuse its storage and match by address.
    hit = env.__dict__.get("_warp_plan")
    if hit is not None and hit[0] is values and hit[1] == (values._version, int(B), opts):
        return hit[2]
    paths = _look_paths(env, opts, sty.paths)
    palette = False
    slots = _floor_slots(env)
    if "floor.color" in sty.paths and "floor.color" not in paths and slots is not None:
        q = torch.floor(sty["floor.color"].reshape(B, 2, 3).double() * 255).clamp(0, 255)
        default = torch.as_tensor(_quantise(slots["default"]), dtype=q.dtype, device=q.device)
        palette = not bool((q == default).all())
    if not paths:
        plan = _Plan([(None, (), None)], palette)
    else:
        cols = torch.cat([sty[p].reshape(B, -1) for p in paths], 1).double().cpu().numpy()
        uniq, inv = np.unique(np.round(cols, 6), axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        widths = [sty[p].reshape(B, -1).shape[1] for p in paths]
        groups = []
        for gi, row in enumerate(uniq):
            look, o = {}, 0
            for p, w in zip(paths, widths):
                look[p] = row[o : o + w].copy()
                o += w
            idx = None if len(uniq) == 1 else torch.as_tensor(np.flatnonzero(inv == gi), device=env.device)
            groups.append((look, tuple(np.round(row, 6).tolist()), idx))
        plan = _Plan(groups, palette)
    env.__dict__["_warp_plan"] = (values, (values._version, int(B), opts), plan)
    return plan


def _upload_rows(env, c: _Ctx, s: torch.Tensor, n: int) -> None:
    """Write compact rows into the render data and the per-world episode entries."""
    _view(c, "d", "qpos")[:n] = s[:, : env.nq]
    for j, (arr, k) in enumerate(env._slots()):
        _view(c, "m", arr)[:n].reshape(n, -1)[:, k] = s[:, env.nq + j]
    # Kinematics does not move world-body elements; their world positions are the episode entries.
    for kind, eid, base, comps in c.static_rows:
        row = base.repeat(n, 1)
        for j, comp in comps:
            row[:, comp] = s[:, env.nq + j]
        _view(c, "d", "geom_xpos" if kind == "geom" else "site_xpos")[:n, eid] = row


def _upload_trackcom(env, c: _Ctx, sty, lo: int, hi: int) -> None:
    """Per-row masses and the floor tilt of the tracking cells, for compact rows."""
    from ._mass_rows import ARRAYS as _MASS_ARRAYS

    n = hi - lo
    masses = env._row_masses(sty, lo, hi) if hasattr(env, "_row_masses") else None
    if masses is not None:
        for name, value in masses.items():
            _view(c, "m", name)[:n] = value.to(torch.float32)
        c.masses_varied = True
    elif c.masses_varied:
        for name, value in env._mass_rows()._constants(env.device).items():
            if name in _MASS_ARRAYS:
                _view(c, "m", name)[:] = value.to(torch.float32)
        c.masses_varied = False
    tilt = env._floor_tilt() if hasattr(env, "_floor_tilt") else None
    floor = env._row_floor(sty, lo, hi) if tilt is not None else None
    xmat = _view(c, "d", "geom_xmat") if tilt is not None else None
    if floor is not None:
        xmat[:n, tilt[1]] = floor
        c.floor_tilted = True
    elif c.floor_tilted:
        xmat[:, tilt[1]] = torch.eye(3, device=xmat.device, dtype=xmat.dtype)
        c.floor_tilted = False


def _upload_style(env, c: _Ctx, sty, sel, n: int, palette: bool) -> None:
    """Write the per-world appearance: materials, light scale, geom colours, sphere radii, gravity, floor."""
    ids, paths = c.mats
    if paths:
        _view(c, "m", "mat_rgba")[:n, ids, :3] = torch.stack([sty[p][sel] for p in paths], 1)
    scale = sty["light.diffuse_scale"][sel].reshape(n, 1, 1)
    _view(c, "m", "light_diffuse")[:n] = torch.clamp(c.light_base.unsqueeze(0) * scale, 0, 1)
    if c.colors:
        rgba = _view(c, "m", "geom_rgba")
        for gid, path in c.colors:
            rgba[:n, gid, :3] = sty[path][sel]
            rgba[:n, gid, 3] = 1.0
    for gid, path in c.spheres:
        _view(c, "m", "geom_size")[:n, gid, 0] = sty[path][sel]
    if c.proxies.tendon is not None and "gravity.x" in sty:
        gravity = c.wp.to_torch(c.gravity)
        for a, axis in enumerate("xyz"):
            gravity[:n, a] = sty[f"gravity.{axis}"][sel]
    if c.reflection is not None and c.reflection.palette is not None:
        rows = None
        if palette:
            rows = (torch.floor(sty["floor.color"][sel].reshape(n, 2, 3).double() * 255).clamp(0, 255) / 255).float()
        c.reflection.recolour(rows)


def _launch(c: _Ctx, device) -> None:
    stream = c.wp.stream_from_torch(torch.cuda.current_stream(device))
    if c.graph is not None:
        c.wp.capture_launch(c.graph, stream=stream)
        return
    with c.wp.ScopedDevice(c.wp_device), c.wp.ScopedStream(stream):
        _frame(c)


def _worlds(env, size: int, batch: int, spp: int, opts: Options) -> int:
    """Worlds per launch: the option, or the ray budget within [dmc_min_worlds, MAX_WORLDS]; at most the batch."""
    if opts.worlds is not None:
        cap = int(opts.worlds)
    else:
        where = dict(path="dmc-trackcom", size=size, batch=batch)
        rays = tuning.get("dmc_launch_rays", **where) // (int(size) * int(size) * int(spp) * int(spp))
        cap = max(tuning.get("dmc_min_worlds", **where), min(MAX_WORLDS, rays))
    return max(1, min(cap, 1 << max(0, int(batch) - 1).bit_length()))


def _context(env, size: int, nworld: int, look, key, opts: Options, spp: int, live: bool) -> _Ctx:
    cache_key = (type(env), int(size), int(nworld), key, opts, int(spp))
    if live:
        store = env.__dict__.setdefault("_warp_live", {})
        hit = store.get(cache_key)
        if hit is None:
            hit = store[cache_key] = _build(env, size, nworld, look, opts, spp, env._simulator.batch)
        return hit
    hit = _CACHE.get(cache_key)
    if hit is None:
        hit = _CACHE.put(cache_key, _build(env, size, nworld, look, opts, spp))
    return hit


def render(env, s, sty, size: int, out: torch.Tensor, spp: int, *, live: bool = False, opts=None, aux=None):
    """Draw rows ``s`` (or, with ``live``, the physics worlds) with style ``sty`` into ``out`` (B, H, W, 3).

    ``aux`` receives ``geom`` (B, H, W) int32 geom ids of the render model, -1 for the sky, and ``depth``
    (B, H, W) float32 ray hit distances when the options keep them, and ``names`` for the geom ids.
    """
    opts = env.render_options if opts is None else opts
    spp = int(opts.samples or spp)
    B = int(out.shape[0])
    plan = _plan(env, sty, B, opts)
    if live and len(plan.groups) > 1:
        s, live = env._pack_render_state(env.get_state()), False
    worlds = _worlds(env, size, B, spp, opts)
    if aux is not None:
        if opts.segmentation:
            aux["geom"] = torch.empty((B, size, size), dtype=torch.int32, device=env.device)
        if opts.depth:
            aux["depth"] = torch.empty((B, size, size), dtype=torch.float32, device=env.device)
    for look, key, idx in plan.groups:
        c = _context(env, size, worlds, look, key, opts, spp, live)
        rows = B if idx is None else len(idx)
        for lo in range(0, rows, worlds):
            hi = min(rows, lo + worlds)
            n = hi - lo
            sel = slice(lo, hi) if idx is None else idx[lo:hi]
            if live:
                c.offset_t.fill_(lo)
            else:
                _upload_rows(env, c, s[sel], n)
                if idx is None:
                    _upload_trackcom(env, c, sty, lo, hi)
            _upload_style(env, c, sty, sel, n, plan.palette)
            _launch(c, env.device)
            frames = c.bgra[:n, ..., [2, 1, 0]]  # MuJoCo Warp packs its bytes as BGRA.
            extra = {}
            if aux is not None and "geom" in aux:
                extra["geom"] = c.seg[:n, :, 0].reshape(n, size, size)
            if aux is not None and "depth" in aux:
                extra["depth"] = c.depth[:n].reshape(n, size, size)
            if idx is None:
                out[lo:hi] = frames
                for name, value in extra.items():
                    aux[name][lo:hi] = value
            else:
                out.index_copy_(0, sel, frames)
                for name, value in extra.items():
                    aux[name].index_copy_(0, sel, value)
            if aux is not None:
                aux["names"], aux["keymap"] = c.names, c.keymap
    return out


def live_renderer(env):
    """A callable that draws the live worlds from MuJoCo Warp physics, or None when that path does not apply."""
    hit = env.__dict__.get("_warp_live_fn", _UNSET)
    if hit is not _UNSET:
        return hit
    fn = None
    sim = env.__dict__.get("_simulator")
    batch = getattr(sim, "batch", None)
    if (
        env.device.type == "cuda"
        and selected(env)
        and getattr(sim, "simulation_backend", None) == "mjwarp"
        and batch is not None
        and _same_layout(type(env)._model(), batch.model)
    ):

        def fn():
            B = int(env.B)
            out = torch.empty((B, env.size, env.size, 3), dtype=torch.uint8, device=env.device)
            sty = env.resolve_style(None, None, batch=B)
            return render(env, None, sty, env.size, out, env.ss, live=True)

    env.__dict__["_warp_live_fn"] = fn
    return fn


def _same_layout(m, physics) -> bool:
    names = ("ngeom", "nsite", "ncam", "nlight", "nbody", "nq")
    return all(int(getattr(m, n)) == int(getattr(physics, n)) for n in names) and np.array_equal(
        np.asarray(m.geom_bodyid), np.asarray(physics.geom_bodyid)
    )


def semantics(env, rows, size: int | None = None) -> dict:
    """Draw compact ``rows`` with one sample per pixel and return the frames and their semantics.

    The result holds ``rgb`` (B, H, W, 3) uint8, ``geom`` (B, H, W) int32 render-model geom ids (-1 for
    the sky), ``depth`` (B, H, W) float32 planar depth along the optical axis, ``names`` (one per render
    geom: the cell's geoms, then ``site:<name>`` and ``tendon:<name>`` proxies) and ``keymap``
    (render geoms, 2) int64 ``(objid, objtype)`` pairs of the cell's model in MuJoCo's ``mjtObj`` numbering.
    """
    size = int(size or env.size)
    s = torch.as_tensor(rows, device=env.device, dtype=torch.float32).reshape(-1, env.state_dim)
    B = int(s.shape[0])
    sty = env.resolve_style(None, None, batch=B)
    opts = replace(env.render_options, samples=1, segmentation=True, depth=True)
    out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=env.device)
    aux: dict = {}
    render(env, s, sty, size, out, 1, opts=opts, aux=aux)
    return {"rgb": out, **aux}


def semantic(env, rows) -> tuple[torch.Tensor, torch.Tensor]:
    """``(seg, depth)`` of compact ``rows`` at ``env.size``, for the semantic fidelity harness.

    ``seg`` is (B, H, W, 2) int64 ``(objid, objtype)`` pairs of the cell's model, which is the model of
    ``reference_env``; the sky is -1. ``depth`` is (B, H, W) float32 planar depth.
    """
    out = semantics(env, rows)
    geom = out["geom"].long()
    pairs = out["keymap"][geom.clamp(min=0)]
    seg = torch.where((geom >= 0).unsqueeze(-1), pairs, torch.full_like(pairs, -1))
    return seg, out["depth"]


def release(env) -> None:
    """Drop the contexts that read this environment's physics."""
    for c in env.__dict__.pop("_warp_live", {}).values():
        MW.release_cached_ctx(c)
    env.__dict__.pop("_warp_live_fn", None)
    env.__dict__.pop("_warp_plan", None)
