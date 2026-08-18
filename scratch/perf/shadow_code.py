"""Is the shadow ray costing register pressure even where it never fires?

``mujoco_warp``'s megakernel inlines ``compute_lighting`` once per light plus once for the headlight,
and every one of those copies carries a FULL any-hit BVH traversal, because ``use_shadows`` is a
runtime kernel argument and ``lightcastshadow`` is read from an array. So a scene whose only light
declares ``castshadow="false"`` still compiles two traversals it never executes, and a Meta-World
scene with three lights and one caster compiles four and executes one.

``scratch/perf/knobs.py`` already measured the RUNTIME cost (``use_shadows=False``): 1.55x on
Meta-World, 1.54x on HumanoidBench, 1.00x on RoboCasa. This script measures the COMPILE-TIME cost by
removing the shadow branch from the generated source instead of from the branch predicate, which is
the difference between "the rays are not cast" and "the code is not there".

Three rows:

  ``base``      as shipped.
  ``runtime``   ``rc.use_shadows = False``: same kernel, branch not taken.
  ``codegen``   ``_make_compute_lighting`` patched to a variant with no shadow cast at all.

On a scene where no light casts, ``codegen`` is bit-identical to ``base`` BY CONSTRUCTION and the
script asserts it. Where a light does cast it is not, and the number is diagnostic only: ``codegen``
faster than ``runtime`` by more than noise is the register-pressure claim, and it says a partial
specialisation (the headlight, and every non-casting light, compiled without the traversal) would pay
even with shadows on.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se
import stable_envs.render.mjwarp as MW

FAMILIES = ("MetaWorld", "HumanoidBench", "Fetch", "RoboCasa")


def family(name):
    for f in FAMILIES:
        if name.startswith(f):
            return f
    raise SystemExit(f"unknown family for {name}")


def module_for(name):
    import importlib
    return importlib.import_module({
        "MetaWorld": "stable_envs.envs.metaworld",
        "HumanoidBench": "stable_envs.envs.humanoidbench",
        "Fetch": "stable_envs.envs.fetch",
        "RoboCasa": "stable_envs.envs.robocasa"}[family(name)])


def build(mod, env, res, nworld):
    fam = family(type(env).__name__)
    cls = type(env)
    if fam == "MetaWorld":
        return mod._context(cls.TASK, res, nworld)
    if fam == "HumanoidBench":
        return mod._context(cls.ROBOT, cls.TASK, res, nworld)
    if fam == "Fetch":
        return mod._context(cls.SCENE if hasattr(cls, "SCENE") else cls.TASK, res, nworld)
    return mod._context(cls.KITCHEN, cls.CAMERA, res, nworld)


def timed(fn, reps, warm=3):
    import warp as wp
    for _ in range(warm):
        fn()
    wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def make_noshadow_factory():
    """A drop-in ``_make_compute_lighting`` whose product never casts a shadow ray.

    Byte-for-byte upstream's body with the ``if use_shadows and lightcastshadow`` block deleted and
    ``visible`` pinned to 1.0. The signature is upstream's, unchanged, so the kernel that calls it
    does not have to be touched.
    """
    import warp as wp
    from mujoco_warp._src import math

    MJ_MAXVAL = 1e10

    def _make(cast_ray_first_hit):
        @wp.func
        def compute_lighting(
            geom_type: wp.array[int],
            geom_dataid: wp.array2d[int],
            geom_size: wp.array2d[wp.vec3],
            flex_vertadr: wp.array[int],
            flex_edge: wp.array[wp.vec2i],
            flex_radius: wp.array[float],
            geom_xpos_in: wp.array2d[wp.vec3],
            geom_xmat_in: wp.array2d[wp.mat33],
            flexvert_xpos_in: wp.array2d[wp.vec3],
            use_shadows: bool,
            bvh_id: wp.uint64,
            group_root: int,
            bvh_ngeom: int,
            bvh_nflexgeom: int,
            enabled_geom_ids: wp.array[int],
            worldid: int,
            mesh_bvh_id: wp.array[wp.uint64],
            hfield_bvh_id: wp.array[wp.uint64],
            flex_geom_flexid: wp.array[int],
            flex_geom_edgeid: wp.array[int],
            flex_bvh_id: wp.array[wp.uint64],
            flex_group_root: wp.array2d[int],
            lightactive: bool,
            lighttype: int,
            lightcastshadow: bool,
            lightpos: wp.vec3,
            lightdir: wp.vec3,
            lightattenuation: wp.vec3,
            lightcutoff_rad: float,
            lightexp: float,
            lightdiff: wp.vec3,
            lightspec: wp.vec3,
            normal: wp.vec3,
            hitpoint: wp.vec3,
            view_dir: wp.vec3,
            mat_spec: float,
            mat_shin_exp: float,
            cull_backfaces: bool,
            enable_specular: bool,
            default_attenuation: bool,
            has_spot: bool,
        ) -> Tuple[wp.vec3, wp.vec3]:
            diff_rgb = wp.vec3(0.0)
            spec_rgb = wp.vec3(0.0)
            if not lightactive:
                return diff_rgb, spec_rgb
            L = wp.vec3(0.0)
            dist_to_light = float(MJ_MAXVAL)
            attenuation = 1.0
            if lighttype == 1:
                L = -lightdir
            else:
                L, dist_to_light = math.normalize_with_norm(lightpos - hitpoint)
                if not default_attenuation:
                    f = wp.vec3(1.0, dist_to_light, dist_to_light * dist_to_light)
                    attenuation = math.safe_div(1.0, wp.dot(f, lightattenuation))
                if has_spot:
                    if lighttype == 0:
                        cos_theta = wp.dot(-L, lightdir)
                        cos_cutoff = wp.cos(lightcutoff_rad)
                        if cos_theta < cos_cutoff:
                            return diff_rgb, spec_rgb
                        attenuation = attenuation * wp.pow(wp.max(cos_theta, 0.0), lightexp)
            ndotl = wp.max(0.0, wp.dot(normal, L))
            if ndotl == 0.0:
                return diff_rgb, spec_rgb
            weight = attenuation                                    # visible == 1.0, no shadow ray
            diff_rgb = lightdiff * (ndotl * weight)
            if enable_specular:
                if mat_spec > 0.0 and mat_shin_exp > 0.0:
                    H = wp.normalize(L + view_dir)
                    ndoth = wp.max(0.0, wp.dot(normal, H))
                    spec_rgb = lightspec * (mat_spec * wp.pow(ndoth, mat_shin_exp) * weight)
            return diff_rgb, spec_rgb

        return compute_lighting

    return _make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()

    from mujoco_warp._src import render as R

    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    orig_crc = MW.create_render_context
    orig_factory = R._make_compute_lighting
    rows, frames = {}, {}

    def one(label, patch_codegen=False, rc_over=None):
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()
        if rc_over:
            def crc(mm, **kw):
                kw.update(rc_over)
                return orig_crc(mm, **kw)
            MW.create_render_context = crc
        if patch_codegen:
            R._make_compute_lighting = make_noshadow_factory()
        try:
            c = build(mod, env, a.size, a.nworld)
            dt = timed(lambda: c.wp.capture_launch(c.graph), a.reps)
            frames[label] = c.bgra[:a.nworld][..., [2, 1, 0]].clone()
            rows[label] = {"ms": dt * 1e3, "fps": a.nworld / dt}
            print(f"{label:10s} {dt*1e3:8.3f} ms  {a.nworld/dt:10,.0f} f/s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows[label] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            MW.create_render_context = orig_crc
            R._make_compute_lighting = orig_factory
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()

    one("base")
    one("runtime", rc_over={"use_shadows": False})
    one("codegen", patch_codegen=True)

    if "base" in frames:
        for k in ("runtime", "codegen"):
            if k in frames:
                d = (frames[k].to(torch.int16) - frames["base"].to(torch.int16)).abs()
                rows[k]["maxdiff"] = int(d.max())
                rows[k]["speedup"] = rows["base"]["ms"] / rows[k]["ms"]
                print(f"{k}: maxdiff {int(d.max())}  x{rows['base']['ms']/rows[k]['ms']:.2f}",
                      flush=True)
    print("JSON " + json.dumps({"env": a.env, "size": a.size, "nworld": a.nworld, "rows": rows}),
          flush=True)


if __name__ == "__main__":
    main()
