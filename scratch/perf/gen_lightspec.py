"""Generate ``stable_envs/render/mjwarp_lightspec.py`` from the INSTALLED mujoco_warp source.

The vendored variant differs from upstream in three places and nowhere else, so it is produced by
transforming upstream's own text rather than retyped: a 470-line hand copy is a transcription bug
waiting to happen, and a generator makes the diff against a future mujoco_warp release one command
rather than a review.

The three edits, and nothing else:

1. ``_make_compute_lighting`` takes ``can_shadow`` and puts the shadow cast under
   ``if wp.static(can_shadow):``. Warp's ``emit_If`` evaluates a constant condition at codegen and
   emits ONLY the taken branch (``warp/_src/codegen.py`` line 3212), so this deletes the any-hit
   traversal from the generated source rather than branching around it at runtime.
2. ``render`` builds BOTH variants and dispatches per light inside the already-unrolled light loop,
   on ``wp.static(LIGHT_CASTS[light_index])``. ``wp.static`` can read an unrolled loop variable
   because the unroller binds it as a constant in ``adj.symbols`` and static evaluation is given
   those constants (same file, line 6025). The loop ORDER is untouched, which is what keeps the
   float accumulation into ``result`` bit-identical.
3. The headlight call uses the shadow-free variant. It already passes ``lightcastshadow=False`` as a
   literal, so this removes a traversal that provably never runs.

Run:  python scratch/perf/gen_lightspec.py [--check]
"""
from __future__ import annotations

import argparse
import os
import re

UPSTREAM = "/mnt/vast/home/randall.balestriero/venvs/se-hb/lib/python3.11/site-packages/mujoco_warp/_src/render.py"
OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render/mjwarp_lightspec.py"

HEADER = '''"""``mujoco_warp.render`` with the shadow traversals that never execute removed at COMPILE time.

WHAT THIS IS FOR, IN ONE MEASUREMENT. ``mujoco_warp``'s megakernel inlines ``compute_lighting`` once
per light plus once for the headlight, and every copy carries a full any-hit BVH traversal, because
``use_shadows`` is a runtime kernel argument and ``lightcastshadow`` is read from a per-world array.
Meta-World has three lights and a headlight, of which exactly ONE ever casts
(``light_castshadow = [false, true, false]``, and the headlight is called with a literal ``False``),
so four traversals are compiled and one runs. Measured at 256 px / 64 worlds on ``MetaWorldReach``
(``scratch/perf/shadow_code.py``):

    baseline                              7.636 ms
    rc.use_shadows = False (rays gone)    4.930 ms   x1.55
    traversal deleted from the SOURCE     4.390 ms   x1.74

so **1.12x of the frame is register pressure from code that never runs**, and it is available without
changing a pixel. That is what this module takes.

HOW, AND WHY IT IS BIT-IDENTICAL. Two things in Warp make this a specialisation rather than a
rewrite, both checked in ``warp/_src/codegen.py`` rather than assumed:

* ``emit_If`` (line 3212) evaluates a CONSTANT condition at codegen time and lowers only the taken
  branch, so ``if wp.static(can_shadow):`` deletes the traversal from the generated C++ instead of
  guarding it with ``if (false)``;
* ``evaluate_static_expression`` (line 6025) hands every constant in ``adj.symbols`` to the static
  evaluator, and the loop unroller binds the loop variable as exactly such a constant (line 3822),
  so ``wp.static(LIGHT_CASTS[light_index])`` inside ``for light_index in range(wp.static(M_NLIGHT))``
  resolves per iteration.

The light loop keeps upstream's ORDER and upstream's arithmetic; only the callee changes, and the
callee it changes to computes ``visible = 1.0`` for a light that could not have shadowed anything
anyway. Nothing else in the megakernel is touched.

WHAT IT IS NOT SAFE FOR, GUARDED IN ``applies_to``. The decision is compile-time, so a model whose
``light_castshadow`` is BATCHED PER WORLD cannot use it -- a world could turn a caster on after the
kernel was built. ``applies_to`` refuses in that case and the caller falls back to stock
``mujoco_warp``, which is also what ``STABLE_ENVS_LIGHTSPEC=0`` does.

``pack_rgba_to_uint32`` IS RE-READ FROM UPSTREAM AT CALL TIME, and that is load-bearing:
``render/mjwarp.py``'s ``gl_rounding()`` patches the name inside ``mujoco_warp._src.render``, and a
vendored module that had bound it at import would silently render with the TRUNCATING quantiser --
worth interior 0.9988 -> 0.791 on Fetch, i.e. the single largest fidelity term on this backend. The
same applies to ``sample_texture``, ``sample_skybox`` and ``_make_cast_ray``: all four are looked up
on the upstream module inside ``render`` so every patch a caller applies to upstream still reaches
this kernel.

GENERATED, DO NOT EDIT BY HAND. ``scratch/perf/gen_lightspec.py`` produces this file by transforming
the installed ``mujoco_warp._src.render`` source in three places; ``--check`` re-generates and diffs,
which is the test to run after a mujoco_warp bump.

Pinned against **mujoco-warp {version}**.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Tuple

import numpy as np
import warp as wp

from mujoco_warp._src import math  # noqa: F401  (referenced by the vendored function bodies)
from mujoco_warp._src.types import MJ_MAXVAL, Data, GeomType, Model, ObjType, RenderContext
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({{"enable_backward": False}})

#: The mujoco_warp release this file was generated from. A bump is not automatically wrong, it is
#: unverified: re-run ``scratch/perf/gen_lightspec.py --check``.
UPSTREAM_VERSION = "{version}"

MAX_SHININESS = 128.0
DEFAULT_MAT_SHININESS_EXPONENT = 0.5 * MAX_SHININESS
DEFAULT_MAT_SPECULAR = 0.5
DEFAULT_MAT_EMISSION = 0.0
NO_LIGHT_AMBIENT_FALLBACK = 0.3


def applies_to(m: Model, rc: RenderContext) -> bool:
    """Is a COMPILE-TIME light specialisation sound for this model and this context?

    Only when ``light_castshadow`` is a constant of the model rather than a per-world field: the
    whole point is that the decision is baked into the kernel, so a caller that randomises which
    lights cast per world must get stock ``mujoco_warp``. Returns False rather than raising, because
    the fallback is correct and merely slower.

    This is the MODEL guard only. Whether a family opts in at all is decided by its ``_context``
    through ``specialised_lights(enabled=...)``, because the answer differs per family -- see that
    function.
    """
    cs = getattr(m, "light_castshadow", None)
    return cs is not None and len(cs.shape) == 2 and cs.shape[0] == 1


#: Memo for ``light_casts``, and it is a correctness requirement rather than a saving. ``render`` is
#: called once to warm up and then AGAIN INSIDE ``wp.ScopedCapture``; reading ``light_castshadow``
#: with ``.numpy()`` is a device-to-host copy, which CUDA refuses during a capture with
#: "operation would make the legacy stream depend on a capturing blocking stream". The warm-up call
#: fills this, so the capturing call never copies. The value keeps a reference to the array so the
#: pointer it is keyed on cannot be freed and reissued to a different model.
_CASTS_CACHE: dict = {{}}


def light_casts(m: Model, rc: RenderContext) -> tuple:
    """``(nlight,)`` of bools: does light ``i`` compile a shadow traversal at all?"""
    key = (int(m.light_castshadow.ptr), int(m.nlight), bool(rc.use_shadows))
    hit = _CASTS_CACHE.get(key)
    if hit is None:
        if not rc.use_shadows:
            casts = tuple(False for _ in range(int(m.nlight)))
        else:
            casts = tuple(bool(x) for x in m.light_castshadow.numpy()[0])
        hit = _CASTS_CACHE[key] = (casts, m.light_castshadow)
    return hit[0]


#: The stock ``mujoco_warp`` render, stashed by ``specialised_lights`` so the vendored one can hand a
#: model it must not specialise straight back to it. Without this the fallback would re-enter the
#: patched name and recurse.
_STOCK_RENDER = None


@contextlib.contextmanager
def specialised_lights(enabled: bool = True):
    """Point ``mujoco_warp.render`` at this module's ``render`` for the duration of the block.

    A context manager for the same reason ``gl_rounding`` is one: the window is short, it covers the
    context build and the graph capture (which is when the megakernel source is generated and
    compiled, so it is the only window that matters), and nothing else in the process ever sees it.

    ``enabled`` IS THE CALLER'S CALL AND DEFAULTS TO OFF FOR A PARITY FAMILY, because the elision is
    bit-identical on most scenes and not on all of them. Measured with ``scratch/perf/lightspec_kernel.py``,
    comparing the raw megakernel output over 64 states at 256 px:

        RoboCasa (2 cells), HumanoidBench (2 cells), Meta-World reach / sweep-into   0 / 4,194,304 px
        Meta-World window-open, window-close                    max 1/255 on 2 / 4,194,304 px

    The segmentation buffer is identical everywhere, so nothing about WHAT is hit changes; what
    changes is that ``compute_lighting_flat`` is a different inlining context for the arithmetic that
    survives, and ``mujoco_warp`` compiles its megakernel with ``fast_math`` enabled, which licenses
    the reassociation that moves a handful of values across a quantisation boundary. It reproduces
    exactly across runs (2 px at 64 states, 21 at 256, on three independent jobs), so it is a
    deterministic difference and not a flake -- the stock-vs-stock control is 0 px.

    Two pixels in four million cannot move Meta-World's declared parity floors, and that is beside
    the point: the family claims bit-identity, so it gets stock ``mujoco_warp`` unless a caller asks
    otherwise. ``STABLE_ENVS_LIGHTSPEC=1`` overrides for anyone who wants the 1.04-1.09x, and
    ``=0`` forces it off everywhere.
    """
    import mujoco_warp as mjw
    from mujoco_warp._src import render as R

    override = os.environ.get("STABLE_ENVS_LIGHTSPEC")
    if override is not None:
        enabled = override == "1"
    if not enabled:
        yield False
        return
    global _STOCK_RENDER
    orig_mod, orig_pkg = R.render, mjw.render
    _STOCK_RENDER = orig_mod
    R.render = render
    mjw.render = render
    try:
        yield True
    finally:
        R.render = orig_mod
        mjw.render = orig_pkg


'''


def take(src: str, name: str) -> str:
    i = src.index(f"def {name}(")
    j = src.rindex("\n", 0, i) + 1
    m = re.search(r"\n(?=@event_scope\n|def |@wp\.func\n)", src[i:])
    return src[j:i + m.start() + 1] if m else src[j:]


def once(hay: str, needle: str) -> None:
    n = hay.count(needle)
    if n != 1:
        raise SystemExit(f"anchor appears {n} times, expected 1:\n{needle!r}")


def patch_compute_lighting(cl: str) -> str:
    sig = "def _make_compute_lighting(cast_ray_first_hit: wp.Function) -> wp.Function:"
    once(cl, sig)
    cl = cl.replace(sig, "def _make_compute_lighting(cast_ray_first_hit, can_shadow: bool) -> wp.Function:")
    cl = cl.replace('  """Build specialized compute_lighting."""',
                    '  """Build specialized compute_lighting. ``can_shadow`` is COMPILE TIME: False deletes\n'
                    "  the any-hit traversal from the generated source (warp's emit_If folds a constant test).\"\"\"")
    start = cl.index("    if use_shadows and lightcastshadow:")
    end = cl.index("    weight = attenuation * visible")
    block = cl[start:end]
    indented = "".join(("  " + ln if ln.strip() else ln) for ln in block.splitlines(keepends=True))
    return cl[:start] + "    if wp.static(can_shadow):\n" + indented + cl[end:]


def patch_render(rd: str) -> str:
    # 1. the two variants, and the per-light table
    anchor = "  compute_lighting = _make_compute_lighting(cast_ray_first_hit)"
    once(rd, anchor)
    rd = rd.replace(anchor,
                    "  # THE SPECIALISATION. Two variants of the same function; which one a light gets is decided\n"
                    "  # below, per light, at codegen. ``compute_lighting_flat`` carries no traversal at all.\n"
                    "  compute_lighting = _make_compute_lighting(cast_ray_first_hit, True)\n"
                    "  compute_lighting_flat = _make_compute_lighting(cast_ray_first_hit, False)\n"
                    "  if not applies_to(m, rc):\n"
                    "    # A per-world ``light_castshadow`` (or the opt-out) means the decision cannot be\n"
                    "    # baked; hand the frame to stock mujoco_warp, which is correct and merely slower.\n"
                    "    return _STOCK_RENDER(m, d, rc)\n"
                    "  LIGHT_CASTS = light_casts(m, rc)")

    # 2. upstream reads these at module import; re-read them here so gl_rounding() still applies.
    anchor = "  rc_static = {"
    once(rd, anchor)
    rd = rd.replace(anchor,
                    "  # RE-READ FROM UPSTREAM, NOT IMPORTED AT MODULE LOAD: ``gl_rounding()`` patches the name\n"
                    "  # inside ``mujoco_warp._src.render``, and a copy bound at import would keep the truncating\n"
                    "  # quantiser for every frame this kernel ever draws. Same for the three helpers.\n"
                    "  from mujoco_warp._src import render as _up\n"
                    "  pack_rgba_to_uint32 = _up.pack_rgba_to_uint32\n"
                    "  sample_texture = _up.sample_texture\n"
                    "  sample_skybox = _up.sample_skybox\n"
                    "  compute_ray = _up.compute_ray\n"
                    "  rc_static = {")
    rd = rd.replace("  cast_ray = _make_cast_ray(geom_ray_types, first_hit=False)\n"
                    "  cast_ray_first_hit = _make_cast_ray(geom_ray_types, first_hit=True)",
                    "  from mujoco_warp._src.render import _make_cast_ray\n"
                    "  cast_ray = _make_cast_ray(geom_ray_types, first_hit=False)\n"
                    "  cast_ray_first_hit = _make_cast_ray(geom_ray_types, first_hit=True)")

    # 3. the per-light call site -> a static per-light dispatch, ORDER UNCHANGED
    head = "      diff_rgb, spec_rgb = compute_lighting(\n"
    once(rd, head)
    s = rd.index(head)
    e = rd.index("\n      )\n", s) + len("\n      )\n")
    call = rd[s:e]
    args = call[len(head):-len("      )\n")]
    args_in = "".join(("  " + ln if ln.strip() else ln) for ln in args.splitlines(keepends=True))
    new = ("      # ONE traversal is compiled for a light that casts and NONE for a light that does not.\n"
           "      # The branch is resolved at codegen (constant test), so the untaken side emits nothing;\n"
           "      # the loop order, the arguments and the accumulation below are upstream's, untouched.\n"
           "      if wp.static(LIGHT_CASTS[light_index]):\n"
           "        diff_rgb, spec_rgb = compute_lighting(\n"
           + args_in + "        )\n"
           "      else:\n"
           "        diff_rgb, spec_rgb = compute_lighting_flat(\n"
           + args_in + "        )\n")
    rd = rd[:s] + new + rd[e:]

    # 4. the headlight already passes lightcastshadow=False as a literal
    hl = "      hl_diff, hl_spec = compute_lighting(\n"
    once(rd, hl)
    rd = rd.replace(hl,
                    "      # The headlight passes ``lightcastshadow=False`` as a literal three arguments down,\n"
                    "      # so its traversal was dead on arrival; this is the same call with it gone.\n"
                    "      hl_diff, hl_spec = compute_lighting_flat(\n")
    return rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="regenerate and diff instead of writing")
    a = ap.parse_args()

    import mujoco_warp
    version = getattr(mujoco_warp, "__version__", "unknown")

    src = open(UPSTREAM).read()
    out = (HEADER.format(version=version)
           + patch_compute_lighting(take(src, "_make_compute_lighting"))
           + "\n"
           + patch_render(take(src, "render")))
    if a.check:
        cur = open(OUT).read() if os.path.exists(OUT) else ""
        if cur == out:
            print("mjwarp_lightspec.py is up to date against mujoco-warp " + version)
        else:
            import difflib
            print("".join(difflib.unified_diff(cur.splitlines(True), out.splitlines(True),
                                               "on disk", "regenerated"))[:8000])
            raise SystemExit(1)
        return
    with open(OUT, "w") as f:
        f.write(out)
    print(f"wrote {OUT} ({out.count(chr(10))} lines) against mujoco-warp {version}")


if __name__ == "__main__":
    main()
