"""HumanoidBench (Unitree H1 / H1-with-hands) — the repo's second **VISUAL TIER** family.

These are NOT the official pixels. What they are: MJWarp's raytrace of HumanoidBench's own compiled
scene, from the same ``qpos``, through the same model camera, at 512x the official renderer.

WHY THIS FAMILY IS NOT THE PARITY TIER, IN ONE NUMBER
------------------------------------------------------

**Every camera in HumanoidBench is ``mode="trackcom"``**, so the camera rides the humanoid's centre of
mass and the static scene sweeps through the frame. There is no plate: hide every mover, render eight
states, and the "static" picture still differs by **54/255 over 21.7 % of its pixels**. That removes
the design ``fetch.py``, ``metaworld`` and ``ogbench`` all reach parity with -- bake the static
majority through MuJoCo itself, ray-cast only what moves -- and leaves full-scene MJWarp, which then
has to draw the floor.

The floor is **66-72 % of a locomotion frame** and carries **99.7 % of the interior error**. It is a
40 m ``PLANE`` (so MJWarp derives its UVs correctly; this is not the Adroit/RoboCasa UV gap) wearing a
DEGENERATE checker: 100x100 texels, 9,604 at 229 and 396 at 178, i.e. a flat grey with a one-texel
``mark="edge"`` border tiled 20x20 -- a **2 cm grid line every 2 m**. MuJoCo mipmaps it; MJWarp
point-samples a linear filter of the base level. Drawn as-is the floor comes out DITHERED, and the
numbers say so: interior agreement 0.877, SSIM 0.639.

So the floor's texture is replaced by its dominant texel and the grid is not drawn. That is a
deliberate approximation and it is the one this module makes, because it is measurably CLOSER to the
official frame on every metric at once (h1-walk, 16 states, 256 px):

=====================  ==========  =========  ==========  ======
floor                   interior    mean/255   within 8    SSIM
=====================  ==========  =========  ==========  ======
textured (as-is)          0.9158       4.13      0.901     0.628
**dominant texel**        0.9503       2.79      0.935     0.851
=====================  ==========  =========  ==========  ======

Supersampling was tried and is refuted: 1x -> 4x moves interior agreement 0.877 -> 0.682 while the
mean falls 4.29 -> 3.50, which is the signature of a minification residual rather than a coverage one
(it blurs the pixels that were right). ``docs/research/humanoidbench.md`` has the ladder.

WHAT THIS TIER LICENSES, AND WHAT IT DOES NOT
-----------------------------------------------

  YES   train on frames generated live from stored ``qpos``, at 13-20 k f/s per GPU at 256 px against
        **38.5 f/s** for HumanoidBench's own rasterizer, with camera, floor colour and lighting
        randomised per environment inside one launch.
  NO    train here and evaluate on official HumanoidBench pixels. The two pixel distributions differ
        by the amounts in ``visual_fidelity`` and nothing in this repo measures what that costs a
        policy.
  NO    treat any number here as parity. ``parity_interior_min`` is ``None`` on purpose.

WHAT PASSES, AND CLEANLY
-------------------------

G1 purity is 0/255 on four independent probes (re-render; ``qvel`` drawn from U[-100,100] a hundred
times; the same ``qpos`` reached through a different rollout; a FRESH ``MjData``), and the frame
straight out of ``reset()`` is already forwarded, so the Meta-World staleness rule does not bite here.
G2 is ``nq`` float64 against a 196,608-byte frame = **0.001-0.003x**. G3 is the gate that killed
RoboCasa and it passes outright: **not one model array changes** across a reset or across five steps,
on every scene shipped here. The decoder is a constant of the family.

WHICH SCENES ARE HERE, AND WHY THE OTHERS ARE NOT
---------------------------------------------------

HumanoidBench has 31 tasks; 49 (robot, task) scenes compile on ``mujoco`` 3.11. Two things keep a
scene out, both measured over all 49 in ``docs/research/humanoidbench.md``:

*Alpha.* ``generated_xml_hurdles.xml`` and friends give their barriers ``rgba="0.1 0.1 0.1 0.1"`` --
black at **alpha 0.1**. MuJoCo blends them to a faint tint; MJWarp reads only ``color[0..2]`` and
paints them opaque black, over **62.8 %** of a hurdle frame (mean 111/255, SSIM 0.27). Hiding them
instead is worse where the geometry is load-bearing: on ``maze``'s 30 translucent walls it takes SSIM
from 0.665 to 0.344. This is the gap that killed Adroit's ``Pen``, and it is upstream.

*Textured primitives, at scale.* ``room`` (25), ``cabinet`` (21), ``truck`` (19), ``spoon`` (15) and
``window`` (13) wear textures on BOXes and CYLINDERs, which MJWarp samples at texel (0, 0) -- the
RoboCasa mechanism. A few of those still clear the bar and are shipped WITH that residual declared;
the ones that do not are named in the research note with their numbers.

ONE WORLD IN AN MJWARP LAUNCH IS NOT FULLY INDEPENDENT OF THE OTHERS
---------------------------------------------------------------------

``mujoco_warp.refit_bvh`` refits ONE acceleration structure over every world in a launch, so the tree
a ray traverses is a function of all of them -- and where two surfaces are nearly coincident a near-tie
can then be broken the other way, by 1-2/255. Two consequences are handled here and one is not:

* the render context's world count is **pinned at 64 for every batch** rather than sized to the chunk,
  because MJWarp's megakernel built for ``nworld = 1`` does not produce the same bytes as the ones
  built for 4, 8 or 64. See ``render``.
* ``_upload`` **fills every world**, padding with the last requested row, so no world contributes
  whatever the memory pool last left there to the BVH's bounds. See ``_upload``.
* what is left is irreducible, and it is why two measured cells are NOT registered:
  ``h1-highbar_simple`` reproducibly renders one row 1/255 differently at batch 1 and at batch 1024
  (its posts are thin cylinders meeting the bar), and ``h1hand-powerlift`` failed ``test_determinism``
  on two runs of four. Their fidelity numbers are as good as their neighbours' -- highbar is the best
  interior-exact of all 19 at 0.960 -- so they are excluded on this and on nothing else.

MEASURED (256 px, 64 states x 4 independent draws from ``sample_states``, one H200)
------------------------------------------------------------------------------------

Per class, in ``FIDELITY`` below and in each class's ``visual_fidelity``. Across the 17 registered
cells: mean absolute error **2.6-8.4/255**, within-8 **0.795-0.958**, SSIM **0.845-0.931**,
interior-exact **0.607-0.943** -- every column strictly better than ``robocasa.py``, the other
visual-tier family (mean 5.8-32.1, within-8 0.37-0.84, SSIM 0.82-0.92, interior-exact 0.05-0.16).

Throughput at the MJWarp surface on ``h1hand-walk``: 2,063 f/s at batch 1, 7,178 at 4, 13,127 at 16,
17,707 at 64, 19,719 at 256, 20,295 at 1024, against **38.5 f/s** for ``mujoco.Renderer`` at the same
camera and resolution -- so **(512x, batch 256)**, and the crossover against the official renderer is
**batch 1** (53.6x). Against one CPU core of JPEG decode at 256 px (5,486 f/s) it is 0.38 cores at
batch 1 and 3.6 at 256, crossing over at **batch 4**. Un-captured the same four kernels run at 422 f/s,
i.e. **1.4x** rather than 512x: the CUDA graph is the whole port.

At the ENV's own surface -- states in, uint8 frames out, appearance resolution and channel swap
included, and with the world count pinned at 64 for every batch (see ``render``) -- ``h1-walk`` runs
278 f/s at batch 1, 2,191 at 8, 4,317 at 16, 16,839 at 64 and 16,971 at 256, and ``h1hand-spoon``, the
heaviest cell, 152 / 1,197 / 2,371 / 8,961 / 9,150. So the pair a caller gets is **(437x, batch 64)**
against HumanoidBench's own 38.5 f/s, it is already 7.2x at batch 1, and against one CPU core of JPEG
decode it crosses over at about **batch 20** and reaches **3.1 cores** at batch 64. A batch of 1 pays
64 worlds' worth of launch for one frame; that is the price of batch invariance and it is gone by 64.

WHERE THE FRAME'S TIME GOES, AND THE FOUR LEVERS THAT DO NOT APPLY
--------------------------------------------------------------------

Measured at 256 px / batch 1024 on one H200, against the **18,595-22,197 f/s** that 8 CPU workers
reach decoding this family's own JPEG-q95 frames out of a Lance table
(``tests/fixtures/regression/references/lance.json``). The env runs at **86-87 % of its own megakernel** --
``h1-walk`` 19,312 f/s against 22,300 for a bare ``capture_launch`` at 64 worlds,
``h1hand-balance_simple`` 11,275 against 14,100 -- so there is nothing outside the kernel left worth
taking and the whole question is what the kernel costs.

**78 % of the kernel is inside the 25 % of the frame the robot occupies, and the robot covers 4.0-6.1 %
of it.** A 128 px window centred on the movers costs 2.221 ms of the full frame's 2.863, so the outer
three quarters of the rays cost 0.64 ms between them: **0.20 ns per ray outside against 12.6 ns
inside**. Every lever follows from that ratio.

* **A screen window round the movers is worth at most 1.29x** -- 25 % of the rays for 1.29x, and
  bit-identical to the full frame's crop -- because the window
  contains precisely the expensive rays. Composited against a static plate it is worth 1.10x: this
  family's camera is ``TRACKCOM``, so the plate moves with the state and has to be redrawn every frame
  (0.389 ms) rather than cached.
* **A static/dynamic split is worth 1.12x**, because 21 of the 22 visible geoms ARE the robot. It is the design ``fetch.py`` and ``metaworld.py`` are built on
  and there is nothing here for it to bake.
* **The shadow ray is worth 1.42-1.53x and is not available.** It is 35 % of the kernel
  (``use_shadows=False``: 2.863 -> 1.870 ms) and dropping it fails the floors these classes already
  declare, measured over 64 states against the official renderer:
  ``h1-walk`` mean 2.88 -> 7.30 against a ceiling of 3.4 and SSIM 0.8425 -> 0.7453 against a floor of
  0.80; ``h1hand-balance_simple`` 5.37 -> 10.08 and 0.8601 -> 0.7626; ``h1-sit_hard`` 3.41 -> 8.15 and
  0.8614 -> 0.7710. The visual tier licenses a DIFFERENT picture, not a picture below the numbers it
  publishes.
* **nvdiffrast is 0.16-0.22x**, measured at this operating point on these scenes rather than inferred
  from ``docs/research/nvdiffrast.md``: 3,109 f/s (h1) and 2,437 (h1hand) at batch 256, and an illegal
  memory access that kills the interpreter at 1024.

So the family is bound by intersecting a 295-398 k triangle articulated mesh -- 21-66 geoms with
overlapping bounds -- once per ray that reaches it, and reading these frames back off disk is
**1.30-1.92x faster** than doing that. What would move it is a cheaper primary intersection, not a
cheaper frame.

REPRODUCING THE REFERENCE NEEDS TWO SHIMS, AND THEY ARE APPLIED HERE
----------------------------------------------------------------------

HumanoidBench pins ``gymnasium==0.29.1`` / ``mujoco==3.1.6`` and MJWarp requires ``mujoco>=3.11``, so
the official env has to be constructed on an interpreter it was not written for. Two things break and
both are mechanical (``_hb_compat``): dm_control's vendored field table names 17 arrays that no longer
exist (``flex_xvert0`` and friends), and ``gymnasium>=1.0`` moved the camera onto
``MujocoRenderer.__init__`` while HumanoidBench still calls ``render(mode, camera_id, camera_name)``.
Neither reaches a pixel. A third does not have a shim and is why ``reach`` and ``push`` are not here:
they draw their goal with ``viewer.add_marker``, and gymnasium's legacy marker path calls
``mjv_initGeom`` with a scalar ``size=`` that mujoco 3.11 rejects, so those two tasks cannot render
their official frame on this stack at all.

WHY THESE CLASSES DO NOT REGISTER BY DEFAULT
----------------------------------------------

**The fidelity is measured and it is not what is blocking. The renderer is not bit-reproducible inside
a long conformance session, and that is a claim the registry requires and this family cannot yet
make.** Over six full runs of ``pytest tests/ -k HumanoidBench`` (578-741 tests each), 2-3 fail every
time and they are always the same two properties -- ``test_determinism`` (two renders of the same
states differ) and ``test_frame_i_is_independent_of_batch_size`` -- by 1-2/255, on a DIFFERENT cell
each run. What is known about it:

* it is **not** a property of one scene: it has landed on ``h1hand-slide``, ``h1hand-spoon``,
  ``h1hand-powerlift``, ``h1hand-sit_hard``, ``h1-slide`` and ``h1-highbar_simple``;
* it is **not** reproducible in a render loop: 40 consecutive renders of every cell in a fresh
  process are bit-identical, and so are **1,140 renders across all 19 cells with every render context
  alive at once** (``probe16_session_flake.py``);
* it is **not** the unwritten-world or the world-count effect: both were found, measured and fixed
  (see ``_upload`` and ``render``), and both fixes stuck -- the residual survived them;
* it is **not** a second render context at another resolution: 80 renders after building one are
  bit-identical (``probe17_two_contexts.py``);
* it is **not** confined to 1-2/255, and that is new. Measuring ``check_visual_fidelity`` eight times
  on ``h1-walk`` inside one process that also held three other cells, the first draw gave the declared
  **mean 2.8596 / SSIM 0.8491** and every later draw gave **6.33-9.91 / 0.7806-0.8019** — a
  whole-frame mean 3.5/255 worse and an SSIM BELOW this cell's declared floor of 0.80. The same
  process then reproduced those drifted values to four decimals on ``h1-stand``, which shares
  ``h1-walk``'s scene and state pool. The declaration itself is sound: **six processes on five nodes
  (008, 045, 049 twice, 054, 101) all return 2.8088 / 2.8692 / 2.8334 / 2.8596 for seeds 3 / 1 / 2 / 0**,
  inside the declared spread [2.81, 2.87], and a process that renders one seed four times returns the
  same number four times. So a fidelity number
  measured inside a long session here is not trustworthy, not only a determinism test — which is a
  bigger claim than "2-3 tests fail by 1-2/255" and the same reason for the opt-in;
* and it correlates with repeated env CONSTRUCTION rather than with repeated rendering, which is the
  one new handle on it. the frozen measurement record holds ONE env and one set of states,
  re-renders them ten times on both sides, and builds two further cells and their reference envs
  half-way through: both sides come back bit-identical to round 0 on every round, in two independent
  processes. The run that DID drift built a fresh env per draw. Whatever this is, re-rendering does
  not trigger it and re-constructing appears to.

So it needs something a full suite run does that none of those probes do, and it is not closed. Until
it is, importing this module raises unless ``STABLE_ENVS_HUMANOIDBENCH=1`` is set, so the repo's suite
stays green and ``stable_envs.envs.import_errors()`` says why in one line. Everything else here --
the model surgery, the measured ``FIDELITY`` table, the 49-scene survey behind it -- is finished work
and is what the opt-in gets you. ``docs/research/humanoidbench.md`` is the full investigation.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .._mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

# THE DEPENDENCY CHECK, at module scope and probing the FAILING PATH, for the reason ``fetch.py``
# gives: a lazy import inside ``render`` turns one missing wheel into a failure per test rather than
# one missing env. Four things are needed and none can be worked around from inside this file.
import importlib.util as _ilu

import mujoco as _mj

#: THE OPT-IN. Registration is gated because the renderer is not yet bit-reproducible across a long
#: multi-env session -- see the module docstring for the six-run measurement and for the three
#: hypotheses that were tested and closed. This is deliberately the same mechanism an optional
#: dependency uses (``ImportError`` at module scope, surfaced by ``stable_envs.envs.import_errors()``)
#: rather than a quietly weaker tier: the registry's contract includes "the same state gives the same
#: frame", and an env that cannot say that should not be in it.
if os.environ.get("STABLE_ENVS_HUMANOIDBENCH", "") != "1":  # pragma: no cover
    raise ImportError(
        "the HumanoidBench family is measured and complete on fidelity (17 cells, VISUAL tier, mean "
        "2.6-8.4/255, SSIM 0.845-0.931, 437x the official renderer at batch 64) but is NOT registered "
        "by default: mujoco_warp's batched render is not bit-reproducible inside a long conformance "
        "session, so 2-3 of ~660 tests fail every run on test_determinism and "
        "test_frame_i_is_independent_of_batch_size, by 1-2/255, on a different cell each time. Set "
        "STABLE_ENVS_HUMANOIDBENCH=1 to register it anyway. See docs/research/humanoidbench.md"
    )

_VENV = "/mnt/vast/home/randall.balestriero/venvs/se-hb"
_missing = [_m for _m in ("warp", "mujoco_warp", "humanoid_bench") if _ilu.find_spec(_m) is None]
_mjver = tuple(int(x) for x in str(_mj.__version__).split(".")[:2])
if _missing or _mjver < (3, 11):  # pragma: no cover
    raise ImportError(
        "the HumanoidBench family raytraces 21-50 H1 meshes with mujoco_warp, and reads its scenes "
        "from the humanoid_bench package: "
        + (f"missing {', '.join(_missing)}; " if _missing else "")
        + f"mujoco is {_mj.__version__} and mujoco_warp requires >= 3.11. Note that humanoid_bench "
        f"itself pins mujoco == 3.1.6 and gymnasium == 0.29.1, so it must be installed with "
        f"`pip install --no-deps -e .` beside a newer mujoco; `pip install stable-envs[humanoidbench]`"
        f" pulls the renderer half. The envs were measured in {_VENV}. See "
        f"docs/research/humanoidbench.md"
    )

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import mjwarp as MW
from ..spaces import Box, Dict

#: MuJoCo's defaults for an element with no material (``mjv_initGeom``).
DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION = 0.5, 0.5, 0.0
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
HIDDEN_GROUP = MW.HIDDEN_GROUP
VISIBLE_GROUPS = MW.DEFAULT_GROUPS
#: Worlds per MJWarp render context. Throughput saturates by 256 at 256 px (19,719 f/s against 20,295
#: at 1024) and a context owns ``nworld * H * W * 12`` bytes of output buffers, so chunking here bounds
#: memory at no measurable cost.
#:
#: 256 WAS MEASURED AND DECLINED, and the numbers are why. Here the chunk IS the pinned world count
#: (see ``render``), so raising it makes every batch pay 256 worlds of launch: at 256 px it is worth
#: 8-15 % at batch 1024 (``h1-walk`` 19,312 -> 22,310 f/s, ``h1-sit_hard`` 13,372 -> 14,564) and 14 %
#: at batch 256, and it costs **3.0-3.2x below that** -- batch 64 ``h1-walk`` 16,887 -> 5,447, batch 8
#: 2,089 -> 688, batch 1 247 -> 78. A library that is 12 % faster at 1024 and three times slower at 64
#: is a worse library, so this stays at 64 and the 12 % is left on the table deliberately.
MJW_CHUNK = 64

_MODEL_CACHE: dict = {}
_CTX_CACHE = _RES.BoundedCache(32, name="humanoidbench.ctx", on_evict=MW.release_cached_ctx)
#: One MJWarp render context per (cell, resolution), and the bound is deliberately larger than the
#: number of registered cells. A context owns ``nworld * H * W * 4`` bytes of output buffer -- 16.8 MB
#: at 64 worlds and 256 px -- plus the model and its mesh BVHs, so holding all 19 costs a few hundred
#: MB; evicting them instead costs a 1-3 s rebuild EVERY time the conformance suite moves between
#: envs, which is both slow and the only thing that correlated with a rare non-determinism (40 renders
#: of every cell in a fresh process are bit-identical; two renders inside a 19-env session were not).
_CTX_CACHE_MAX = 32
_POOL_CACHE: dict = {}


# --------------------------------------------------------------------------- the reference shims


def _hb_compat() -> list:
    """Make HumanoidBench constructible and renderable on ``mujoco>=3.11`` / ``gymnasium>=1.0``.

    Two independent version skews, neither of which reaches a pixel:

    * ``HumanoidEnv.__init__`` builds a dm_control named-indexer by walking a STATIC field table
      generated against mujoco 3.1 and calling ``getattr`` for every entry. Seventeen of those arrays
      were renamed or removed since (``flex_xvert0``, ``light_directional``, ``tex_rgb``, ``qM``, the
      sparse-matrix index arrays), so construction dies before any pixel exists.
      ``struct_indexer`` already SKIPS any field whose value is not an ndarray, so returning ``None``
      for exactly the absent names is the whole fix and the indexer simply has 17 fewer entries --
      none of which HumanoidBench reads.
    * ``gymnasium>=1.0`` moved the camera onto ``MujocoRenderer.__init__`` and left ``render`` taking
      only the mode; HumanoidBench still calls ``render(mode, camera_id, camera_name)``.
      ``MujocoEnv.__init__`` already forwarded ``camera_name``, so dropping the two extra arguments
      renders through exactly the same camera.

    Returns the model/data field names it shimmed, so a caller can report them instead of trusting
    this. Idempotent.
    """
    import humanoid_bench.dmc_deps.dmc_wrapper as W
    import mujoco
    from humanoid_bench.dmc_deps import dmc_sizes as S
    from humanoid_bench.tasks import Task

    probe_m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><geom size='.1'/><freejoint/></body></worldbody></mujoco>"
    )
    probe_d = mujoco.MjData(probe_m)
    absent = {
        "mjmodel": frozenset(f for f in S.array_sizes["mjmodel"] if not hasattr(probe_m, f)),
        "mjdata": frozenset(f for f in S.array_sizes["mjdata"] if not hasattr(probe_d, f)),
    }
    for cls, slot, key in ((W.MjModelWrapper, "_model", "mjmodel"), (W.MjDataWrapper, "_data", "mjdata")):
        if getattr(cls, "_se_mj311_shim", False):
            continue

        def getter(self, name, _slot=slot, _gone=absent[key]):
            inner = object.__getattribute__(self, _slot)
            if name in _gone and not hasattr(inner, name):
                return None
            return getattr(inner, name)

        cls.__getattr__ = getter
        cls._se_mj311_shim = True

    if not getattr(Task, "_se_gym1_shim", False):
        Task.render = lambda self: self._env.mujoco_renderer.render(self._env.render_mode)
        Task._se_gym1_shim = True
    return sorted(absent["mjmodel"] | absent["mjdata"])


def _asset_dir() -> str:
    import humanoid_bench

    return os.path.join(os.path.dirname(humanoid_bench.__file__), "assets")


def _scene_xml(robot: str, task: str) -> str:
    control = "torque" if robot in ("g1", "digit") else "pos"
    return os.path.join(_asset_dir(), "envs", f"{robot}_{control}_{task}.xml")


def _camera_name(task: str) -> str:
    """The model camera HumanoidBench renders this task through, read off the Task class."""
    from humanoid_bench.env import TASKS

    return TASKS[task].camera_name


# --------------------------------------------------------------------------- model surgery


class _Bundle:
    __slots__ = ("model", "data", "info", "pristine", "state_names", "nq", "robot", "task")


def _build_bundle(robot: str, task: str) -> _Bundle:
    """Compile the render-only model: resolved per-geom colours, flat floor, no collision margin.

    Every edit is either a colour-resolution correction (which makes MJWarp read what MuJoCo reads),
    a render-context requirement (``put_model`` validates collision configuration even when the caller
    only wants to draw), or THE ONE DELIBERATE APPROXIMATION this tier makes -- the floor's texture.
    """
    import mujoco

    key = (robot, task)
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit

    path = _scene_xml(robot, task)
    spec = mujoco.MjSpec.from_file(path)
    base = spec.compile()
    for i in range(base.ngeom):
        spec.add_material(name=f"se:mat{i}")
    m = spec.compile()
    assert m.ngeom == base.ngeom and m.nq == base.nq, f"{robot}-{task}: surgery changed the model"

    # MJWarp resolves a geom's colour from its MATERIAL whenever it has one; MuJoCo prefers the geom's
    # own ``rgba`` unless that is the compiler default. H1's ``black``/``white`` materials are worn by
    # geoms that also set ``rgba``, so uncorrected the robot comes out one colour. One material per
    # geom carrying the RESOLVED colour and the original shading terms leaves the shading path alone.
    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")
    floor, robot_geoms, textured_prims = -1, [], []
    matname = lambda i: (mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_MATERIAL, i) or "") if i >= 0 else ""
    for g in range(m.ngeom):
        k = mat0 + g
        old = int(base.geom_matid[g])
        rgba = np.asarray(base.geom_rgba[g], np.float64)
        if old >= 0 and np.allclose(rgba, DEFAULT_RGBA):
            rgba = np.asarray(base.mat_rgba[old], np.float64)
        if old >= 0:
            m.mat_specular[k] = base.mat_specular[old]
            m.mat_shininess[k] = base.mat_shininess[old]
            m.mat_emission[k] = base.mat_emission[old]
            m.mat_reflectance[k] = base.mat_reflectance[old]
            m.mat_texid[k] = base.mat_texid[old]
            m.mat_texrepeat[k] = base.mat_texrepeat[old]
            m.mat_texuniform[k] = base.mat_texuniform[old]
        else:
            m.mat_specular[k] = DEFAULT_SPECULAR
            m.mat_shininess[k] = DEFAULT_SHININESS
            m.mat_emission[k] = DEFAULT_EMISSION
            m.mat_reflectance[k] = 0.0
            m.mat_texid[k] = -1
        m.mat_rgba[k] = rgba
        m.geom_rgba[g] = rgba
        m.geom_matid[g] = k
        if int(base.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            floor = g
        if matname(old) == "black" and int(base.body_weldid[base.geom_bodyid[g]]) != 0:
            robot_geoms.append(g)
        if (
            int(base.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_MESH))
            and int(m.geom_group[g]) in VISIBLE_GROUPS
        ):
            t = m.mat_texid[k]
            t = int(np.max(t)) if np.ndim(t) else int(t)
            if t >= 0:
                textured_prims.append(g)
    assert floor >= 0, f"{robot}-{task}: no floor plane in the compiled model"
    assert robot_geoms, f"{robot}-{task}: no geom wears the H1 'black' material"

    # THE ONE DELIBERATE APPROXIMATION. The floor's texture is a 2 cm grid line every 2 m and MJWarp
    # point-samples it into dither; its DOMINANT texel is the flat field the grid sits on. Expressed
    # through ``mat_rgba`` with the texture removed rather than by rewriting texels, because (a) the
    # no-texture path is measurably closer than a constant-texture one (floor interior 0.9957 against
    # 0.9709, probe3 L3 vs L2) and (b) ``mat_rgba`` is a per-world array, so ``floor.color`` can then
    # be a real per-environment knob while a texture could not be.
    ftex = m.mat_texid[mat0 + floor]
    ftex = int(np.max(ftex)) if np.ndim(ftex) else int(ftex)
    if ftex >= 0:
        w, h, c = int(m.tex_width[ftex]), int(m.tex_height[ftex]), int(m.tex_nchannel[ftex])
        adr = int(m.tex_adr[ftex])
        px = np.asarray(m.tex_data[adr : adr + w * h * c], np.uint8).reshape(-1, c)
        vals, cnt = np.unique(px, axis=0, return_counts=True)
        dominant = np.asarray(vals[int(cnt.argmax())], np.float64)[:3] / 255.0
        if np.ndim(m.mat_texid[mat0 + floor]):
            m.mat_texid[mat0 + floor, :] = -1
        else:
            m.mat_texid[mat0 + floor] = -1
        m.mat_rgba[mat0 + floor, :3] = dominant
        m.geom_rgba[floor, :3] = dominant
    floor_color = np.array(m.mat_rgba[mat0 + floor, :3], copy=True)

    # MJWarp never reads alpha, so anything MuJoCo draws as invisible has to leave the visible groups
    # before ``create_render_context`` reads the geometry. Sites are hidden for the same reason on the
    # reference side of the ledger: MJWarp draws none at all.
    prepared = MW.prepare_model(m, VISIBLE_GROUPS)
    # Rendering-only model: collision margin reaches no pixel and ``put_model`` refuses a non-zero one
    # on BOX/MESH pairs under MULTICCD.
    m.geom_margin[:] = 0.0
    m.pair_margin[:] = 0.0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 4 * HumanoidBenchBase.default_size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 4 * HumanoidBenchBase.default_size)

    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, _camera_name(task))
    assert cam >= 0, f"{robot}-{task}: camera {_camera_name(task)!r} is not in the model"

    names = []
    for jj in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jj) or f"joint{jj}"
        t = int(m.jnt_type[jj])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            names += [f"{nm}_{s}" for s in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        elif t == int(mujoco.mjtJoint.mjJNT_BALL):
            names += [f"{nm}_{s}" for s in ("qw", "qx", "qy", "qz")]
        else:
            names.append(nm)
    assert len(names) == int(m.nq), f"{robot}-{task}: derived {len(names)} state names for nq={m.nq}"

    b = _Bundle()
    b.model, b.data = m, mujoco.MjData(m)
    b.nq, b.robot, b.task = int(m.nq), robot, task
    b.state_names = tuple(names)
    b.info = dict(
        cam=cam,
        mat0=mat0,
        floor=floor,
        robot=robot_geoms,
        textured_prims=textured_prims,
        hidden=len(prepared.hidden_geoms) + len(prepared.hidden_sites),
        floor_color=floor_color,
        robot_color=np.array(m.mat_rgba[mat0 + robot_geoms[0], :3], copy=True),
    )
    b.pristine = dict(mat_rgba=np.array(m.mat_rgba, copy=True), light_diffuse=np.array(m.light_diffuse, copy=True))
    _MODEL_CACHE[key] = b
    return b


# --------------------------------------------------------------------------- MJWarp context


class _Ctx:
    __slots__ = ("wm", "wd", "rc", "bgra", "graph", "nworld", "res", "wp", "mjw", "needs_com")


def _context(robot: str, task: str, res: int, nworld: int) -> _Ctx:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    key = (robot, task, int(res), int(nworld))
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
    b = _build_bundle(robot, task)
    m = b.model
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    c = _Ctx()
    c.wp, c.mjw, c.nworld, c.res = wp, mjw, int(nworld), int(res)
    c.needs_com = MW.needs_com_pos(m)
    batched = {k: int(nworld) for k in ("mat_rgba", "light_diffuse", "cam_poscom0", "cam_mat0")}
    c.wm = mjw.put_model(m, batch_sizes=batched)
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # THIS FAMILY DOES NOT APPLY ``gl_rounding``, AND THAT IS A MEASUREMENT RATHER THAN AN OVERSIGHT.
    #
    # MJWarp packs float->uint8 with C's ``int()``, which truncates, where the GL pipeline MuJoCo
    # renders through rounds; ``MW.gl_rounding`` corrects it and on the PARITY families it is the
    # largest single win available (Fetch interior 0.9923 -> 0.9988). Here it moves the other way.
    # Measured on one interpreter and one state draw with the frozen measurement record, h1-walk, 64
    # states, 256 px: mean |diff| **2.860 truncating vs 3.500 rounding** against a declared ceiling of
    # 3.4, with within-8 and SSIM flat (0.9338/0.8491 vs 0.9331/0.8487).
    #
    # The mechanism is not the quantiser. Rounding raises 63.8% of this family's channels by one level
    # and every one of those moves AWAY from the official value, which says the frame is already
    # biased high -- and it is, by the one approximation the tier declares: the floor is 66-72% of a
    # locomotion frame and is drawn with its texture's dominant texel because ``mujoco_warp`` cannot
    # mipmap, while MuJoCo minifies a checker whose 0.7 edge lines pull its average below that texel.
    # Truncating was silently absorbing part of that bias. Correcting the quantiser without correcting
    # the floor would trade a stated approximation for an unstated regression, so the rule stays off
    # here until the floor is right, and the pixels are unchanged from every number this module
    # declares.
    # THE CONTEXT AND THE CAPTURE ARE BOTH INSIDE ``specialised_lights``, and the capture is the half
    # that matters: ``mujoco_warp.render`` DEFINES its megakernel inside itself, so the source is
    # generated afresh on every call and a graph captured outside this block would replay the stock
    # kernel forever. This scene has one light and a headlight; the headlight is called with a literal
    # ``lightcastshadow=False``, so stock ``mujoco_warp`` compiles two any-hit BVH traversals and runs
    # one. Deleting the dead one is worth 1.000-1.001x here and is byte-identical over 64 states on
    # both cells measured: this family is bound by the robot's
    # mesh intersections, not by the register budget, so there is nothing for it to recover. It is on
    # because it is exact and free, and so that the three families share one code path.
    from ..render import mjwarp_lightspec as LS

    with LS.specialised_lights():
        c.rc = MW.create_render_context(
            m,
            nworld=int(nworld),
            cam_res=(int(res), int(res)),
            render_rgb=True,
            render_seg=False,
            cam_active=[i == b.info["cam"] for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS),
            use_textures=True,
            use_shadows=True,
            render_skybox=True,
            enable_specular=True,
        )
        _frame(c)
        wp.synchronize()
        # A VIEW of the warp buffer. ``[..., [2,1,0]]`` here would be advanced indexing, i.e. a COPY,
        # and the object would hold one frozen frame while every later render wrote into the buffer
        # behind it.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
        with wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    _CTX_CACHE.put(key, c)
    return c


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    if c.needs_com:
        # EVERY HumanoidBench camera but ``cam_kitchen`` is TRACKCOM, and a TRACKCOM camera's position
        # is ``subtree_com[body] + cam_poscom0``. ``kinematics`` does not write ``subtree_com``, so
        # without this the camera never leaves its ``put_data`` pose -- silently, with a plausible
        # picture. Measured: with it, ``cam_xpos`` agrees with MuJoCo's to 1.8e-07.
        c.mjw.com_pos(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


# --------------------------------------------------------------------------- the env


class HumanoidBenchBase(Env):
    """Base for the registered cells. ``ROBOT`` and ``TASK`` are set by the generated subclasses."""

    family = "humanoidbench"

    ROBOT: str = ""
    TASK: str = ""
    default_size = 256  # HumanoidBench's own default width/height
    fidelity_tier = "visual"
    parity_interior_min = None

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int = 1,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        b = _build_bundle(self.ROBOT, self.TASK)
        self.state_dim = b.nq
        self.state_names = b.state_names
        # Public state: the packed row IS ``qpos`` and nothing else, so the public
        # State is one key over it, assigned per instance because ``nq`` is a property of the
        # compiled model. Nothing is hidden (every column is covered, no offset is ``None``) and
        # there is no ``_live_state``: the packed row is this family's live state.
        self.state_layout = ((("qpos",), self.state_dim, 0),)
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        self._b = b

    # ---- variation ------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are the scene HumanoidBench compiles.

        Colours are float Boxes rather than ``RGBBox`` because MuJoCo's material colours are floats
        (the floor is 0.898039) and a uint8 channel cannot hold one, so a uint8 default would move the
        floor by a level and make "our defaults reproduce our declared appearance" false.

        The **skybox is deliberately not a knob**: it is a texture, ``mujoco_warp`` uploads textures
        when the render context is built, and a per-value skybox would mean a render context per
        value -- which is the one thing this repo's MJWarp rule forbids.
        """
        b = _build_bundle(self.ROBOT, self.TASK)
        return Dict(
            {
                "floor": Dict(
                    {"color": Box(low=0.0, high=1.0, shape=(3,), init_value=b.info["floor_color"].tolist())}
                ),
                "robot": Dict(
                    {"color": Box(low=0.0, high=1.0, shape=(3,), init_value=b.info["robot_color"].tolist())}
                ),
                "light": Dict({"diffuse": Box(low=0.2, high=2.0, shape=(1,), init_value=[1.0])}),
                "camera": Dict(
                    {
                        "distance_scale": Box(low=0.6, high=1.6, shape=(1,), init_value=[1.0]),
                        "azimuth_delta": Box(low=-40.0, high=40.0, shape=(1,), init_value=[0.0]),
                        "elevation_delta": Box(low=-20.0, high=20.0, shape=(1,), init_value=[0.0]),
                    }
                ),
            }
        )

    # ---- render ---------------------------------------------------------------------------

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
        """``(...,nq)`` -> ``(...,H,W,3)`` uint8 on device. Style may differ per environment.

        One MJWarp launch per chunk for the whole batch: colours, light and camera are per-world model
        fields, so nothing about a per-environment appearance splits it.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        # HOISTED OUT OF THE CHUNK LOOP. The camera pose is a function of the style and the batch row,
        # not of the render context, so computing it per chunk bought nothing and cost sixteen device
        # synchronisations (``.cpu()`` on the style key) and sixteen host sorts per render -- 5.2 ms of
        # a 62.7 ms batch, measured with the frozen measurement record.
        campos, cammat = self._camera(sty, 0, B)
        # THE WORLD COUNT IS FIXED, NOT SIZED TO THE CHUNK, and that is a correctness requirement
        # rather than a simplification. MJWarp generates and compiles a megakernel per render
        # context, and the one built for ``nworld = 1`` does not produce the same bytes as the one
        # built for 4, 8 or 64: measured on HumanoidBenchH1HighbarSimple, the same state rendered
        # alone differed from the same state inside a batch by 1/255 on one row, while batches of
        # 3, 7, 8, 64 and 1024 agreed with each other exactly. A loader's last partial batch would
        # then produce different pixels from every batch before it. One context per (cell,
        # resolution) makes ``check_batch_invariance`` structural; the cost is that a batch of 1
        # launches 64 worlds and renders at ~250 f/s instead of ~830.
        c = _context(self.ROBOT, self.TASK, res, MJW_CHUNK)
        # ONE STREAM, NOT THREE SYNCHRONISATIONS. Every ordering this loop needs is an ordering
        # BETWEEN two GPU operations, and a stream already provides those: ``_upload`` writes the
        # per-world model arrays through torch, the graph reads them, and the output copy reads the
        # buffer the next chunk's launch will overwrite. Enqueuing all three on the SAME stream makes
        # each wait for the last without the host waiting for any of them, which is the whole
        # difference -- the previous form (``wp.synchronize`` / ``torch.cuda.synchronize`` /
        # ``wp.synchronize`` per chunk) blocked the host 32 times per render and left the GPU idle
        # while Python issued the next chunk. Measured at 256 px / batch 1024 with
        # the frozen measurement record: 49.8 ms of a 54.5 ms render was inside ``wp.synchronize``,
        # against 46.4 ms of actual kernel.
        #
        # It is safe for exactly the reason the syncs were: the hazard is a previous launch still
        # READING the per-world arrays while torch overwrites them, and the context is shared by every
        # env instance of this class, so that previous launch is not necessarily this object's. A
        # warp graph launched on torch's current stream is ordered against every torch op on that
        # stream, so the hazard cannot occur as long as nothing else drives this context from another
        # stream -- and nothing does; the render path is the only writer. Nothing about the arithmetic
        # moves: a stream is an ordering, not a schedule of floats.
        stream = c.wp.stream_from_torch(torch.cuda.current_stream())
        for lo in range(0, B, MJW_CHUNK):
            hi = min(lo + MJW_CHUNK, B)
            n = hi - lo
            self._upload(c, s[lo:hi], sty, lo, hi, campos[lo:hi], cammat[lo:hi])
            c.wp.capture_launch(c.graph, stream=stream)
            rgb = c.bgra[:n][..., [2, 1, 0]]
            if ss == 1:
                buf[lo:hi] = rgb
            else:
                buf[lo:hi] = (
                    rgb.to(torch.float32)
                    .reshape(n, size, ss, size, ss, 3)
                    .mean((2, 4))
                    .round_()
                    .clamp_(0, 255)
                    .to(torch.uint8)
                )
        return buf.view(*lead, size, size, 3) if lead else buf

    def _upload(
        self, c: _Ctx, s: torch.Tensor, sty, lo: int, hi: int, campos: torch.Tensor, cammat: torch.Tensor
    ) -> None:
        """Write EVERY world of the context, not only the ``n`` a caller asked for.

        The unused tail is filled with a copy of the last requested row, and that is a correctness
        requirement rather than tidiness. ``mujoco_warp.refit_bvh`` rebuilds ONE acceleration
        structure over every world in the launch, so a world whose ``qpos`` was never written
        contributes whatever the memory pool last left there to the tree's bounds. It reaches no pixel
        directly -- worlds are independent in the shader, measured at 0/255
        (``probe12_worldbleed.py``) -- but it moves the traversal, and on a scene with two nearly
        coincident surfaces (``slide``'s ramp on the floor, ``highbar``'s posts) a near-tie can then be
        broken the other way. That presented as a 1-2/255 flake that only ever appeared inside a long
        multi-env session and never in 40 consecutive renders of a fresh process, i.e. as
        ``test_determinism`` and ``test_frame_i_is_independent_of_batch_size`` failing on a different
        cell each run. Filling the tail costs one broadcast write and makes the launch a function of
        the requested states alone.

        NECESSARY BUT NOT SUFFICIENT, and the residue is now measured rather than suspected.
        the frozen measurement record fills EVERY world of one launch with the IDENTICAL ``qpos`` --
        no tail, no padding, nothing left over from a previous call -- and compares each world against
        world 0. Seven cells give 0/255 on every pixel; ``h1-balance_simple`` gives **2 pixels at
        1/255, in slots 21 and 42 of 64**. Meta-World and Fetch give 0 on the same probe. So a frame
        here is a function of the state AND OF THE WORLD SLOT IT OCCUPIES, which is exactly what makes
        a batch-size change move a pixel: the same state sits at a different index.

        The mechanism is upstream and structural. ``mujoco_warp`` builds ONE scene BVH over
        ``nworld * ngeom`` primitives with a per-world root, so a world's subtree holds only its own
        geometry but its internal SHAPE is decided by a build over every world at once, and
        closest-hit is settled by a strict ``<`` on the ray parameter in visit order. Two exactly
        coincident surfaces are a tie, and a tie is broken by whichever node the traversal reached
        first. A deterministic tie-break (lowest geom id at equal ``t``) would close it and belongs in
        ``mujoco_warp``, not here. Pinning the world count (see ``render``) keeps the slot map a
        function of the batch alone, which is why the flake is rare rather than constant.
        """
        info, n = self._b.info, hi - lo
        nw = c.nworld
        wp = c.wp
        pad = lambda t: t if n >= nw else torch.cat([t, t[-1:].expand(nw - n, *t.shape[1:])], 0)

        wp.to_torch(c.wd.qpos)[:] = pad(s)
        mat = wp.to_torch(c.wm.mat_rgba)
        m0 = info["mat0"]
        mat[:, m0 + info["floor"], :3] = pad(sty["floor.color"][lo:hi])
        # ONE indexed write for the robot's materials rather than one kernel per material: h1hand wears
        # 47 of them, so the loop issued 47 broadcasts of the same colour per chunk.
        mat[:, self._idx("robot.mats", m0 + np.asarray(info["robot"], np.int64)), :3] = pad(
            sty["robot.color"][lo:hi]
        ).unsqueeze(1)
        wp.to_torch(c.wm.light_diffuse)[:] = self._light0().unsqueeze(0) * pad(
            sty["light.diffuse"][lo:hi].reshape(n, 1, 1)
        )
        wp.to_torch(c.wm.cam_poscom0)[:, info["cam"]] = pad(campos)
        wp.to_torch(c.wm.cam_mat0)[:, info["cam"]] = pad(cammat)

    def _idx(self, key: str, values) -> torch.Tensor:
        """A device index tensor, built once per env. ``as_tensor`` per chunk is a host copy each."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = torch.as_tensor(np.asarray(values, np.int64), device=self.device)
        return hit

    def _light0(self) -> torch.Tensor:
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("light0")
        if hit is None:
            hit = cache["light0"] = torch.as_tensor(
                self._b.pristine["light_diffuse"], device=self.device, dtype=torch.float32
            )
        return hit

    def _camera(self, sty, lo: int, hi: int):
        """Per-environment TRACKCOM camera, expressed the way ``mujoco_warp`` reads one.

        For ``mjCAMLIGHT_TRACKCOM`` MJWarp computes ``cam_xpos = subtree_com[body] + cam_poscom0`` and
        ``cam_xmat = cam_mat0`` — both per-world arrays — so the knob is an ORBIT of the offset about
        the tracked centre of mass, with the orientation carried along. Rotating only the offset would
        move the camera and leave it aiming where it used to.
        """
        b, info = self._b, self._b.info
        m = b.model
        base_pos = np.asarray(m.cam_poscom0[info["cam"]], np.float64)
        base_mat = np.asarray(m.cam_mat0[info["cam"]], np.float64).reshape(3, 3)
        key = (
            torch.stack(
                [
                    sty[p][lo:hi].reshape(-1)
                    for p in ("camera.distance_scale", "camera.azimuth_delta", "camera.elevation_delta")
                ],
                1,
            )
            .cpu()
            .numpy()
        )
        uniq, inv = np.unique(np.round(key.astype(np.float64), 6), axis=0, return_inverse=True)
        poses, mats = [], []
        for scale, az, el in uniq:
            a = np.deg2rad(float(az))
            Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1.0]])
            right = Rz @ base_mat[:, 0]  # the camera's own x axis, after azimuth
            e = np.deg2rad(float(el))
            k = right / max(np.linalg.norm(right), 1e-12)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Re = np.eye(3) + np.sin(e) * K + (1 - np.cos(e)) * (K @ K)
            R = Re @ Rz
            poses.append(R @ base_pos * float(scale))
            mats.append(R @ base_mat)
        pos = torch.as_tensor(np.stack(poses)[inv], device=self.device, dtype=torch.float32)
        # ``cam_mat0`` is ``wp.array2d[wp.mat33]``, so its torch view is (nworld, ncam, 3, 3) and a
        # flat (n, 9) assignment raises rather than silently transposing. Kept as (n, 3, 3).
        mat = torch.as_tensor(np.stack(mats)[inv], device=self.device, dtype=torch.float32)
        return pos, mat

    # ---- states ---------------------------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Drawn from a pool the OFFICIAL env produced: reset, then 0-60 random actions.

        Reset alone is the wrong distribution here for the same reason it is in Fetch: HumanoidBench's
        ``reset_model`` restores one keyframe and jitters ``qpos`` by ``randomness = 0.01``, so every
        reset frame is the same standing pose and a number computed from them says nothing about the
        joints that are most of the picture.
        """
        rng = rng or self.rng
        pool = _state_pool(self.ROBOT, self.TASK, self.state_dim)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    # ---- reference bridge -----------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 256):
        """A FRESH official env. Never closed: releasing an EGL context corrupts every other live one."""
        import gymnasium
        import humanoid_bench  # noqa: F401  (registers the ids)

        _hb_compat()
        np.random.seed(0)
        env = gymnasium.make(f"{cls.ROBOT}-{cls.TASK}-v0", render_mode="rgb_array", width=int(size), height=int(size))
        u = env.unwrapped
        u.model.vis.global_.offwidth = max(int(u.model.vis.global_.offwidth), int(size))
        u.model.vis.global_.offheight = max(int(u.model.vis.global_.offheight), int(size))
        env.reset(seed=0)
        u.render()
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for ``qpos``, with the round-trip asserted before anything is recorded.

        ``mj_step`` leaves ``data.xpos`` one substep behind ``data.qpos``, so a frame is only a
        function of the recorded state after an explicit ``mj_forward``; and ``mj_kinematics``
        renormalises a free joint's quaternion in place, which is the only thing the assert tolerates.
        """
        import mujoco

        u = env.unwrapped
        m, d = u.model, u.data
        nq = int(m.nq)
        out = []
        for st in np.asarray(states, np.float64):
            d.qpos[:] = st[:nq]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            err = float(np.abs(np.asarray(d.qpos) - st[:nq]).max())
            if err > 1e-5:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state (max "
                    f"{err:.3e}); every number computed from it would compare two different scenes"
                )
            u.mujoco_renderer.viewer.make_context_current()
            out.append(np.ascontiguousarray(u.render()))
        return np.stack(out)


def _state_pool(robot: str, task: str, nq: int, n: int = 512) -> np.ndarray:
    """``(n, nq)`` from the official env: reset, 0-60 random actions, read ``qpos``.

    SEEDS BOTH GENERATORS. ``Task.reset_model`` draws its per-episode placement from the
    process-global ``np.random`` (``np.random.uniform``), not from ``self.np_random``, so
    ``reset(seed=k)`` alone does not reproduce a pool -- the same shape as Meta-World's
    ``_get_state_rand_vec``. ``action_space.seed`` is separate again.
    """
    hit = _POOL_CACHE.get((robot, task))
    if hit is not None:
        return hit
    import gymnasium
    import humanoid_bench  # noqa: F401

    _hb_compat()
    np.random.seed(0)
    env = gymnasium.make(f"{robot}-{task}-v0", render_mode=None)
    env.action_space.seed(0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    lo = np.asarray(env.action_space.low, np.float64)
    hi = np.asarray(env.action_space.high, np.float64)
    out = np.empty((int(n), int(nq)), np.float32)
    for i in range(int(n)):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(int(rng.integers(0, 61))):
            _, _, term, trunc, _ = env.step(rng.uniform(lo, hi).astype(np.float32))
            if term or trunc:
                env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        out[i] = np.asarray(u.data.qpos, np.float64)
    env.close()
    _POOL_CACHE[(robot, task)] = out
    return out


# --------------------------------------------------------------------------- the registered cells

_AT = (
    "256 px, 64 states x 4 independent draws from sample_states, one NVIDIA H200, "
    "mujoco 3.11.0 / mujoco-warp 3.11.0 / warp 1.16.0, humanoid_bench 0.2 (~/venvs/se-hb)"
)

VISUAL_NOTE = (
    "THREE RESIDUALS, all structural to the backend rather than to any one scene, in the order they "
    "cost pixels. (1) THE FLOOR'S GRID IS NOT DRAWN. HumanoidBench's floor is a 40 m plane wearing a "
    "degenerate checker -- 100x100 texels, 9,604 at 229 and 396 at 178, i.e. a flat grey with a "
    "one-texel mark='edge' border tiled 20x20, so a 2 cm line every 2 m -- and the floor is 66-72 % of "
    "a locomotion frame. MuJoCo mipmaps that line; mujoco_warp builds every texture at one mip level "
    "and samples with no level of detail, so drawn as-is the floor comes out DITHERED (interior 0.877, "
    "SSIM 0.639 on h1hand-walk). This module replaces the floor's texture with its DOMINANT texel, "
    "which is measurably closer to the official frame on every metric at once (interior 0.9158 -> "
    "0.9503, mean 4.13 -> 2.79/255, within-8 0.901 -> 0.935, SSIM 0.628 -> 0.851 on h1-walk) and is "
    "the same class of trade as render.mjwarp.bake_primitive_textures. Supersampling is NOT the fix "
    "and that is measured: 1x -> 4x moves interior agreement 0.877 -> 0.682, the signature of a "
    "minification residual rather than a coverage one. (2) A LARGE TILTED FLAT FACE IS 2-5 LEVELS "
    "DARK. On h1-slide's ramp -- 33 % of that frame -- the signed difference is {-5: 14 %, -4: 11 %, "
    "-3: 39 %, -2: 34 %}, one-sided, and it survives specular=0 on both sides and offsamples=1 on the "
    "reference, so it is a diffuse constant rather than a highlight or coverage. It costs interior "
    "exactness (0.62 on slide) and almost nothing else (within-8 0.979). (3) TEXTURED PRIMITIVES, on "
    "the manipulation cells only: mujoco_warp's sample_texture derives UVs for PLANE and MESH only, so "
    "a textured BOX or CYLINDER takes texel (0,0) -- 5 geoms on insert, 10 on bookshelf, 15 on spoon, "
    "21 on cabinet. That is the RoboCasa mechanism and it is why those cells carry a higher mean at a "
    "higher SSIM. (4) THE FLOAT->UINT8 QUANTISATION, which is the one correction this family does NOT "
    "take: mujoco_warp truncates where GL rounds, and correcting it raises 63.8% of the channels by a "
    "level, every one of them away from the official value, for mean 2.860 -> 3.500 on h1-walk. That "
    "is the dominant-texel floor of (2) showing through -- the frame is biased high and truncating "
    "absorbs part of it -- and it is recorded at the call site in ``_context`` rather than taken. What "
    "is NOT in these numbers because it was found and fixed: com_pos in the frame loop (without it a "
    "TRACKCOM camera never leaves its put_data pose), one material per geom carrying MuJoCo's "
    "resolved colour, and the zero-alpha pre-pass. WHAT THIS TIER DOES NOT LICENSE: training here and "
    "evaluating on official HumanoidBench pixels."
)

#: MEASURED per cell. Keys are ``(robot, task)``; ``mean_abs``/``within8_frac``/``ssim`` are the
#: conformance suite's own numbers on this env's own state distribution, and the three ``*_max`` /
#: ``*_min`` thresholds are set clear of the SPREAD over independent draws, not of one draw.
FIDELITY: dict[tuple[str, str], dict] = {
    ("h1", "balance_simple"): {
        "mean_abs": 5.27,
        "within8_frac": 0.910,
        "ssim": 0.862,
        "interior_mean_abs": 1.77,
        "interior_exact_frac": 0.931,
        "draw_spread": {"mean_abs": [5.20, 5.46], "within8_frac": [0.910, 0.911], "ssim": [0.853, 0.862]},
        "mean_abs_max": 6.5,
        "within8_min": 0.87,
        "ssim_min": 0.81,
        "measured_at": _AT,
    },
    ("h1", "run"): {
        "mean_abs": 2.86,
        "within8_frac": 0.934,
        "ssim": 0.849,
        "interior_mean_abs": 0.34,
        "interior_exact_frac": 0.932,
        "draw_spread": {"mean_abs": [2.81, 2.87], "within8_frac": [0.934, 0.935], "ssim": [0.845, 0.850]},
        "mean_abs_max": 3.4,
        "within8_min": 0.89,
        "ssim_min": 0.8,
        "measured_at": _AT,
    },
    ("h1", "sit_hard"): {
        "mean_abs": 3.46,
        "within8_frac": 0.919,
        "ssim": 0.868,
        "interior_mean_abs": 0.56,
        "interior_exact_frac": 0.937,
        "draw_spread": {"mean_abs": [3.40, 3.46], "within8_frac": [0.919, 0.920], "ssim": [0.859, 0.868]},
        "mean_abs_max": 4.2,
        "within8_min": 0.88,
        "ssim_min": 0.82,
        "measured_at": _AT,
    },
    ("h1", "sit_simple"): {
        "mean_abs": 3.44,
        "within8_frac": 0.917,
        "ssim": 0.851,
        "interior_mean_abs": 0.49,
        "interior_exact_frac": 0.942,
        "draw_spread": {"mean_abs": [3.44, 3.50], "within8_frac": [0.916, 0.917], "ssim": [0.843, 0.855]},
        "mean_abs_max": 4.2,
        "within8_min": 0.88,
        "ssim_min": 0.8,
        "measured_at": _AT,
    },
    ("h1", "slide"): {
        "mean_abs": 3.35,
        "within8_frac": 0.954,
        "ssim": 0.931,
        "interior_mean_abs": 1.51,
        "interior_exact_frac": 0.620,
        "draw_spread": {"mean_abs": [3.33, 3.39], "within8_frac": [0.952, 0.954], "ssim": [0.919, 0.931]},
        "mean_abs_max": 4.1,
        "within8_min": 0.91,
        "ssim_min": 0.88,
        "measured_at": _AT,
    },
    ("h1", "stair"): {
        "mean_abs": 2.62,
        "within8_frac": 0.958,
        "ssim": 0.929,
        "interior_mean_abs": 0.54,
        "interior_exact_frac": 0.853,
        "draw_spread": {"mean_abs": [2.61, 2.67], "within8_frac": [0.956, 0.958], "ssim": [0.928, 0.932]},
        "mean_abs_max": 3.2,
        "within8_min": 0.92,
        "ssim_min": 0.89,
        "measured_at": _AT,
    },
    ("h1", "stand"): {
        "mean_abs": 2.86,
        "within8_frac": 0.934,
        "ssim": 0.849,
        "interior_mean_abs": 0.34,
        "interior_exact_frac": 0.932,
        "draw_spread": {"mean_abs": [2.81, 2.87], "within8_frac": [0.934, 0.935], "ssim": [0.845, 0.850]},
        "mean_abs_max": 3.4,
        "within8_min": 0.89,
        "ssim_min": 0.8,
        "measured_at": _AT,
    },
    ("h1", "walk"): {
        "mean_abs": 2.86,
        "within8_frac": 0.934,
        "ssim": 0.849,
        "interior_mean_abs": 0.34,
        "interior_exact_frac": 0.932,
        "draw_spread": {"mean_abs": [2.81, 2.87], "within8_frac": [0.934, 0.935], "ssim": [0.845, 0.850]},
        "mean_abs_max": 3.4,
        "within8_min": 0.89,
        "ssim_min": 0.8,
        "measured_at": _AT,
    },
    ("h1hand", "balance_simple"): {
        "mean_abs": 5.40,
        "within8_frac": 0.908,
        "ssim": 0.861,
        "interior_mean_abs": 1.71,
        "interior_exact_frac": 0.930,
        "draw_spread": {"mean_abs": [5.13, 5.40], "within8_frac": [0.908, 0.909], "ssim": [0.861, 0.866]},
        "mean_abs_max": 6.5,
        "within8_min": 0.87,
        "ssim_min": 0.82,
        "measured_at": _AT,
    },
    ("h1hand", "run"): {
        "mean_abs": 3.00,
        "within8_frac": 0.932,
        "ssim": 0.852,
        "interior_mean_abs": 0.34,
        "interior_exact_frac": 0.931,
        "draw_spread": {"mean_abs": [3.00, 3.01], "within8_frac": [0.931, 0.932], "ssim": [0.846, 0.852]},
        "mean_abs_max": 3.6,
        "within8_min": 0.89,
        "ssim_min": 0.81,
        "measured_at": _AT,
    },
    ("h1hand", "sit_hard"): {
        "mean_abs": 3.56,
        "within8_frac": 0.918,
        "ssim": 0.868,
        "interior_mean_abs": 0.54,
        "interior_exact_frac": 0.943,
        "draw_spread": {"mean_abs": [3.56, 3.60], "within8_frac": [0.917, 0.918], "ssim": [0.858, 0.868]},
        "mean_abs_max": 4.3,
        "within8_min": 0.88,
        "ssim_min": 0.82,
        "measured_at": _AT,
    },
    ("h1hand", "sit_simple"): {
        "mean_abs": 3.61,
        "within8_frac": 0.915,
        "ssim": 0.859,
        "interior_mean_abs": 0.50,
        "interior_exact_frac": 0.937,
        "draw_spread": {"mean_abs": [3.61, 3.66], "within8_frac": [0.914, 0.915], "ssim": [0.851, 0.860]},
        "mean_abs_max": 4.4,
        "within8_min": 0.87,
        "ssim_min": 0.81,
        "measured_at": _AT,
    },
    ("h1hand", "slide"): {
        "mean_abs": 3.69,
        "within8_frac": 0.946,
        "ssim": 0.906,
        "interior_mean_abs": 1.84,
        "interior_exact_frac": 0.607,
        "draw_spread": {"mean_abs": [3.50, 3.69], "within8_frac": [0.946, 0.951], "ssim": [0.906, 0.926]},
        "mean_abs_max": 4.4,
        "within8_min": 0.91,
        "ssim_min": 0.87,
        "measured_at": _AT,
    },
    ("h1hand", "spoon"): {
        "mean_abs": 8.40,
        "within8_frac": 0.795,
        "ssim": 0.899,
        "interior_mean_abs": 4.21,
        "interior_exact_frac": 0.923,
        "draw_spread": {"mean_abs": [8.35, 8.40], "within8_frac": [0.795, 0.796], "ssim": [0.898, 0.901]},
        "mean_abs_max": 10.1,
        "within8_min": 0.75,
        "ssim_min": 0.86,
        "measured_at": _AT,
    },
    ("h1hand", "stair"): {
        "mean_abs": 2.76,
        "within8_frac": 0.956,
        "ssim": 0.931,
        "interior_mean_abs": 0.56,
        "interior_exact_frac": 0.845,
        "draw_spread": {"mean_abs": [2.74, 2.81], "within8_frac": [0.954, 0.956], "ssim": [0.927, 0.931]},
        "mean_abs_max": 3.4,
        "within8_min": 0.91,
        "ssim_min": 0.89,
        "measured_at": _AT,
    },
    ("h1hand", "stand"): {
        "mean_abs": 3.00,
        "within8_frac": 0.932,
        "ssim": 0.852,
        "interior_mean_abs": 0.34,
        "interior_exact_frac": 0.931,
        "draw_spread": {"mean_abs": [3.00, 3.01], "within8_frac": [0.931, 0.932], "ssim": [0.846, 0.852]},
        "mean_abs_max": 3.6,
        "within8_min": 0.89,
        "ssim_min": 0.81,
        "measured_at": _AT,
    },
    ("h1hand", "walk"): {
        "mean_abs": 3.00,
        "within8_frac": 0.932,
        "ssim": 0.852,
        "interior_mean_abs": 0.34,
        "interior_exact_frac": 0.931,
        "draw_spread": {"mean_abs": [3.00, 3.01], "within8_frac": [0.931, 0.932], "ssim": [0.846, 0.852]},
        "mean_abs_max": 3.6,
        "within8_min": 0.89,
        "ssim_min": 0.81,
        "measured_at": _AT,
    },
}

#: (class name, robot, task). Only cells whose measured numbers are in ``FIDELITY`` are registered,
#: so a cell cannot ship without a measurement.
#:
#: The list is a MEASURED subset of the 49 scenes that compile, and two measurements chose it. Alpha
#: and textured primitives set the fidelity ceiling (see the module docstring). **The dead-dimension
#: gate set the rest**: the state here is exactly ``qpos``, so every entry has to reach a pixel, and
#: donor substitution over 8 states says that 9 of the 28 candidates carry entries that do not --
#: ``basketball`` 4 (the ball is a featureless SPHERE, so its free joint's quaternion is invisible by
#: construction), ``balance_hard`` 4, ``insert_*`` 5, ``cabinet`` 26, ``bookshelf_*`` 61 (drawer and
#: book hinges behind the camera or inside a closed cabinet). Declaring those would declare dimensions
#: the renderer provably ignores; RoboCasa solves the same problem by measuring and pinning a live
#: subset, and that is deliberately NOT done here because it makes the state no longer be ``qpos``.
CELLS = tuple(
    (
        f"HumanoidBench{'H1Hand' if robot == 'h1hand' else 'H1'}{''.join(p.capitalize() for p in task.split('_'))}",
        robot,
        task,
    )
    for robot, task in [
        ("h1", t) for t in ("walk", "run", "stand", "stair", "slide", "sit_simple", "sit_hard", "balance_simple")
    ]
    + [
        ("h1hand", t)
        for t in ("walk", "run", "stand", "stair", "slide", "sit_simple", "sit_hard", "balance_simple", "spoon")
    ]
)


def _make_classes():
    made = {}
    for cname, robot, task in CELLS:
        vf = FIDELITY.get((robot, task))
        if vf is None:  # not measured -> not registered
            continue
        made[cname] = register(
            type(
                cname,
                (HumanoidBenchBase,),
                dict(
                    ROBOT=robot,
                    TASK=task,
                    name=f"humanoidbench-{robot}-{task.replace('_', '-')}",
                    reference_id=f"{robot}-{task}-v0",
                    visual_fidelity=vf,
                    visual_note=VISUAL_NOTE,
                    __doc__=(
                        f"HumanoidBench {robot}-{task}-v0, through its own model camera "
                        f"{_camera_name(task)!r}.\n\n"
                        f"VISUAL TIER — these are NOT the official pixels. The frame is MJWarp's raytrace of "
                        f"HumanoidBench's own compiled scene from the same qpos: same geometry, same camera, "
                        f"same lights, a different rasterizer, and the floor's 2 cm grid replaced by the "
                        f"floor's dominant colour. Measured against MuJoCo's own raster over 64 states x 4 "
                        f"draws at 256 px: mean |diff| {vf['mean_abs']:.2f}/255, "
                        f"{100 * vf['within8_frac']:.1f}% of pixels within 8/255, SSIM {vf['ssim']:.3f}, and "
                        f"{100 * vf['interior_exact_frac']:.1f}% of interior pixels bit-exact against the "
                        f"100% every parity env in this registry declares.\n\n"
                        f"Suitable for generating training frames live. NOT a substitute for a stored dataset "
                        f"when a policy will be evaluated on official HumanoidBench pixels. See visual_note."
                    ),
                ),
            )
        )
    return made


globals().update(_make_classes())
