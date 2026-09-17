"""RoboCasa kitchens — the repo's first **VISUAL TIER** family. These are not the official pixels.

WHAT THIS IS
------------

A RoboCasa kitchen (robocasa 1.0.1 / robosuite 1.5.2) pinned to one compiled scene, re-rendered from
``qpos`` by ``mujoco_warp``'s batch raytracer at thousands of frames a second, with per-environment
camera and lighting so a batch is a batch of *different* pictures. One class is one
``(task, layout, style, camera)``; the drawn kitchen is a constant of ``(layout, style)``
(``docs/research/robocasa.md`` Section 8: 357 visible geoms identical across 64 episodes), so pinning
the scene loses the object resample and nothing else about the room.

WHY IT IS THE VISUAL TIER AND NOT THE PARITY TIER
-------------------------------------------------

Every other env in this registry claims that its frame IS the official frame — 100 % of interior
pixels bit-identical. RoboCasa cannot make that claim through this renderer and the reason is
structural, not a tuning problem. ``mujoco_warp``'s ``sample_texture`` derives UVs for ``PLANE`` and
``MESH`` only, so a textured ``BOX`` or ``CYLINDER`` samples texel (0, 0) and the whole surface comes
out one flat colour. A RoboCasa kitchen has **178-401 such geoms** — counters, backsplashes, cabinet
fronts — between a third and a half of everything drawn. MuJoCo paints a marble counter; MJWarp paints
one grey. That is the artefact, it is visible, and ``examples/robocasa_visual/`` shows it side by side
rather than describing it.

Two prior investigations concluded RoboCasa is a NO-GO for the parity tier and **both verdicts stand**:
the only renderer that reproduces the official frame exactly is MuJoCo's own, at 1,925 f/s across 16
processes against 5,137 f/s for one CPU core of JPEG decode — 0.37 cores, i.e. regenerating the frame
exactly costs 2.7x more than reading it. Nothing here contradicts that. This env exists for the other
use, the one bit-exactness is the wrong bar for: **generating training frames live**, where what
matters is that the picture is right and self-consistent.

WHAT THE TIER LICENSES, AND WHAT IT DOES NOT
---------------------------------------------

  YES   train on frames generated live from stored states, at ~5.6 k f/s per GPU at 256 px against
        ~800 f/s for the official rasterizer, with camera and lighting randomised per environment in
        the same launch.
  NO    train here and evaluate on official RoboCasa pixels. The train and eval pixel distributions
        differ by the amount in ``visual_fidelity``, and NOTHING in this repo measures what that costs
        a policy. If a stored dataset exists and the evaluation runs on official frames, use the
        stored dataset.
  NO    treat any number here as parity. ``parity_interior_min`` is ``None`` on purpose.

TWO MEASUREMENT TRAPS, BOTH FOUND HERE AND BOTH SILENT
-------------------------------------------------------

*The official frame is rendered from STALE kinematics.* ``mj_step`` evaluates the forward dynamics at
the old ``qpos`` and then integrates, so when it returns ``data.qpos`` is the new state and
``data.xpos`` — what ``mjv_updateScene`` reads — still describes the old one. Recording ``qpos`` and
the frame at that moment gives a pair that no renderer can reproduce: re-rendering the recorded
``qpos`` through ``mj_forward`` on the SAME live model disagrees on 7.7 % of pixels (39.0 % on the
wrist camera). ``examples/robocasa_visual/dump_trajectory.py`` calls ``sim.forward()`` first and then
ASSERTS the round trip is bit-exact before writing anything.

*A bare ``mujoco.Renderer`` is the official renderer, but only at geom group 1.* It defaults to groups
0-2 and paints robosuite's ~730 collision primitives over the visual meshes. Restricted to group 1 it
reproduces ``env.sim.render`` with max abs diff **0** on the live model
(``scratch/robocasa/probe_visual_reference.py``, case A) — which is what lets this env carry a
reference renderer at all, because ``robocasa/__init__.py`` asserts ``mujoco.__version__ == "3.3.1"``
while ``mujoco_warp`` requires >= 3.11, so **no interpreter can hold both robocasa and MJWarp**.

THE SCENE CACHE, AND WHY THIS ENV NEEDS ONE
--------------------------------------------

Because of that version pin, the compiled kitchen is produced in a separate interpreter and cached:

    conda run -n robocasa python examples/robocasa_visual/dump_trajectory.py \\
        --kitchen L52S03 --steps 200 --res 256 --out ~/.cache/stable-envs/robocasa

writes ``scene.xml`` (the expanded MJCF, absolute asset paths into the 24 GB library), ``qpos.npy``
(the state pool) and ``meta.json``. Override the location with ``STABLE_ENVS_ROBOCASA_CACHE``. With no
cache the module raises at import and ``stable_envs.envs.import_errors()`` says so, exactly as
``dmcontrol.py`` and ``fetch.py`` do for their optional dependencies.

THE STATE IS THE qpos THIS CAMERA CAN SEE, MEASURED
-----------------------------------------------------

A pinned kitchen has ``nq`` 91-133, and most of it is fixture hinges that are behind the camera or
inside a closed cabinet. Declaring all of it would declare dimensions the renderer provably ignores
and ``check_state_dims_matter`` would correctly fail. So the live set is MEASURED once per
``(kitchen, camera)`` — every index is perturbed, by donor substitution from the pool and then by the
same nudges the conformance check uses, and kept only if it moves the frame for EVERY probe state —
and cached beside the scene. The rest are constants of the compiled model and are pinned at the
pool's first row. The rule is deliberately stricter than the conformance check's (all probe states,
not any), so that everything declared is something that check will also find live.

WHY THERE IS NO STATIC PLATE HERE, MEASURED
--------------------------------------------

``fetch.py`` and ``metaworld.py`` both bake the static scene through MuJoCo once and composite MJWarp's
movers over it, which is where their 9-21x comes from, and a kitchen looks like the ideal case for it:
**273 of the 344 visible geoms cannot be moved by any state this env accepts**, because ``_live_dims``
pins the rest of the fixture hinges. It does not work, and the reason is the camera. All four
registered cameras are model cameras attached to a body that MOVES — ``robot0_agentview_left`` and
``_right`` to ``mobilebase0_support``, ``robot0_eye_in_hand`` to ``robot0_right_hand`` — so
``cam_xpos`` is a function of ``qpos``, and over this env's own state pool it moves **2.2-3.4 cm** on
the room cameras and **59 cm** on the wrist. There is no static view of the room to bake: a plate keyed
on the appearance is a picture from the wrong place, and a plate keyed on the camera pose is one bake
per state.

Built, measured and reverted rather than argued. The split renders **1.26x / 1.26x / 1.97x / 1.45x**
faster and disagrees with the fused frame on **19 % / 19 % / 76 % / 18 %** of pixels at up to 228/255 —
and the disagreement is the camera, not the compositing rule: the structural mover set and the one
measured over the pool agree exactly (nothing missing on any of the four cells). The two halves cost
5.08 ms (static) and 7.06 ms (dynamic) against 10.86 for the fused frame at 64 worlds and 256 px, so
paying both is **1.11x SLOWER** and the whole design rests on a cache that can never hit.
``scratch/perf/split_static.py`` and ``scratch/perf/rc_split_diag.py``.

Two further routes measured at the same time and closed by the same fact. A screen window around the
movers DOES scale here — 0.562 of the rays for 1.56x, 0.25 for 3.01x, bit-identical to the full frame's
crop (``scratch/perf/window_dyn.py``) — but it needs a plate for the pixels outside it. And nvdiffrast
reaches **1,520 f/s at batch 64** on this kitchen at 256 px and dies at 256 with an illegal memory
access (``scratch/nvdiff/bench256.py``), i.e. 0.27x of MJWarp and 0.10x of the 15,937 f/s that 8 CPU
workers reach reading these frames off disk.

So the ray-cast ceiling for the room cameras is **5,700-5,900 f/s**, this env is at 97-100 % of it, and
the wrist camera's 2.3x lead over its siblings is the same fact from the other side: it sees a fifth of
the kitchen. What would move this family is a rasterizer — 442 k triangles re-projected per world is
cheap and does not care that the camera moved.

MEASURED (256 px, 64 states x 4 draws from ``sample_states``, one H200)
-----------------------------------------------------------------------

============================  =========  =========  ======  =============  ==============
class                          mean/255   within 8   SSIM    interior mean  interior exact
============================  =========  =========  ======  =============  ==============
RoboCasaCounterToSink              6.41     0.833    0.907           1.27           0.408
RoboCasaCounterToSinkRight         5.52     0.846    0.921           0.63           0.672
RoboCasaCounterToSinkWrist         9.69     0.826    0.857           4.02           0.433
RoboCasaCounterToSinkL45S33       32.01     0.364    0.822          45.53           0.153
============================  =========  =========  ======  =============  ==============

The interior-exact column was 0.129 / 0.079 / 0.158 / 0.048 until 2026-08-11 and the change is a fix
rather than a drift: those numbers were measured with the CUDA graph captured OUTSIDE
``MW.gl_rounding``, i.e. replaying the truncating kernel. See the note above ``FIDELITY``.

Throughput at the env's own surface (states in, uint8 frames out): 1.1-1.6 k f/s at batch 1, 4.1-7.7 k
at 16, 4.9-10.6 k at 64, against ~812 f/s for the official rasterizer at 256 px. Setup is 3.0-3.5 s per
compiled kitchen, warm, so a kitchen pays for its own setup after ~1,700 frames against that
rasterizer. Full tables, the side-by-side video and the reproduction commands are in
``examples/robocasa_visual/``; the investigation is ``docs/research/robocasa.md``.

The spread between kitchens is the headline caveat and it is a factor of five, because L45 S33's
counters and cabinet fronts are textured boxes MJWarp flattens. Both ends are registered; shipping only
the good kitchen would make every number above a selection effect.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

# MuJoCo picks its GL backend from MUJOCO_GL and defaults to GLFW, which needs an X display; on a
# headless GPU node that aborts the interpreter rather than raising.
os.environ.setdefault("MUJOCO_GL", "egl")

# THE DEPENDENCY CHECK, at module scope for the reason ``fetch.py`` gives: a lazy import inside
# ``render`` turns one missing wheel into a failure per test rather than one missing env. Two things
# are needed and neither can be worked around from inside this file — the batch raytracer, and a
# compiled kitchen produced by an interpreter this one cannot be.
import importlib.util as _ilu

import mujoco as _mj

#: Where a compiled kitchen is cached. One directory per ``(layout, style)``.
CACHE_ROOT = os.path.expanduser(
    os.environ.get("STABLE_ENVS_ROBOCASA_CACHE", "~/.cache/stable-envs/robocasa"))

_BUILD_CMD = ("conda run -n robocasa python examples/robocasa_visual/dump_trajectory.py "
              "--kitchen {kitchen} --steps 200 --res 256 --out " + CACHE_ROOT)

_missing = [_m for _m in ("warp", "mujoco_warp") if _ilu.find_spec(_m) is None]
_mjver = tuple(int(x) for x in str(_mj.__version__).split(".")[:2])
if _missing or _mjver < (3, 11):                                              # pragma: no cover
    raise ImportError(
        "the RoboCasa family renders its kitchens with mujoco_warp's batch raytracer, which this "
        "interpreter cannot run: "
        + (f"missing {', '.join(_missing)}; " if _missing else "")
        + f"mujoco is {_mj.__version__} and mujoco_warp requires >= 3.11. Note that robocasa itself "
        f"pins mujoco == 3.3.1, so the interpreter that BUILDS a kitchen is never the one that "
        f"renders it; see the module docstring. See docs/research/robocasa.md")

from ..env import Env
from ..registry import register
from ..render import mjwarp as MW
from ..render import rendernet as RN
from ..spaces import Box, Dict

#: ``(class name, cache key, camera, official task/layout/style)``. A class is one compiled kitchen
#: seen through one camera, because both change the state and the fidelity.
CELLS = (
    ("RoboCasaCounterToSink", "L52S03", "robot0_agentview_left"),
    ("RoboCasaCounterToSinkRight", "L52S03", "robot0_agentview_right"),
    ("RoboCasaCounterToSinkWrist", "L52S03", "robot0_eye_in_hand"),
    ("RoboCasaCounterToSinkL45S33", "L45S33", "robot0_agentview_left"),
)

#: Only geom group 1 is drawn, on BOTH sides. This is robosuite's own convention
#: (``vopt.geomgroup[0] = render_collision_mesh``, default False; ``[1] = render_visual_mesh``, True)
#: and getting it wrong paints ~730 collision primitives over the scene on one side only.
VISIBLE_GROUPS = (1,)
#: Worlds per MJWarp render context. The ray-cast saturates by 64 at 256 px (5,462 f/s at 32, 5,599 at
#: 64, 5,710 at 256) and this family is bound by it, so the constant buys only the per-chunk
#: overheads: measured end to end at 256 px / batch 1024, **64 -> 5,504 f/s, 128 -> 5,628,
#: 256 -> 5,689**. Bit-identical at all of them (``scratch/perf/chunk_invariance.py``). A context owns
#: ``nworld * H * W * 4`` bytes of output buffer plus this family's ~356 MB of textures, and four are
#: cached, which is why this stops at 256.
MJW_CHUNK = 256
_CTX_CACHE: dict = {}
_CTX_CACHE_MAX = 4
_SCENE_CACHE: dict = {}
_REF_CACHE: dict = {}
_POOL_CACHE: dict = {}
#: ``(class, size) -> a CPU env`` borrowed by ``reference_render(style=)`` for its camera arithmetic.
_STYLE_PROBE: dict = {}


def available_kitchens() -> list[str]:
    """Cache keys with a usable compiled kitchen on this machine."""
    if not os.path.isdir(CACHE_ROOT):
        return []
    return sorted(k for k in os.listdir(CACHE_ROOT)
                  if os.path.exists(os.path.join(CACHE_ROOT, k, "scene.xml"))
                  and os.path.exists(os.path.join(CACHE_ROOT, k, "qpos.npy")))


if not available_kitchens():                                                  # pragma: no cover
    raise ImportError(
        f"no compiled RoboCasa kitchen is cached under {CACHE_ROOT}. robocasa pins mujoco == 3.3.1 "
        f"and mujoco_warp needs >= 3.11, so the kitchen has to be built by a different interpreter "
        f"and cached:\n\n    " + _BUILD_CMD.format(kitchen="L52S03") + "\n\n"
        "It needs robocasa's 24 GB asset library. Set STABLE_ENVS_ROBOCASA_CACHE to point somewhere "
        "else. See docs/research/robocasa.md")


def _joint_qpos_names(m) -> list[str]:
    """One label per ``qpos`` entry, from the compiled model's own joint table."""
    import mujoco
    suffix = {int(mujoco.mjtJoint.mjJNT_FREE): ("x", "y", "z", "qw", "qx", "qy", "qz"),
              int(mujoco.mjtJoint.mjJNT_BALL): ("qw", "qx", "qy", "qz")}
    names = [""] * int(m.nq)
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
        adr = int(m.jnt_qposadr[j])
        sfx = suffix.get(int(m.jnt_type[j]))
        if sfx is None:
            names[adr] = nm
        else:
            for k, s in enumerate(sfx):
                names[adr + k] = f"{nm}_{s}"
    return names


class _Scene:
    """One compiled kitchen, prepared for MJWarp, plus everything that does not depend on the state."""

    __slots__ = ("model", "kitchen", "xml", "sha", "qpos_names", "pool", "pinned", "prepared",
                 "gaps", "meta")


def _scene(kitchen: str) -> _Scene:
    """Compile ``kitchen`` and apply the zero-alpha pre-pass, once per process.

    The model is prepared and LEFT prepared, as ``mjwarp.BatchRenderer`` documents: MJWarp reads the
    geometry once, at ``create_render_context``, so restoring the groups afterwards would not un-draw
    anything and would only make the object's behaviour depend on when it was inspected. The reference
    renderer gets its own pristine compilation rather than sharing this one — two compilations of one
    MJCF are bit-identical (``probe_visual_reference.py``, case C), so the isolation is free.
    """
    import mujoco
    hit = _SCENE_CACHE.get(kitchen)
    if hit is not None:
        return hit
    d = os.path.join(CACHE_ROOT, kitchen)
    xml = os.path.join(d, "scene.xml")
    if not os.path.exists(xml):
        raise FileNotFoundError(
            f"no compiled kitchen for {kitchen} under {CACHE_ROOT}; build it with\n    "
            + _BUILD_CMD.format(kitchen=kitchen))
    s = _Scene()
    s.kitchen, s.xml = kitchen, xml
    with open(xml, "rb") as f:
        s.sha = hashlib.sha1(f.read()).hexdigest()[:16]
    s.meta = json.load(open(os.path.join(d, "meta.json")))
    s.model = mujoco.MjModel.from_xml_path(xml)
    s.prepared = MW.prepare_model(s.model, groups=VISIBLE_GROUPS)
    s.gaps = MW.texture_gaps(s.model, groups=VISIBLE_GROUPS)
    s.qpos_names = _joint_qpos_names(s.model)
    pool = np.load(os.path.join(d, "qpos.npy")).astype(np.float64)
    if pool.shape[1] != int(s.model.nq):
        raise ValueError(f"{kitchen}: cached qpos has {pool.shape[1]} columns, model nq is "
                         f"{s.model.nq}; the cache and the MJCF are from different builds")
    s.pool = pool
    # Everything the camera cannot see is a constant of the compiled model, pinned HERE, at the
    # pool's first row, so a fresh env in another process pins the same numbers.
    s.pinned = pool[0].copy()
    _SCENE_CACHE[kitchen] = s
    return s


class _Ctx:
    """An MJWarp render context plus the device views a render writes through."""

    __slots__ = ("wm", "wd", "rc", "bgra", "graph", "nworld", "res", "wp", "mjw", "cam")


def _context(kitchen: str, camera: str, res: int, nworld: int,
             tex_lod: int = 1, backface_cull: bool | None = None) -> _Ctx:
    """One MJWarp render context per ``(kitchen, camera, resolution, worlds, RENDER PROFILE)``.

    ``tex_lod`` and ``backface_cull`` are the profile, and they are part of the cache key rather than
    a global switch because two profiles of one cell coexist in one process: the shipped class renders
    at 256 px with full-resolution textures, and its learned-corrector sibling renders at 112 px with
    textures at LOD 2 (``render/rendernet.py``). Keying on them is what stops the second one silently
    handing the first a context built under the wrong configuration -- which is not a crash, it is a
    different picture.

    ``backface_cull=None`` means "do not pass the flag", i.e. exactly what every render in this repo
    did before the profile existed. ``False`` turns culling off, which is measured at **0 pixels
    changed** on all four cells and is worth a few per cent of the render rate
    (``scratch/altrender/README.md``); it is only taken on the corrector path, because a byte-identical
    speed-up still changes a context's identity and the declared numbers were measured without it.
    """
    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    key = (kitchen, camera, int(res), int(nworld), int(tex_lod), backface_cull)
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_CTX_CACHE) >= _CTX_CACHE_MAX:
        _CTX_CACHE.clear()
    s = _scene(kitchen)
    m = s.model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise ValueError(f"{kitchen}: camera {camera!r} is not in the compiled model")
    d0 = mujoco.MjData(m)
    d0.qpos[:] = s.pinned
    mujoco.mj_forward(m, d0)
    c = _Ctx()
    c.wp, c.mjw, c.nworld, c.res, c.cam = wp, mjw, int(nworld), int(res), cid
    # Per-world model fields. This is what makes domain randomisation ONE launch rather than N: the
    # camera pose and the lights vary per world inside a single graph replay.
    c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in
                                         ("cam_pos", "cam_quat", "light_diffuse")})
    # THE RENDER BLOCK IS A LAUNCH GEOMETRY, NOT ARITHMETIC: the megakernel is a plain
    # ``wp.launch(dim=(nworld, rays))``, one ray per thread, so the block size changes how the threads
    # are packed onto SMs and changes no float. Asserted rather than argued -- 32, 64 and 128 render
    # this kitchen to max 0/255 on 0 pixels (``scratch/perf/blockdim_identity.py``). Swept in
    # ``scratch/perf/knobs.py`` at 256 px / 64 worlds: 32 -> 10.85 ms, 64 (mujoco_warp's default) ->
    # 11.43, 128 -> 11.11, 256 -> 16.42, and 512 fails to launch, over the register budget. A kitchen
    # is the one family where it pays -- 5.4% -- and the other three are flat or worse at 32, which is
    # why this is set here and not in ``render/mjwarp.py``.
    c.wm.block_dim.render = 32
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # THE CAPTURE IS INSIDE THE PATCH. ``mujoco_warp.render`` defines its megakernel inside itself, so
    # the source is regenerated per call from whatever ``pack_rgba_to_uint32`` is bound then; captured
    # outside, the graph every later render replays is the TRUNCATING kernel and this block reaches
    # only the warm-up frame (``scratch/perf/rounding.py``).
    #
    # ``specialised_lights`` is in the same window and is ON here. A kitchen's one light declares
    # ``castshadow=0``, so stock ``mujoco_warp`` compiles two any-hit BVH traversals -- one for the
    # light, one for the headlight -- and executes NEITHER; this deletes both. Measured at 256 px /
    # batch 1024 against stock in the same process: 5,699 -> 5,771 f/s and 13,159 -> 13,215, with the
    # frames byte-identical over 64 states (``scratch/perf/lightspec_check.py``). 1.0-1.3% is small
    # because this family is bound by 442 k triangles rather than by registers -- it is taken because
    # it is free and exact, not because it is large.
    from ..render import mjwarp_lightspec as LS
    profile = {} if backface_cull is None else {"enable_backface_culling": bool(backface_cull)}
    with MW.gl_rounding(), LS.specialised_lights():
        c.rc = MW.create_render_context(
            m, tex_lod=int(tex_lod), nworld=int(nworld), cam_res=(int(res), int(res)),
            render_rgb=True, cam_active=[i == cid for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS), use_textures=True, use_shadows=True,
            render_skybox=True, **profile)
        _frame(c)
        wp.synchronize()
        # A VIEW of the warp buffer. The channel swap is applied per render rather than here:
        # ``[..., [2,1,0]]`` at construction is advanced indexing, which COPIES, and the object would
        # hold one frozen frame while every later render wrote into the buffer behind it.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(
            int(nworld), int(res), int(res), 4)
        with wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    _CTX_CACHE[key] = c
    return c


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


def _reference(kitchen: str, size: int):
    """A pristine compilation plus a ``mujoco.Renderer`` restricted to geom group 1 — the OFFICIAL
    renderer, without robosuite in the process.

    Built once per ``(kitchen, size)`` and NEVER closed: releasing an EGL context corrupts rendering in
    every other live context in the process, silently. See ``testing/conformance.py``.
    """
    import mujoco
    key = (kitchen, int(size))
    hit = _REF_CACHE.get(key)
    if hit is None:
        m = mujoco.MjModel.from_xml_path(_scene(kitchen).xml)
        vopt = mujoco.MjvOption()
        vopt.geomgroup[:] = 0
        vopt.geomgroup[1] = 1
        hit = _REF_CACHE[key] = (m, mujoco.MjData(m),
                                 mujoco.Renderer(m, width=int(size), height=int(size)), vopt)
    return hit


class RoboCasaBase(Env):
    """Base for the RoboCasa cells. VISUAL TIER — see the module docstring.

    ``KITCHEN`` and ``CAMERA`` are set by the generated subclasses.
    """

    KITCHEN: str = ""
    CAMERA: str = ""
    default_size = 256
    fidelity_tier = "visual"
    parity_interior_min = None
    #: The MJWarp render profile. ``(1, None)`` is the shipped configuration and is what every number
    #: in ``FIDELITY`` was measured through; the learned-corrector subclasses below move it, and
    #: nothing else may. See ``_context``.
    TEX_LOD: int = 1
    BACKFACE_CULL: bool | None = None

    def __init__(self, batch: int = 1, size: int | None = None, supersample: int = 1,
                 device: str = "cuda", backend: str = "auto", seed: int = 0):
        s = _scene(self.KITCHEN)
        live = _live_dims(self.KITCHEN, self.CAMERA, int(size or self.default_size))
        self._live_idx = live
        self.state_dim = len(live)
        self.state_names = tuple(s.qpos_names[i] for i in live)
        super().__init__(batch=batch, size=size, supersample=supersample, device=device,
                         backend=backend, seed=seed)
        self._s = s
        self._live_t = torch.as_tensor(live, device=self.device, dtype=torch.long)
        self._pinned_t = torch.as_tensor(s.pinned, device=self.device, dtype=torch.float32)

    # ---- state ----------------------------------------------------------------------------

    def _full_qpos(self, states: torch.Tensor) -> torch.Tensor:
        """``(B, state_dim)`` -> ``(B, nq)``: the declared dims, on top of the pinned constants."""
        q = self._pinned_t.unsqueeze(0).repeat(states.shape[0], 1)
        q[:, self._live_t] = states
        return q

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draws from the recorded trajectory pool, with every free joint's YAW randomised.

        The pool is 200 states the official simulator visited under a random arm policy, which is the
        right distribution for the arm and the wrong one for the objects: they are barely disturbed, so
        their quaternions are near-constant and ``check_state_dims_matter`` would substitute a donor
        equal to the original, fall back to an additive nudge, and watch every nudge normalise straight
        back to the same rotation — a live dimension reading as dead. This is the identity-quaternion
        trap the OGBench and Fetch ports both record. A resting object at an arbitrary yaw is a state
        this env reaches the moment the gripper touches one.
        """
        rng = rng or self.rng
        pool = _yawed_pool(self.KITCHEN)
        idx = rng.integers(0, len(pool), int(batch))
        return pool[idx][:, self._live_idx].astype(np.float32)

    # ---- variation ------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are the official picture, and which vary PER WORLD.

        Every leaf here lands in a per-world MJWarp model field — ``cam_pos``, ``cam_quat`` or
        ``light_diffuse`` — so a batch of 64 appearances is one graph replay rather than 64 renders.
        That is the capability this family is for, and it is what ``media/dr_one_batch.png`` shows: 12
        cameras and 12 lightings in 3.1 ms. What is deliberately NOT a knob: anything that would need a
        different
        TEXTURE. ``mujoco_warp`` uploads textures when the render context is built, so a per-value
        texture means a render context per value, which is the one thing this repo's EGL/context rule
        forbids. Material colours are not exposed either, for the reason ``mjwarp.py`` records: MJWarp
        reads a geom's colour from its material and one RoboCasa material is worn by geoms of several
        colours, so a colour knob here would be applied to the wrong surfaces.
        """
        return Dict({
            "camera": Dict({
                # In the camera's own frame, which for a robot-mounted camera is the frame the mount
                # tolerance lives in.
                "offset": Box(low=-0.06, high=0.06, shape=(3,), init_value=[0.0, 0.0, 0.0]),
                "pan": Box(low=-8.0, high=8.0, shape=(1,), init_value=[0.0]),
                "tilt": Box(low=-8.0, high=8.0, shape=(1,), init_value=[0.0]),
            }),
            "light": Dict({
                "scale": Box(low=0.45, high=1.6, shape=(1,), init_value=[1.0]),
                "tint": Box(low=0.6, high=1.4, shape=(3,), init_value=[1.0, 1.0, 1.0]),
            }),
        })

    def _camera_pose(self, offset: np.ndarray, pan: np.ndarray, tilt: np.ndarray):
        """``(B,3) (B,1) (B,1)`` -> ``(B,3), (B,4)`` camera pose, in the model's own convention.

        MuJoCo's camera looks down its local -z with +y up, so a pan is a rotation about local y and a
        tilt about local x, and both are applied on the RIGHT of the model quaternion so that they act
        in the camera's frame rather than the world's.
        """
        import mujoco
        s = self._s
        base_p = np.asarray(s.model.cam_pos[self._cam_id()], np.float64)
        base_q = np.asarray(s.model.cam_quat[self._cam_id()], np.float64)
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, base_q)
        R = R.reshape(3, 3)
        n = len(offset)
        pos = base_p[None] + offset @ R.T
        quat = np.empty((n, 4))
        # ``.reshape(-1)`` and not ``float(pan[i])``: pan arrives as (B, 1), and NumPy 2 refuses to
        # convert a 1-element ARRAY to a scalar rather than doing it silently as NumPy 1 did.
        pan, tilt = np.asarray(pan).reshape(-1), np.asarray(tilt).reshape(-1)
        # ONE quaternion product per DISTINCT (pan, tilt), not per row. The loop below is two
        # ``mju_mulQuat`` calls into MuJoCo per row through the Python/ctypes boundary, which on a
        # batch of 1024 was 9.2 ms of a 201 ms render -- for a knob whose default gives every row the
        # same pair. Exact float equality is the right key here because the inputs are copied, not
        # computed: identical bits in, identical bits out, so this is a dedup and not a tolerance.
        uniq, inv = np.unique(np.stack([pan, tilt], 1), axis=0, return_inverse=True)
        tmp, out = np.empty(4), np.empty(4)
        rows = np.empty((len(uniq), 4))
        for i, (p, t) in enumerate(uniq):
            a, b = np.deg2rad(p), np.deg2rad(t)
            qy = np.array([np.cos(a / 2), 0.0, np.sin(a / 2), 0.0])
            qx = np.array([np.cos(b / 2), np.sin(b / 2), 0.0, 0.0])
            mujoco.mju_mulQuat(tmp, qy, qx)
            mujoco.mju_mulQuat(out, base_q, tmp)
            rows[i] = out
        quat[:] = rows[inv]
        return pos, quat

    def _cam_id(self) -> int:
        import mujoco
        return mujoco.mj_name2id(self._s.model, mujoco.mjtObj.mjOBJ_CAMERA, self.CAMERA)

    # ---- render ---------------------------------------------------------------------------

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device. Style may differ per environment."""
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
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

        qpos = self._full_qpos(s)
        offset = sty["camera.offset"].reshape(B, 3).cpu().numpy().astype(np.float64)
        pan = sty["camera.pan"].reshape(B, 1).cpu().numpy().astype(np.float64)
        tilt = sty["camera.tilt"].reshape(B, 1).cpu().numpy().astype(np.float64)
        cpos, cquat = self._camera_pose(offset, pan, tilt)
        cpos = torch.as_tensor(cpos, device=self.device, dtype=torch.float32)
        cquat = torch.as_tensor(cquat, device=self.device, dtype=torch.float32)
        base_light = torch.as_tensor(np.asarray(self._s.model.light_diffuse, np.float64),
                                     device=self.device, dtype=torch.float32)
        light = (base_light.unsqueeze(0) * sty["light.scale"].reshape(B, 1, 1)
                 * sty["light.tint"].reshape(B, 1, 3))

        for lo in range(0, B, MJW_CHUNK):
            hi = min(lo + MJW_CHUNK, B)
            n = hi - lo
            nw = 1 if n <= 1 else min(MJW_CHUNK, 1 << (n - 1).bit_length())
            c = _context(self.KITCHEN, self.CAMERA, res, nw, self.TEX_LOD, self.BACKFACE_CULL)
            wp = c.wp
            wp.to_torch(c.wd.qpos)[:n] = qpos[lo:hi]
            wp.to_torch(c.wm.cam_pos)[:n, c.cam] = cpos[lo:hi]
            wp.to_torch(c.wm.cam_quat)[:n, c.cam] = cquat[lo:hi]
            wp.to_torch(c.wm.light_diffuse)[:n] = light[lo:hi]
            # torch wrote the per-world arrays on its stream; warp replays the graph on its own. The
            # two syncs bracket the launch so a frame can never be rendered from half-written state.
            torch.cuda.synchronize()
            wp.capture_launch(c.graph)
            wp.synchronize()
            rgb = c.bgra[:n][..., [2, 1, 0]]
            if ss == 1:
                buf[lo:hi] = rgb
            else:
                buf[lo:hi] = (rgb.to(torch.float32).reshape(n, size, ss, size, ss, 3)
                              .mean((2, 4)).round_().clamp_(0, 255).to(torch.uint8))
        return buf.view(*lead, size, size, 3) if lead else buf

    # ---- reference bridge -----------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 256):
        """MuJoCo's own rasterizer over the same compiled kitchen, at geom group 1.

        This IS the official renderer: restricted to group 1 it reproduces robosuite's
        ``env.sim.render`` with max abs diff 0 on the live model. It is reconstructed here rather than
        driven through robosuite because ``robocasa/__init__.py`` asserts ``mujoco.__version__ ==
        "3.3.1"`` and ``mujoco_warp`` needs >= 3.11, so the two cannot share an interpreter.
        """
        return _reference(cls.KITCHEN, int(size))

    @classmethod
    def reference_render(cls, ref, states, style=None) -> np.ndarray:
        """Official frames for these states, optionally at a per-state APPEARANCE.

        ``style`` is a ``(B, style_dim)`` packed appearance in this env's own variation space, and it
        is written into the reference MODEL — ``cam_pos``, ``cam_quat``, ``light_diffuse`` — because
        ``mujoco.Renderer`` takes no appearance argument. It exists so that a claim measured under the
        env's declared variation compares two renders of the SAME scene: rendering our side with a
        drawn camera and the official side at the default camera reports mean 41.6 and SSIM 0.22,
        which is a number about the harness rather than about the renderer, and it looks like a
        catastrophic regression.

        The pose arithmetic is ``_camera_pose``, CALLED rather than reimplemented, so the two
        renderers are handed the same camera by construction instead of by two implementations
        agreeing. The model is restored afterwards; the reference is a process-wide cached object and
        leaving a drawn camera in it would move every later measurement.
        """
        import mujoco
        m, d, r, vopt = ref
        s = _scene(cls.KITCHEN)
        live = _live_dims(cls.KITCHEN, cls.CAMERA, int(r.height))
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cls.CAMERA)
        st = np.asarray(states, np.float64)
        out = np.empty((len(st), int(r.height), int(r.width), 3), np.uint8)
        cpos = cquat = lights = None
        base_pos = np.asarray(m.cam_pos[cid], np.float64).copy()
        base_quat = np.asarray(m.cam_quat[cid], np.float64).copy()
        base_light = np.asarray(m.light_diffuse, np.float64).copy()
        if style is not None:
            # One probe per (class, size), kept: it exists only to lend its ``_camera_pose`` and its
            # variation layout, and rebuilding it per chunk of a 16 k-frame generation is a
            # measurable share of the run for an object with no per-call state.
            probe = _STYLE_PROBE.get((cls, int(r.height)))
            if probe is None:
                probe = _STYLE_PROBE[(cls, int(r.height))] = cls(batch=1, size=int(r.height),
                                                                 device="cpu")
            packed = np.asarray(style, np.float64).reshape(len(st), -1)
            leaf = {p: packed[:, off:off + w] for p, off, w in probe.variation_space.flat_layout()}
            cpos, cquat = probe._camera_pose(leaf["camera.offset"], leaf["camera.pan"],
                                             leaf["camera.tilt"])
            lights = (base_light[None] * leaf["light.scale"].reshape(-1, 1, 1)
                      * leaf["light.tint"].reshape(-1, 1, 3))
        try:
            for i, row in enumerate(st):
                if style is not None:
                    m.cam_pos[cid] = cpos[i]
                    m.cam_quat[cid] = cquat[i]
                    m.light_diffuse[:] = lights[i]
                d.qpos[:] = s.pinned
                d.qpos[live] = row
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=cid, scene_option=vopt)
                out[i] = r.render()
        finally:
            m.cam_pos[cid] = base_pos
            m.cam_quat[cid] = base_quat
            m.light_diffuse[:] = base_light
        return out

    @classmethod
    def corrector_cell_key(cls) -> dict:
        """The identity a learned render corrector is keyed to. See ``render/rendernet.py``.

        Four fields, and every one of them changes the picture: the class (so a checkpoint cannot be
        moved between siblings), the compiled kitchen, the camera on it, and the SHA of the expanded
        MJCF. The last is what stops a corrector fitted before a kitchen was rebuilt from being applied
        after — the live-qpos cache is keyed on the same hash for the same reason, and a rebuilt
        kitchen is a scene whose geometry the weights have never seen.
        """
        return {"env_class": cls.__name__, "family": "robocasa", "scene": cls.KITCHEN,
                "camera": cls.CAMERA, "scene_sha": _scene(cls.KITCHEN).sha}


def _yawed_pool(kitchen: str) -> np.ndarray:
    """The recorded pool with every free joint's yaw randomised. Built once per process, seeded."""
    hit = _POOL_CACHE.get(kitchen)
    if hit is not None:
        return hit
    import mujoco
    s = _scene(kitchen)
    m = s.model
    pool = np.repeat(s.pool, 2, axis=0).copy()
    rng = np.random.default_rng(0)
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        adr = int(m.jnt_qposadr[j]) + 3
        yaw = rng.uniform(-np.pi, np.pi, len(pool))
        pool[:, adr:adr + 4] = np.stack(
            [np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], 1)
    _POOL_CACHE[kitchen] = pool
    return pool


def _live_dims(kitchen: str, camera: str, size: int) -> np.ndarray:
    """Which ``qpos`` indices this camera can see, MEASURED once and cached beside the scene.

    A pinned kitchen carries 91-133 joints and most of them are fixture hinges behind the camera or
    inside a closed cabinet. The rule matches ``check_state_dims_matter`` — substitute a donor value
    from the pool, then try the additive nudges — with one deliberate difference: a dimension is kept
    only if it moves the frame for EVERY probe state, where the conformance check accepts any. Strictly
    stronger, so nothing declared here can fail that check, and a dimension dropped for being invisible
    from this camera is pinned in the model rather than advertised and ignored.

    Cached as JSON keyed by the MJCF's own hash, so a rebuilt kitchen re-measures instead of silently
    reusing indices into a different model.
    """
    s = _scene(kitchen)
    path = os.path.join(CACHE_ROOT, kitchen, f"live_qpos_{camera}_{int(size)}.json")
    if os.path.exists(path):
        rec = json.load(open(path))
        if rec.get("sha") == s.sha:
            return np.asarray(rec["live"], np.int64)

    import mujoco
    nq = int(s.model.nq)
    pool = _yawed_pool(kitchen)
    rng = np.random.default_rng(0)
    m2 = mujoco.MjModel.from_xml_path(s.xml)
    d2 = mujoco.MjData(m2)
    _, _, r, vopt = _reference(kitchen, int(size))
    cid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_CAMERA, camera)

    def draw(q):
        d2.qpos[:] = q
        d2.qvel[:] = 0.0
        mujoco.mj_forward(m2, d2)
        r.update_scene(d2, camera=cid, scene_option=vopt)
        return r.render().copy()

    # Measured through MuJoCo rather than through MJWarp on purpose. The state contract is a claim
    # about the SCENE -- "this number reaches the picture" -- and settling it with the approximate
    # renderer would let a dimension be declared dead because the fast path happens to flatten the
    # surface it moves.
    probes = 4
    base_states = pool[rng.integers(0, len(pool), probes)]
    donors = pool[rng.integers(0, len(pool), 4)]
    live = []
    for dim in range(nq):
        alive_everywhere = True
        for q0 in base_states:
            base = draw(q0)
            moved = False
            for cand in list(donors[:, dim]) + [q0[dim] + 25.0, q0[dim] - 25.0,
                                                q0[dim] + 1.0, q0[dim] + 0.5]:
                if cand == q0[dim]:
                    continue
                q = q0.copy()
                q[dim] = cand
                if not np.array_equal(draw(q), base):
                    moved = True
                    break
            if not moved:
                alive_everywhere = False
                break
        if alive_everywhere:
            live.append(dim)
    live = np.asarray(live, np.int64)
    names = _joint_qpos_names(s.model)
    with open(path, "w") as f:
        json.dump({"sha": s.sha, "camera": camera, "size": int(size), "nq": nq,
                   "live": live.tolist(), "live_names": [names[i] for i in live]}, f, indent=1)
    return live


#: MEASURED, not chosen: 64 states from each env's own ``sample_states`` at its default 256 px on one
#: H200, against MuJoCo's raster of the same compiled kitchen
#: (``examples/robocasa_visual/calibrate_tier.py``; the 200-state trajectory version of the same table
#: is in ``examples/robocasa_visual/media/manifest.json`` and agrees to ~0.1/255). ``*_max`` / ``*_min``
#: are the thresholds the conformance suite asserts and sit clear of the measurements so a state draw
#: cannot flake the gate; both are here so the margin is visible rather than implied.
#:
#: L45S33 is in this table for the same reason a negative result is in a paper. It is the worst of the
#: four kitchens measured in ``docs/research/robocasa.md`` Section 11 and it is five times worse than
#: L52S03 on every column, because its counters and cabinet fronts are textured boxes that MJWarp
#: draws flat. Registering only the good kitchen would make the tier's numbers a selection effect.
#: RE-MEASURED 2026-08-11, AND ``interior_exact_frac`` MOVED BY 3x ON EVERY CELL. The previous table
#: was taken before ``_context`` moved the CUDA-graph capture INSIDE ``MW.gl_rounding`` (see the
#: comment there): captured outside, the graph every render replayed was the TRUNCATING kernel and the
#: patch reached only the warm-up frame. Putting the graph back outside reproduces the OLD numbers to
#: three decimals -- 0.1299 / 0.0498 against the 0.129 / 0.048 that were declared, and mean 6.55
#: against the declared 6.54-6.58 (``scratch/perf/draw_spread.py --legacy``) -- which is what says the
#: capture fix is the cause and not a harness difference. Rounding instead of truncating is worth
#: interior-exact **0.129 -> 0.413, 0.079 -> 0.683, 0.158 -> 0.451, 0.048 -> 0.153** and only 0.1-0.4
#: of a level on ``mean_abs``, which is exactly why the other three columns looked like they still
#: reproduced and this one did not. The thresholds are UNCHANGED: every one of them still passes, now
#: with more margin, and a threshold moved to fit a measurement is not a threshold.
_AT = ("256 px, 64 states x 4 independent draws from sample_states (fresh env per draw, seeds 0-3), "
       "H200, mujoco 3.11.0 / mujoco-warp 3.11.0 / warp 1.16.0, graph captured inside gl_rounding; "
       "the measured values below are the WORST of the four and draw_spread is [min, max]")
FIDELITY = {
    ("L52S03", "robot0_agentview_left"): {
        "mean_abs": 6.41, "within8_frac": 0.833, "ssim": 0.907,
        "interior_mean_abs": 1.27, "interior_exact_frac": 0.408,
        "draw_spread": {"mean_abs": [6.36, 6.41], "within8_frac": [0.833, 0.834],
                        "ssim": [0.907, 0.909], "interior_exact_frac": [0.408, 0.430]},
        "mean_abs_max": 8.5, "within8_min": 0.78, "ssim_min": 0.86, "measured_at": _AT,
    },
    ("L52S03", "robot0_agentview_right"): {
        "mean_abs": 5.52, "within8_frac": 0.846, "ssim": 0.921,
        "interior_mean_abs": 0.63, "interior_exact_frac": 0.672,
        "draw_spread": {"mean_abs": [5.37, 5.52], "within8_frac": [0.846, 0.850],
                        "ssim": [0.921, 0.925], "interior_exact_frac": [0.672, 0.687]},
        "mean_abs_max": 7.5, "within8_min": 0.79, "ssim_min": 0.88, "measured_at": _AT,
    },
    ("L52S03", "robot0_eye_in_hand"): {
        # The widest spread in the family, and it is the state distribution rather than the renderer:
        # a wrist camera 10 cm from an object is mostly showing that object, so randomising its yaw
        # (see ``sample_states``) moves the frame far more than it does on a room-scale view. The
        # thresholds are set from the worst draw, not the first.
        "mean_abs": 9.69, "within8_frac": 0.826, "ssim": 0.857,
        "interior_mean_abs": 4.02, "interior_exact_frac": 0.433,
        "draw_spread": {"mean_abs": [4.30, 9.69], "within8_frac": [0.826, 0.868],
                        "ssim": [0.857, 0.953], "interior_exact_frac": [0.433, 0.458]},
        "mean_abs_max": 14.0, "within8_min": 0.76, "ssim_min": 0.80, "measured_at": _AT,
    },
    ("L45S33", "robot0_agentview_left"): {
        "mean_abs": 32.01, "within8_frac": 0.364, "ssim": 0.822,
        "interior_mean_abs": 45.53, "interior_exact_frac": 0.153,
        "draw_spread": {"mean_abs": [31.90, 32.01], "within8_frac": [0.364, 0.370],
                        "ssim": [0.822, 0.824], "interior_exact_frac": [0.153, 0.154]},
        "mean_abs_max": 38.0, "within8_min": 0.32, "ssim_min": 0.77, "measured_at": _AT,
    },
}


def _make_classes():
    made = {}
    have = set(available_kitchens())
    for cname, kitchen, camera in CELLS:
        if kitchen not in have:
            continue
        vf = FIDELITY.get((kitchen, camera))
        if vf is None:
            continue
        made[cname] = register(type(cname, (RoboCasaBase,), dict(
            KITCHEN=kitchen, CAMERA=camera,
            name=f"robocasa-{kitchen.lower()}-{camera.replace('robot0_', '').replace('_', '-')}",
            reference_id=f"robocasa:PickPlaceCounterToSink/{kitchen}/{camera}",
            visual_fidelity=vf,
            visual_note=VISUAL_NOTE,
            __doc__=(
                f"RoboCasa PickPlaceCounterToSink, kitchen {kitchen}, camera {camera}.\n\n"
                f"VISUAL TIER — these are NOT the official pixels. The frame is MJWarp's raytrace of "
                f"the same compiled kitchen from the same qpos: same geometry, same camera, same "
                f"materials, a different rasterizer. Measured against MuJoCo's own raster of this "
                f"scene over 200 trajectory states at 256 px: mean |diff| "
                f"{vf['mean_abs']:.2f}/255, {100 * vf['within8_frac']:.1f}% of pixels within 8/255, "
                f"SSIM {vf['ssim']:.4f}, and {100 * vf['interior_exact_frac']:.1f}% of interior "
                f"pixels bit-exact against the 100% every parity env in this registry declares.\n\n"
                f"Suitable for generating training frames live. NOT a substitute for a stored dataset "
                f"when a policy will be evaluated on official RoboCasa pixels. See visual_note."),
        )))
    return made


VISUAL_NOTE = (
    "ONE DOMINANT RESIDUAL, structural to the backend and not to this scene: TEXTURED PRIMITIVES ARE "
    "DRAWN AS ONE FLAT COLOUR. mujoco_warp's sample_texture derives UVs for PLANE and MESH only, so a "
    "textured BOX, CYLINDER, SPHERE or CAPSULE samples texel (0,0) and the entire surface takes that "
    "one corner texel. A RoboCasa kitchen has 178-401 such geoms in geom group 1 -- counters, "
    "backsplashes, cabinet fronts, sink basins -- between a third and a half of everything drawn, so "
    "a marble counter comes out one grey. It is visible: examples/robocasa_visual/ shows it beside "
    "the official frame with the difference amplified 4x. It is also why the spread between kitchens "
    "is a factor of five rather than noise -- the kitchens with more textured boxes score worse, and "
    "both ends are registered rather than only the good one. Baking those materials to their "
    "texture's MEAN colour instead of its corner texel was tried and measured and does NOT pay "
    "(mean 6.62 -> 6.71 on L52S03), because RoboCasa's box materials are often near-uniform and only "
    "38 of ~137 textured materials are worn exclusively by primitives. Secondary residuals, both "
    "shared with every other MJWarp scene in this repo: texture MINIFICATION (MuJoCo mipmaps; MJWarp "
    "point-samples a linear filter of the base level) and MSAA on silhouettes. What is NOT in these "
    "numbers because it was found and fixed: the zero-alpha pre-pass for geoms AND sites (without it "
    "170-198 invisible placement markers per kitchen render as opaque green over ~40% of the frame) "
    "and the float->uint8 quantisation rule (mujoco_warp truncates, GL rounds). WHAT THIS TIER DOES "
    "NOT LICENSE: training here and evaluating on official RoboCasa pixels. The two pixel "
    "distributions differ by the amounts above and nothing in this repo measures what that costs a "
    "policy.")


#: ``env class -> LoadedCorrector``. One per process, like the render contexts and for the same
#: reason: it owns a compiled kernel, a CUDA graph and two static buffers, and building it twice is
#: tens of seconds of autotuning for an object that has no per-instance state.
_CORRECTOR_CACHE: dict = {}


class RoboCasaCorrectedBase(RoboCasaBase):
    """A RoboCasa cell whose frame is drawn by MJWarp at 112 px and then by a LEARNED corrector.

    VISUAL TIER, LEARNED decoder. See ``stable_envs/render/rendernet.py`` for what that is and
    ``docs/concepts/fidelity-tiers.md`` for what it does and does not license.

    Everything about the state, the variation space and the reference bridge is inherited unchanged
    from the plain class: the same qpos reaches the same MJWarp scene through the same camera. The two
    differences are the RENDER PROFILE (112 px, textures at LOD 2, backface culling off — all three
    read from the checkpoint, never chosen here) and the network applied to its output.
    """

    render_decoder = "learned"
    #: An UNSEALED corrector, for the one caller that cannot use a sealed one: the training script,
    #: which must run this render path in order to measure the declaration a sealed checkpoint is
    #: required to carry. Never set by anything a user reaches, and a class with one set cannot be
    #: registered, because registration reads the declaration out of the sealed file.
    _corrector_override = None

    def corrector(self):
        """The verified corrector for this cell. Built once per process, on first use."""
        if self._corrector_override is not None:
            return self._corrector_override
        cls = type(self)
        hit = _CORRECTOR_CACHE.get(cls.__name__)
        if hit is None:
            hit = _CORRECTOR_CACHE[cls.__name__] = RN.load(cls.corrector_cell_key())
        return hit

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,256,256,3)`` uint8 on device, through the corrector.

        ``size`` and ``supersample`` are REFUSED rather than serviced when they are not the pair the
        checkpoint was fitted on. A corrector is a fixed map from one resolution to another; asked for
        a 128 px frame it could only be given a resized 256 px one, which is a different picture from
        the one every declared number here describes, produced silently. The plain sibling class
        renders at any size and is the right answer to that request.
        """
        corr = self.corrector()
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        if size != corr.out_size or ss != 1:
            raise ValueError(
                f"{type(self).__name__} delivers {corr.out_size}x{corr.out_size} at supersample 1 "
                f"and nothing else -- that is the map its checkpoint was fitted on -- and was asked "
                f"for {size}x{size} at supersample {ss}. Serving it would mean resizing the "
                f"{corr.out_size} px frame, which is a different picture from the one every declared "
                f"number here describes. Use "
                f"{type(self).__name__.removesuffix('Corrected')!r}, the plain visual-tier class for "
                f"this cell, which renders at any size.")
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        b = int(s.shape[0])
        if out is None:
            out = torch.empty((b, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(b, size, size, 3)
        low = super().render(s, size=corr.render_size, supersample=1, variation=variation,
                             style=style)
        corr(low, out=buf)
        return buf.view(*lead, size, size, 3) if lead else buf


def _corrected_note(base_note: str, hdr: dict) -> str:
    vf, base = hdr["visual_fidelity"], hdr.get("baseline") or {}
    return (
        "A LEARNED DECODER DREW THIS FRAME. MJWarp renders this cell at "
        f"{hdr['render']['render_size']} px with textures at LOD {hdr['render']['tex_lod']}, and a "
        f"{hdr['net']} convolutional corrector fitted on THIS cell maps that to "
        f"{hdr['render']['out_size']} px. The frame is still a pure deterministic function of "
        "(state, style) -- the checkpoint is fixed and the render feeding it was already such a "
        "function -- so render is still f(get_state()), still batch-invariant, still reproducible "
        "bit-for-bit by anyone holding the weights. WHAT CHANGED IS THE DECODER: this is no longer "
        "MuJoCo's renderer approximated, it is a learned approximation of MuJoCo's renderer, with "
        "declared error. It is closer to the official frame than the plain class on every declared "
        f"number at once (mean |diff| {base.get('mean_abs', float('nan')):.2f} -> "
        f"{vf['mean_abs']:.2f}/255, within-8 {base.get('within8_frac', float('nan')):.3f} -> "
        f"{vf['within8_frac']:.3f}, SSIM {base.get('ssim', float('nan')):.3f} -> {vf['ssim']:.3f}) "
        "AND the detail it restores on textured surfaces is SYNTHESISED rather than measured: right "
        "statistics, wrong pixels, which a frozen pretrained encoder can tell apart even where the "
        "pixel metrics improve by 2.5x. It is specific to this scene and this camera -- applied to a "
        "sibling cell it is measured as worse than not correcting at all, and the loader refuses to "
        "do it. WHAT THIS DOES NOT LICENSE: everything the plain class does not license, and one "
        "more -- do not describe these frames as MuJoCo's renderer at reduced fidelity.\n\nTHE "
        "UNDERLYING RENDER'S OWN RESIDUAL, unchanged and still present under the correction: "
        + base_note)


def _make_corrected_classes() -> dict:
    """One class per cell with a sealed corrector on this machine. NO CHECKPOINT, NO CLASS.

    This is the whole opt-in mechanism and it is deliberately not a flag. A corrector exists for a
    cell or it does not; when it does, it is a DIFFERENT env with a different name, a different
    declared fidelity and a different answer to "what drew this", so it is a different registry entry.
    When it does not, nothing appears and the plain cell is untouched — the absence of weights removes
    an env, it never changes one.

    Everything a class needs comes out of the checkpoint header: the architecture, the render profile,
    and the MEASURED declaration, which is a property of the (renderer, checkpoint) pair rather than
    of the class. So a new sealed checkpoint dropped into the cache directory registers a new class
    with its own numbers and no code change here.
    """
    made = {}
    have = set(available_kitchens())
    heads = RN.available()
    for cname, kitchen, camera in CELLS:
        hdr = heads.get(cname + "Corrected")
        if hdr is None or kitchen not in have:
            continue
        cell = hdr.get("cell") or {}
        if cell.get("scene") != kitchen or cell.get("camera") != camera:
            continue                       # the loader will say why; do not register a wrong class
        vf = dict(hdr["visual_fidelity"])
        made[cname + "Corrected"] = register(type(cname + "Corrected", (RoboCasaCorrectedBase,), dict(
            KITCHEN=kitchen, CAMERA=camera,
            TEX_LOD=int(hdr["render"]["tex_lod"]),
            BACKFACE_CULL=hdr["render"]["backface_cull"],
            default_size=int(hdr["render"]["out_size"]),
            name=f"robocasa-{kitchen.lower()}-"
                 f"{camera.replace('robot0_', '').replace('_', '-')}-corrected",
            reference_id=f"robocasa:PickPlaceCounterToSink/{kitchen}/{camera}",
            visual_fidelity=vf,
            variation_fidelity=dict(hdr.get("variation_fidelity") or {}),
            visual_note=_corrected_note(VISUAL_NOTE, hdr),
            __doc__=(
                f"RoboCasa PickPlaceCounterToSink, kitchen {kitchen}, camera {camera}, through a "
                f"LEARNED render corrector.\n\n"
                f"VISUAL TIER, LEARNED DECODER — these are NOT the official pixels, and they are not "
                f"a renderer's output either. MJWarp draws this cell at "
                f"{hdr['render']['render_size']} px and a {hdr['net']} network fitted on this cell "
                f"maps that to {hdr['render']['out_size']} px, correcting both the resolution and "
                f"mujoco_warp's missing texture coordinates on BOX/CYLINDER primitives. Measured "
                f"against MuJoCo's own raster of this scene, 64 states x 4 draws at "
                f"{hdr['render']['out_size']} px, worst draw: mean |diff| {vf['mean_abs']:.2f}/255, "
                f"{100 * vf['within8_frac']:.1f}% of pixels within 8/255, SSIM {vf['ssim']:.4f} — "
                f"against {vf.get('plain_mean_abs', float('nan')):.2f}, "
                f"{100 * vf.get('plain_within8_frac', float('nan')):.1f}% and "
                f"{vf.get('plain_ssim', float('nan')):.4f} for {cname}, the plain class, on the same "
                f"draws.\n\n"
                f"CLOSER TO THE OFFICIAL FRAME AND PARTLY SYNTHESISED — both, at the same time. The "
                f"texture it puts back on a marble counter came out of its weights, not out of the "
                f"frame it was given. Suitable for generating training frames live; NOT a substitute "
                f"for a stored dataset when a policy will be evaluated on official RoboCasa pixels, "
                f"and not describable as MuJoCo's renderer. See visual_note and "
                f"stable_envs/render/rendernet.py."),
        )))
    return made


def missing_kitchens() -> dict[str, str]:
    """Declared cells with no compiled kitchen on this machine, and the command that builds each."""
    have = set(available_kitchens())
    return {k: _BUILD_CMD.format(kitchen=k) for _, k, _ in CELLS if k not in have}


globals().update(_make_classes())
globals().update(_make_corrected_classes())
