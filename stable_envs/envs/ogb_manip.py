"""OGBench MANIPULATION — ``swm/OGBCube-v0`` and ``swm/OGBScene-v0``, on the Fetch pattern.

Reference: ``stable_worldmodel``'s own forks of OGBench's ``ManipSpaceEnv`` — ``CubeEnv`` at
**224 px** and ``SceneEnv`` at **200 px**, both ``ob_type='states'``, both rendered from the model
camera ``front_pixels``. Not ogbench's ``cube-single-v0`` / ``scene-v0``: swm subclasses
``ManipSpaceEnv`` directly and registers its subclasses with no kwargs, so the registered defaults
are swm's and those are what is certified here. (``SceneEnv`` renders at 200 px only because it
forgets to override the ogbench default while swm's own collection scripts pass 224; the registered
default is certified as it stands and the omission is recorded as an upstream note.)

WHY THIS FAMILY IS A ``visual`` CELL AND FETCH IS NOT
-----------------------------------------------------

Structurally it is the Fetch pattern exactly: a plate baked through MuJoCo itself, MJWarp's batch
ray-caster for the movers, composited by MJWarp's own segmentation buffer. Two of Fetch's four
corrections are not needed (the camera is a MODEL camera, so nothing has to be recovered from
``mjvScene``; ``texture_gaps`` is empty), a third is absent (no visible sites), and the fourth
(``gl_rounding``) applies unchanged.

What is different is the LIGHTING, and it was attributed term by term before this class was written
(``examples/results/step7/ogb-manipulation/m1/``, ``m1b/``; jobs 6396646, 6397454, 6397621 on a
3090). Raw MJWarp scored **0.0091** interior-exact against MuJoCo's EGL frames. One term explained
almost all of it — ``mujoco_warp`` honours a spot cutoff MuJoCo discards — and that correction now
lives in the SHARED path (``render/mjwarp.py::neutralise_illegal_spots``), because it is a property
of the two renderers and not of this scene. It is worth **0.0091 -> 0.3507**. What remains, measured
on top of it and in order:

* the ``global`` DIRECTIONAL light, now the largest term (interior 0.347 -> 0.702 when neutralised);
* the spot's DISTANCE attenuation (a further +0.113; MJWarp takes ``attenuation = 1``);
* the headlight, which is a MOVER-pixel term rather than a floor one (interior +0.001, mover
  exactness 0.757 -> 0.896).

Those are why the tier is ``visual``. They are declared in ``visual_fidelity`` rather than chased:
the coordinator bounded the attribution to one further pass and this class was written at the tier
the evidence supported.

WHERE THE FIDELITY ACTUALLY COMES FROM
--------------------------------------

The interior mask on this scene keeps 0.41 (Cube) / 0.28 (Scene) of pixels, and most of what it
keeps is the floor -- which is the plate, baked through MuJoCo itself and composited in wherever
MJWarp did not hit a MOVER. Getting that composite right is the single largest fidelity term in this
family, larger than every lighting correction put together: an earlier version masked with
``seg >= 0``, which is true wherever MJWarp hit ANYTHING, so the plate was used only on sky and the
cell scored 0.351 interior-exact. With the mover LUT it scores **0.958** (Cube) and **0.922**
(Scene), and whole-frame exactness is **0.872** and **0.775**. All four are measured, on three
independent 64-state draws, and are the numbers in ``visual_fidelity``.

STATE
-----

``(qpos, mocap_pos)`` — and for the Scene, two extra dimensions for the button bits.
Measured in phase 0: the frame is a bit-exact function of exactly that, verified by imposing a state
from one episode onto a DIFFERENT episode (different seed, recompiled model) and getting an identical
frame. The mocap pose is what positions the translucent goal marker, and it is NOT in swm's info
dict, so a caller has to read it off ``data``.

Two things are deliberately NOT in the state, for two different reasons. **``qvel``** reaches no
pixel — the mover pass runs ``kinematics``, which reads ``qpos`` and the mocap pose — and Gate A
measured exactly that: carrying it, 20 of 48 dimensions moved no pixel. swm's own
``reset(options={'state'})`` contract is still ``concat(qpos, qvel)``; the RENDER state is the part
that reaches a pixel. **``mocap_quat``** is CONSTANT under the registered distribution: over 520
states spanning both cells, all five task ids and 12 steps per episode it has exactly one unique
row, the identity, because ``initialize_episode`` writes ``lie.SO3.identity().wxyz`` and nothing in
task mode rotates the goal marker. A dimension that never varies cannot be shown to matter and the
repo's ``check_state_dims_matter`` reports it dead, so it is supplied as a constant instead.
``fetch.py`` made the same call when it dropped mocap from its own state.

The Scene's two button bits are not in ``qpos`` either. They are required keyword arguments of
``SceneEnv.set_state`` — so swm's own API exposes them — and they drive ``geom_rgba`` of the two
buttons and ``mat_rgba`` of the drawer/window handles, worth 1.0-2.2 % of pixels. (swm's
``reset(options={'state': ...})`` cannot supply them and raises ``KeyError``; recorded as an upstream
defect.)
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from ..core.env import StableEnv, register
from ..spaces import Box, Dict
from ..render import mjwarp as MW

#: Groups ``mujoco.Renderer`` draws at the env's own ``MjvOption`` (``geomgroup = [1,1,1,0,0,0]``).
VISIBLE_GROUPS = (0, 1, 2)


def IDENTITY_QUAT(nmocap: int) -> np.ndarray:
    """``(nmocap, 4)`` of ``(1,0,0,0)``. The only mocap orientation this family's distribution has."""
    q = np.zeros((int(nmocap), 4), np.float64)
    q[:, 0] = 1.0
    return q
#: MJWarp worlds per launch. Same constant and the same reason as Fetch: a context is per (res,
#: nworld) and an unbounded batch would build one per batch size.
MJW_CHUNK = 256
#: The leaves swm resamples on every reset by default (`CubeEnv.DEFAULT_VARIATIONS`). They are drawn
#: explicitly from the caller's `rng` so a state draw is reproducible; see `_variation_values`.
DEFAULT_VARIATION_PATHS = {
    "Cube": ("cube.start_position", "cube.start_yaw", "cube.goal_position", "cube.goal_yaw"),
    "Scene": ("cube.start_position", "cube.start_yaw", "cube.goal_position", "cube.goal_yaw"),
}
_CTX_CACHE: dict = {}
_CTX_CACHE_MAX = 4
_PLATE_CACHE: dict = {}
_PLATE_CACHE_MAX = 8
_BUNDLE_CACHE: dict = {}
_REF_CACHE: dict = {}


import contextlib as _contextlib


@_contextlib.contextmanager
def _seeded_action_space(u, seed: int):
    """Give the env a STABLE, seeded action space for the duration of the block.

    THE DEFECT THIS WORKS AROUND IS UPSTREAM AND IS WHY `sample_states` was not reproducible.
    `ManipSpaceEnv.action_space` is a PROPERTY that constructs a brand-new `gym.spaces.Box` on every
    access, each with its own OS-seeded RNG -- verified: two consecutive accesses return different
    objects. `initialize_episode` then calls `self.step(self.action_space.sample())` twice to settle
    the scene, so every `reset()` consumes two actions from a generator nothing can seed, and
    `reset(seed=...)` cannot make a reset reproducible. Seeding `env.action_space` or
    `env.unwrapped.action_space` from outside seeds a throwaway.

    Because the property is a data descriptor on a base class, an instance attribute cannot shadow
    it; the override has to be on the class that defines it, and is undone in `finally`. Scoped to
    this block only, so nothing outside state sampling sees a changed env.
    """
    import gymnasium as gym
    owner = next((k for k in type(u).__mro__ if "action_space" in k.__dict__), None)
    if owner is None:
        yield
        return
    original = owner.__dict__["action_space"]
    box = u.action_space
    box.seed(int(seed))
    try:
        setattr(owner, "action_space", box)
        yield
    finally:
        setattr(owner, "action_space", original)


def _reference_env(cell: str, size: int | None = None, role: str = "render"):
    """The swm-registered env itself. Needs ``stable_worldmodel``, ``ogbench`` and a GL backend.

    Cached per (cell, size, ROLE) because each ``mujoco.Renderer`` holds a GL context and this family
    would otherwise open one per call. The env is reset once so the model is compiled.

    THE ROLE IS THE FIX FOR A RENDERER CHURN THAT IS UPSTREAM AND STRUCTURAL. OGBench recompiles on
    every reset, and `manipspace/envs/env.py:170-172` closes its `mujoco.Renderer` and builds a new
    one each time it does -- but only if one already exists, because the renderer is created lazily
    on the first `render()`. So an env that is ONLY reset never opens one and never closes one, and
    an env that is only rendered and state-written never closes one either: measured on CPU, five
    `set_state` + `render` cycles the way `reference_render` drives it produce 1 birth and 0 closes,
    while a single reset AFTER a render produces a close and a rebirth.

    The churn came from one env doing both jobs. `sample_states` resets its env 64 times per draw and
    `reference_env()` handed out THE SAME INSTANCE whenever it was called without a size, so every
    draw after the first reference render closed a renderer -- four of them in the gate run, which
    the resource guard attributes to whichever test was running when the collector woke up. Splitting
    the roles means the sampling env is never rendered and the rendering env is never reset.
    """
    key = (cell, int(size) if size else None, str(role))
    hit = _REF_CACHE.get(key)
    if hit is not None:
        return hit
    import gymnasium
    import stable_worldmodel  # noqa: F401  (registers swm/*)
    kw = {}
    if size:
        kw = {"width": int(size), "height": int(size)}
    env = gymnasium.make(SPEC[cell]["swm_id"], **kw)
    env.reset(seed=0)
    _REF_CACHE[key] = env
    return env


class _Bundle:
    """The mover set and the state layout, resolved once per cell; the MjModel, always fresh.

    `model` and `data` are PROPERTIES, not stored handles, and that is not style. Every dirty
    `reset()` runs `compile_model_and_data`, which REBINDS `_model` and `_data` to new objects
    (phase 0 measured `model_object_reused: False` on every reset). A bundle that cached them went
    stale the first time anything reset the env -- and `sample_states` resets it 64 times -- so the
    plate would be baked from, and the button colours written into, a model the env had already
    discarded. Caught by a test that passed alone and failed after any test that resets.

    The derived facts below are cached because they are properties of the MJCF, not of a particular
    compilation: the same scene recompiles to the same geom ids, the same movers and the same state
    layout. Only the array handles move.
    """

    __slots__ = ("cam", "movers", "nq", "nv", "nmocap", "nbuttons",
                 "state_names", "pristine", "size", "unwrapped", "button_tables")

    @property
    def model(self):
        return self.unwrapped.model

    @property
    def data(self):
        return self.unwrapped.data


def _movers(m) -> list[int]:
    """Geoms on a body with degrees of freedom, or on a mocap body. Everything else is the plate.

    The mocap bodies matter: the translucent goal marker is a mocap geom and moves with the goal, so
    a plate that treated it as static would bake the goal into the floor.
    """
    out = []
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i])
        while b > 0:
            if int(m.body_dofnum[b]) > 0 or int(m.body_mocapid[b]) >= 0:
                out.append(i)
                break
            b = int(m.body_parentid[b])
    return out


def _bundle(cell: str) -> _Bundle:
    hit = _BUNDLE_CACHE.get(cell)
    if hit is not None:
        return hit
    import mujoco
    spec = SPEC[cell]
    env = _reference_env(cell, role="sample")
    u = env.unwrapped
    m, d = u.model, u.data
    b = _Bundle()
    b.unwrapped = u
    b.size = int(u._render_height)
    b.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, spec["camera"])
    assert b.cam >= 0, f"{cell}: no camera {spec['camera']!r}"
    b.movers = _movers(m)
    b.nq, b.nv, b.nmocap = int(m.nq), int(m.nv), int(m.nmocap)
    b.nbuttons = int(getattr(u, "_num_buttons", 0) or 0)
    # mocap_quat is NOT in the state either, and for a different reason from qvel: it is CONSTANT
    # under the registered distribution. Measured over 520 states spanning both cells, all five task
    # ids and 12 steps per episode, `data.mocap_quat` has exactly ONE unique row -- the identity
    # (1,0,0,0) -- because `initialize_episode` writes `lie.SO3.identity().wxyz` and nothing in task
    # mode ever rotates the goal marker. A dimension that never varies in the distribution the cell
    # is certified on is not a state dimension: it cannot be perturbed to show it matters, and the
    # repo's own `check_state_dims_matter` reports it dead. The renderer still needs a value, so
    # `_split` and `reference_render` supply the identity explicitly.
    #
    # qvel is NOT in the state. It reaches no pixel -- MJWarp's mover pass runs `kinematics`, which
    # reads qpos and the mocap pose and nothing else -- and Gate A measured exactly that: with qvel
    # carried, 20 of 48 dimensions moved no pixel under a 0.05 perturbation. `check_state_dims_matter`
    # would correctly fail that, and fetch.py made the same call for the same reason when it dropped
    # mocap from ITS state. swm's own `reset(options={'state'})` contract is concat(qpos, qvel), which
    # is why the split helper still understands qvel; the RENDER state is the part that reaches a pixel.
    names = ([f"qpos[{i}]" for i in range(b.nq)]
             + [f"mocap_pos[{i}]" for i in range(3 * b.nmocap)]
             + [f"button_state_{i}" for i in range(b.nbuttons)])
    b.state_names = tuple(names)
    b.pristine = {"geom_group": np.array(m.geom_group, copy=True),
                  "geom_rgba": np.array(m.geom_rgba, copy=True),
                  "mat_rgba": np.array(m.mat_rgba, copy=True)}
    b.button_tables = _button_tables(u, m, b.nbuttons)
    _BUNDLE_CACHE[cell] = b
    return b


def _button_tables(u, m, nbuttons: int) -> dict | None:
    """``{bit tuple: (geom_rgba, mat_rgba)}`` for every button combination, taken from SWM'S OWN CODE.

    The Scene's two button bits are state, and they reach the frame only through colours:
    ``SceneEnv._apply_button_states`` writes ``geom_rgba`` of the two buttons and ``mat_rgba`` of the
    drawer and window handles. They are model fields, so MJWarp reads them when the render context is
    built -- which is why the first version of this cell carried the bits in the state and drew the
    same picture whatever they were. Gate A caught it: both bits moved zero pixels.

    The tables are captured by CALLING ``_apply_button_states`` rather than by reimplementing its
    rule here. Reimplementing would be a second copy of a colour law that already exists, free to
    drift from it silently; calling it means the only thing this file asserts is "these are the
    colours swm produces for these bits", which is checkable by construction.
    """
    if not nbuttons:
        return None
    saved = list(getattr(u, "_cur_button_states", [0] * nbuttons))
    out = {}
    for code in range(1 << nbuttons):
        bits = tuple((code >> k) & 1 for k in range(nbuttons))
        u._cur_button_states = list(bits)
        u._apply_button_states()
        out[bits] = (np.array(m.geom_rgba, copy=True), np.array(m.mat_rgba, copy=True))
    u._cur_button_states = saved
    u._apply_button_states()
    # The tables must actually DIFFER, or the bits are not colours and this design is wrong.
    first = out[tuple([0] * nbuttons)]
    assert any(not np.array_equal(first[0], v[0]) or not np.array_equal(first[1], v[1])
               for v in out.values()), (
        "button states produce identical colour tables: either swm's rule changed or this cell is "
        "capturing the wrong arrays, and carrying the bits in the state would be a lie either way")
    return out


class _Ctx:
    """An MJWarp render context plus the device views a render writes through."""

    __slots__ = ("wm", "wd", "rc", "bgra", "seg", "graph", "nworld", "res", "wp", "mjw",
                 "prepared", "baked", "button_tables")


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


def _context(cell: str, res: int, nworld: int) -> _Ctx:
    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    key = (cell, int(res), int(nworld))
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_CTX_CACHE) >= _CTX_CACHE_MAX:
        _CTX_CACHE.clear()
    b = _bundle(cell)
    # A DEEP COPY, and this is not tidiness. `_bundle.model` is the swm env's OWN MjModel -- the
    # object `reference_render` and `_bake_plate` render the reference through. `prepare_model`
    # mutates in place and is deliberately left applied for the life of the context (MJWarp reads
    # geometry once, at `create_render_context`), so applying it to the shared model would move the
    # REFERENCE as well as ours: the spot correction would rewrite the reference's own light and the
    # cell would "agree" by corrupting the thing it is measured against. On this scene that happens
    # to be invisible -- MuJoCo renders cutoff 100 and 180 identically, which is the whole finding --
    # but agreeing by luck is not agreeing.
    # THE COPY IS TAKEN AT KNOWN BUTTON BITS. `_ctx_button_tables` applies each combination as a
    # delta against the all-zero-bits table, which is only correct if the copy itself is at all-zero
    # bits -- and the reference env is at whatever bits its last reset chose. Those vary: measured
    # over 40 resets, OGBScene starts at (0,0), (1,0) and (1,1), so roughly 40 % of processes would
    # have built every table wrong, silently, because the disjointness assertion cannot see a
    # wrong BASE. The reference is forced to all-zero, copied, and restored.
    zero_bits = [0] * b.nbuttons
    saved_bits = list(getattr(b.unwrapped, "_cur_button_states", []))
    if b.nbuttons:
        b.unwrapped._cur_button_states = list(zero_bits)
        b.unwrapped._apply_button_states()
    m = copy.deepcopy(b.model)
    if b.nbuttons:
        b.unwrapped._cur_button_states = list(saved_bits)
        b.unwrapped._apply_button_states()
    prepared = MW.prepare_model(m, VISIBLE_GROUPS)
    # Textures MJWarp cannot sample at all. On OGBScene this is `btn_metal`, a 512x512 CUBE map on
    # two UV-less button meshes, and without it mujoco_warp <= 3.12.0 dies with an illegal memory
    # access before the first frame. A DECLARED residual: 0.139 % of pixels, interior_exact 1.0.
    baked = MW.bake_unsamplable_textures(m, VISIBLE_GROUPS)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    c = _Ctx()
    c.wp, c.mjw, c.nworld, c.res, c.prepared, c.baked = (
        wp, mjw, int(nworld), int(res), prepared, baked)
    # geom_rgba and mat_rgba are PER WORLD, so the Scene's button bits can differ across the batch.
    # Without this they are one shared array and every world in a launch would wear the same button
    # colours whatever its state said.
    c.wm = mjw.put_model(m, batch_sizes={"geom_rgba": int(nworld), "mat_rgba": int(nworld)})
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # THE CAPTURE IS INSIDE THE PATCH, for the reason fetch.py records: mujoco_warp defines its
    # megakernel inside `render` from whatever `pack_rgba_to_uint32` is bound at that moment, so a
    # graph captured outside this block replays a TRUNCATING kernel and the patch reaches only the
    # warm-up frame.
    with MW.gl_rounding():
        c.rc = MW.create_render_context(
            m, nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True, render_seg=True,
            cam_active=[i == b.cam for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS), use_textures=True, use_shadows=True,
            render_skybox=True)
        _frame(c)
        wp.synchronize()
        # VIEWS of the warp buffers. The channel swap is applied per render rather than here:
        # `[..., [2,1,0]]` at construction is advanced indexing, which copies, and the object would
        # then hold one frozen frame while every later render wrote into the buffer behind it.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(
            int(nworld), int(res), int(res), 4)
        c.seg = wp.to_torch(c.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)[..., 0]
        with wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    c.button_tables = _ctx_button_tables(b, m, res, device="cuda")
    _CTX_CACHE[key] = c
    return c


def _ctx_button_tables(b: "_Bundle", m, res: int, device: str = "cuda") -> dict | None:
    """``{bit code: (geom_rgba, mat_rgba)}`` on the CONTEXT's model, as device tensors.

    The tables in ``_bundle`` were captured from swm's own ``_apply_button_states`` on the REFERENCE
    model. The context renders through a deep copy that has had ``prepare_model`` and
    ``bake_unsamplable_textures`` applied, so its ``mat_rgba`` already differs from the reference's by
    the baked button texture. Copying the reference tables in wholesale would silently undo the bake
    and put the crash back.

    So each table is applied as a DELTA against the reference's own all-zero-bits table, on top of
    the context model's current arrays. The two edits are disjoint by construction -- the bake
    touches ``btn_metal``, the button rule touches the two button geoms and the two handle materials
    -- and the assertion below states that rather than assuming it.
    """
    if not b.nbuttons or b.button_tables is None:
        return None
    # `m` was deep-copied from the reference while it was forced to all-zero bits (see `_context`),
    # so the all-zero table is the right delta base. Asserted below rather than assumed.
    base_g = np.array(m.geom_rgba, copy=True)
    base_m = np.array(m.mat_rgba, copy=True)
    zero = tuple([0] * b.nbuttons)
    ref_g0, ref_m0 = b.button_tables[zero]
    touched = sorted({int(i) for bits, (rg, rm) in b.button_tables.items()
                      for i in np.nonzero(np.any(rg != ref_g0, axis=1))[0]})
    assert np.array_equal(base_g[touched], ref_g0[touched]), (
        "the context model was copied at button bits other than all-zero, so every delta below "
        "would be applied to the wrong base; see _context's forced-zero block")
    # WHAT THE BAKE CHANGED, measured against the right baseline. `base_m` is the context copy:
    # all-zero bits PLUS the bake. `ref_m0` is all-zero bits WITHOUT it. Their difference is
    # therefore exactly the bake, and nothing else.
    #
    # The first version compared `base_m` against `b.pristine["mat_rgba"]`, which was snapshotted at
    # bundle time under whatever bits the env's last reset had chosen. On a Scene that started at
    # (1,0) or (1,1) that difference included the two HANDLE materials -- which the button rule owns
    # and the bake never touches -- so the assertion fired on a conflict that did not exist and
    # killed job 6406972 after OGBCube had already measured cleanly.
    baked_rows = np.nonzero(np.any(base_m != ref_m0, axis=1))[0]
    out = {}
    for bits, (rg, rm) in b.button_tables.items():
        dg, dm = rg - ref_g0, rm - ref_m0
        if len(baked_rows):
            assert not np.any(dm[baked_rows]), (
                f"the texture bake and the button colour rule both write material rows "
                f"{sorted(set(baked_rows) & set(np.nonzero(np.any(dm, axis=1))[0]))}; applying one "
                f"as a delta over the other would undo it")
        code = sum(bit << k for k, bit in enumerate(bits))
        out[code] = (torch.as_tensor(base_g + dg, dtype=torch.float32, device=device),
                     torch.as_tensor(base_m + dm, dtype=torch.float32, device=device))
    return out


#: Per-cell facts. ``swm_id`` is the registration under test; ``camera`` is what swm's own ``render``
#: defaults to (both subclasses override the ogbench base to use ``front_pixels``).
SPEC = {
    # `visible_tri` is the phase-0 census figure for geoms in the rendered groups, used only in the
    # device="cpu" refusal so the message quotes THIS cell's size rather than its sibling's.
    # `state_dim` is a CONSTANT here, not `len(_bundle(cell).state_names)`. Calling `_bundle` from
    # `_make_classes` meant `import stable_envs` built both swm environments: 4.4 s of import time,
    # and without `stable_worldmodel` installed the module raised and BOTH ids vanished from the
    # registry (76 instead of 78). A docstring is not worth either. `tests/unit/test_ogb_manip.py`
    # asserts these agree with what `_bundle` computes, wherever swm IS importable.
    "Cube": {"swm_id": "swm/OGBCube-v0", "camera": "front_pixels", "size": 224,
             "swm_ref": "swm/OGBCube-v0", "visible_tri": 193113, "state_dim": 24},
    "Scene": {"swm_id": "swm/OGBScene-v0", "camera": "front_pixels", "size": 200,
              "swm_ref": "swm/OGBScene-v0", "visible_tri": 196633, "state_dim": 30},
}


class OGBManipBase(StableEnv):
    """Base for the two manipulation cells. ``CELL`` is set by the generated subclasses."""

    CELL: str = ""
    default_supersample = 1
    fidelity_tier = "visual"

    def __init__(self, batch: int = 1, size: int | None = None, supersample: int | None = None,
                 device: str = "cuda", backend: str = "auto", seed: int = 0):
        if str(device) != "cuda":
            # Honest device selection, per the Step-6 policy addendum. There is no CPU backend for
            # this family and the measurement that says why is CPU3D-OGB-1: the binding CPU bar
            # leaves 0.75 ns per triangle for 193,113 triangles, while the Step-6 span rasteriser's
            # SETUP-ONLY floor -- transform, bbox, area, cull, no fill, no shade -- is 24 ns per
            # triangle. That is 12.9x over `cores_cpu >= 1` before a single pixel is written, and
            # 72x at the full measured 135 ns. Falling back to CUDA behind a device="cpu" request is
            # what this rule exists to forbid.
            tri = SPEC.get(self.CELL, {}).get("visible_tri")
            raise ValueError(
                f"{type(self).__name__} has no CPU backend (CPU3D-OGB-1): this scene is "
                f"{tri:,} visible triangles at a median projected edge of ~0.7 px, and the "
                f"certification CPU bar leaves under a nanosecond per triangle against a "
                f"24 ns/triangle vertex-setup floor -- 9.9-12.9x short before a single pixel is "
                f"filled. See examples/results/step7/ogb-manipulation/findings.md. "
                f"Use device='cuda'.")
        b = _bundle(self.CELL)
        self.state_dim = len(b.state_names)
        self.state_names = b.state_names
        super().__init__(batch=batch, size=size or b.size,
                         supersample=self.default_supersample if supersample is None else supersample,
                         device=device, backend=backend, seed=seed)
        self._b = b
        self._check_plate_is_constant()

    # ---- the assumption the whole design rests on, checked rather than assumed -------------

    def _check_plate_is_constant(self) -> None:
        """With every mover hidden, the frame must not depend on the state. Asserted, once per class.

        Phase 0 measured it bit-static over 6 states, but "measured once in a probe" is not the same
        as "true of the model this class just built", and Fetch's own history is the argument: a
        site left in the static set made its plate depend on ``qpos[15:17]`` over 1123 pixels.
        Cheap: two states, 128 px, one MuJoCo render each.
        """
        st = self.sample_states(2, rng=np.random.default_rng(0))
        a = self._bake_plate(128, states=st[0])
        c = self._bake_plate(128, states=st[1])
        d = int((a.to(torch.int16) - c.to(torch.int16)).abs().max())
        assert d == 0, (
            f"{type(self).__name__}: the baked static plate differs by {d}/255 between two states "
            f"with every mover hidden, so the floor and skybox are NOT a function of the appearance "
            f"alone and the plate design is unsound for this scene")

    # ---- state ----------------------------------------------------------------------------

    def _split(self, s: np.ndarray):
        """``(state_dim,)`` -> qpos, mocap_pos, mocap_quat, button bits. No qvel: see ``_bundle``."""
        b = self._b
        i = 0
        qpos = s[i:i + b.nq]; i += b.nq
        mp = s[i:i + 3 * b.nmocap].reshape(b.nmocap, 3); i += 3 * b.nmocap
        bits = s[i:i + b.nbuttons]
        return qpos, mp, IDENTITY_QUAT(b.nmocap), bits

    def _variation_values(self, rng: np.random.Generator) -> dict:
        """Explicit values for the registered default variation leaves, drawn from ``rng``.

        `reset_variation_space` is called by swm with `seed=None`, so the leaves it resamples come
        from OS entropy and no reset is reproducible. It does honour an explicit
        `options['variation_values']`, which is the one handle that makes the draw ours. Uniform over
        each leaf's own bounds, which is what `Box.sample` would have done, so the distribution is
        unchanged and only its generator moves.
        """
        vs = getattr(self._b.unwrapped, "variation_space", None)
        if vs is None:
            return {}
        out = {}
        for path in DEFAULT_VARIATION_PATHS.get(self.CELL, ()):
            node = vs
            try:
                for part in path.split("."):
                    node = node[part]
            except Exception:
                continue
            lo = np.asarray(node.low, np.float64)
            hi = np.asarray(node.high, np.float64)
            out[path] = rng.uniform(lo, hi).reshape(np.asarray(node.value).shape)
        return out

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States a dataset would actually hold: reset, then 0-25 random actions.

        Reset states alone do not exercise this env -- every one has the arm at its home pose and
        only the cube moved -- so a fidelity number computed from them would say nothing about the
        14 arm joints that are most of the picture. The same argument fetch.py records.
        """
        import mujoco
        rng = np.random.default_rng(0) if rng is None else rng
        b = self._b
        env = _reference_env(self.CELL, role="sample")   # reset many times; never rendered
        u = env.unwrapped
        out = []
        # THE GLOBAL NUMPY RNG IS SEEDED FROM `rng`, and this is not belt-and-braces. swm's
        # `CubeEnv.reset` calls `reset_variation_space(..., seed=None, ...)` and the task is picked
        # with the module-level `np.random`, so two calls with the same `rng` drew DIFFERENT states
        # and `test_seeded_sampling_reproduces_in_a_fresh_process` could not hold. Seeding the global
        # generator is the only handle swm exposes; its prior state is saved and restored so this
        # function does not become a side effect on the caller's own draws.
        gstate = np.random.get_state()
        try:
            for k in range(int(batch)):
                sd = int(rng.integers(0, 1 << 30))
                np.random.seed(int(rng.integers(0, 1 << 31)))
                # THE SEED `reset` DROPS. `CubeEnv.reset(self, seed=None, ...)` and
                # `SceneEnv.reset` accept `seed` and then call `super().reset(options=options, ...)`
                # WITHOUT it (cube_env.py:822, scene_env.py:325), so `gym.Env.reset` never reseeds
                # `self.np_random` and `reset(seed=...)` has no effect at all. Everything random in
                # `initialize_episode` -- the arm's initial yaw, the cube jitter, the block
                # permutation -- runs off that generator. Setting it directly is the only way to make
                # a draw reproducible, and with it the whole reset is bit-reproducible (verified).
                u.np_random = np.random.default_rng(sd)
                with _seeded_action_space(u, sd):
                    env.reset(seed=sd, options={"variation_values": self._variation_values(rng)})
                    m, d = u.model, u.data
                    for _ in range(int(rng.integers(0, 26))):
                        env.step(u.action_space.sample())
                bits = [float(x) for x in list(getattr(u, "_cur_button_states", []))[:b.nbuttons]]
                out.append(np.concatenate([d.qpos.copy(), d.mocap_pos.ravel().copy(),
                                           np.asarray(bits, np.float64)]))
        finally:
            np.random.set_state(gstate)
        # FLOAT32, and this is a correctness requirement rather than a memory one.
        # `StableEnv.set_state` stores the batch as float32, so `check_state_roundtrip` compares a
        # float32 read-back against whatever this returns; returning float64 made
        # `set_state(get_state())` differ by the narrowing error -- 1.17e-07, against a bound the
        # suite spells as exactly 0.0. The state is carried in the dtype it round-trips in.
        return np.stack(out).astype(np.float32)

    # ---- variation ------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """The REGISTERED DEFAULT variation set, and nothing else yet.

        swm's own space has appearance leaves -- ``floor.color``, ``light.intensity``,
        ``camera.angle_delta``, ``cube.size``, ``agent.color`` and (Scene) ``lock_color`` -- and each
        of them makes the baked plate a function of the variation vector rather than a constant. They
        are tractable exactly as the DM Control cells handle theirs, as per-env tables keyed into the
        bake cache, and they are deliberately OUT of scope until parity is established: widening the
        bake's cache key silently is how a plate stops being the thing that was asserted constant.
        The Gate-F gap list says so explicitly rather than leaving it to be inferred.

        What IS in the registered default set -- ``cube.start_position``, ``cube.start_yaw``,
        ``cube.goal_position``, ``cube.goal_yaw`` -- reaches the frame only through ``qpos`` and the
        mocap pose, which are state, not appearance. That is why this space is empty and the plate is
        bit-static: measured in phase 0 across resets, every ``MjModel`` appearance array is
        identical.
        """
        return Dict({})

    # ---- the plate ------------------------------------------------------------------------

    def _plate(self, res: int) -> torch.Tensor:
        key = (self.CELL, int(res), str(self.device))
        hit = _PLATE_CACHE.get(key)
        if hit is not None:
            return hit
        if len(_PLATE_CACHE) >= _PLATE_CACHE_MAX:
            _PLATE_CACHE.clear()
        out = _PLATE_CACHE[key] = self._bake_plate(res)
        return out

    def _bake_plate(self, res: int, states=None) -> torch.Tensor:
        """MuJoCo's own raster of the static scene: the floor plane and the skybox.

        Exact because MuJoCo drew it -- and only reaching the frame where the mover LUT says MJWarp
        did not hit a mover. Measured worth, three 64-state draws: interior-exact 0.958 (Cube) and
        0.922 (Scene), whole-frame 0.872 and 0.775.

        Drawn through the rig's PRIVATE model and data, so this mutates nothing the reference env or
        anything else can see, and the "every appearance array is identical across resets" claim the
        cached plate rests on is asserted here rather than assumed.
        """
        import mujoco
        b = self._b
        rig = _plate_rig(self.CELL, int(res))
        live = b.model
        shape = (int(live.ngeom), int(live.nq), int(live.nmocap))
        assert shape == rig.shape, (
            f"{self.CELL}: the env recompiled to a different model shape {shape} than the plate rig "
            f"was built for {rig.shape}; the cached plate would be a picture of the old scene")
        now = _appearance_digest(live)
        assert now == rig.appearance, (
            f"{self.CELL}: an appearance array changed since the plate rig was built "
            f"({rig.appearance} -> {now}). The cached plate is a picture of the scene as it was, so "
            f"it is now wrong; phase 0 measured these arrays identical across resets and this "
            f"assertion is what keeps that measurement honest")
        d = rig.data
        if states is None:
            d.qpos[:] = b.data.qpos
            d.mocap_pos[:] = b.data.mocap_pos
            d.mocap_quat[:] = b.data.mocap_quat
        else:
            qpos, mp, mq, _ = self._split(np.asarray(states, np.float64))
            d.qpos[:] = qpos
            d.mocap_pos[:] = mp
            d.mocap_quat[:] = mq
        d.qvel[:] = 0.0
        mujoco.mj_forward(rig.model, d)
        rig.renderer.update_scene(d, camera=b.cam, scene_option=b.unwrapped._scene_option)
        return torch.as_tensor(rig.renderer.render().copy(), device=self.device)

    # ---- render ---------------------------------------------------------------------------

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8 on device.

        One MJWarp launch per chunk for the movers, composited over the single baked plate by
        MJWarp's own segmentation buffer -- not by a depth heuristic, so the arm being occluded by
        the cube is resolved by the ray-caster that drew them.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        plate = self._plate(res)
        plate1 = plate if ss == 1 else self._plate(size)
        lut = self._mover_lut()
        b = self._b
        nq, nmoc = b.nq, b.nmocap

        for lo in range(0, B, MJW_CHUNK):
            hi = min(lo + MJW_CHUNK, B)
            n = hi - lo
            nw = 1 if n <= 1 else min(MJW_CHUNK, 1 << (n - 1).bit_length())
            c = _context(self.CELL, res, nw)
            chunk = s[lo:hi]
            wp = c.wp
            wp.to_torch(c.wd.qpos)[:n] = chunk[:, :nq]
            if nmoc:
                # mocap_pos IS state; mocap_quat is NOT -- it is the identity everywhere in the
                # registered distribution (see `_bundle`), so it is written as a constant rather
                # than sliced out of a state vector that no longer carries it. Slicing it anyway is
                # how job 6405900 died: `chunk[:, nq+3*nmoc : nq+7*nmoc]` is empty once the dim is
                # dropped, and the reshape raised `shape '[64, 1, 4]' is invalid for input of size 0`.
                mp = chunk[:, nq: nq + 3 * nmoc].reshape(n, nmoc, 3)
                wp.to_torch(c.wd.mocap_pos)[:n] = mp
                wp.to_torch(c.wd.mocap_quat)[:n] = self._identity_quat(nmoc, n)
            if b.nbuttons:
                self._upload_button_colours(c, chunk, n)
            # torch wrote the per-world arrays on its stream; warp replays the graph on its own.
            # The two syncs bracket the launch so a frame can never be rendered from half-written
            # state -- microseconds against the difference between deterministic and nearly so.
            torch.cuda.synchronize()
            wp.capture_launch(c.graph)
            wp.synchronize()
            rgb, seg = c.bgra[:n][..., [2, 1, 0]], c.seg[:n]
            # THE MOVER LUT, not `seg >= 0`. The floor plane and the skybox are geoms too, so
            # `seg >= 0` is true wherever MJWarp hit ANYTHING and the composite then takes MJWarp's
            # floor over the baked plate -- which is the plate never being used except on sky. The
            # first version of this cell did exactly that, and its "0.92-0.95 composite exactness"
            # was measured by a probe that masked with the mover set while the renderer did not.
            # `index_select` on the int32 segmentation rather than `lut[seg.long() + 1]`: the latter
            # materialises a 64-bit copy of the whole frame stack to index with.
            mv = lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
            comp = torch.where(mv.unsqueeze(-1), rgb, plate)
            if ss == 1:
                buf[lo:hi] = comp
            else:
                small = (comp.to(torch.float32)
                         .reshape(n, size, ss, size, ss, 3).mean((2, 4)).round_()
                         .clamp_(0, 255).to(torch.uint8))
                any_mv = mv.reshape(n, size, ss, size, ss).any(4).any(2)
                buf[lo:hi] = torch.where(any_mv.unsqueeze(-1), small, plate1)
        return buf.view(*lead, size, size, 3) if lead else buf

    def _identity_quat(self, nmoc: int, n: int) -> torch.Tensor:
        """``(n, nmoc, 4)`` of ``(1,0,0,0)`` on device, built once per (nmoc, batch)."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        key = ("idq", int(nmoc), int(n))
        hit = cache.get(key)
        if hit is None:
            q = torch.zeros((int(n), int(nmoc), 4), dtype=torch.float32, device=self.device)
            q[..., 0] = 1.0
            hit = cache[key] = q
        return hit

    def _mover_lut(self) -> torch.Tensor:
        """``(ngeom+1,)`` bool: is this segmentation id a MOVER? Indexed, not searched.

        `torch.isin(seg, movers)` touches the frame once per mover; a table lookup touches it once.
        The +1 offset makes segmentation id -1 (nothing hit) index row 0, which is False, so sky
        falls through to the plate like every other non-mover pixel.
        """
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("mover_lut")
        if hit is None:
            n = int(self._b.model.ngeom)
            lut = torch.zeros(n + 1, dtype=torch.bool, device=self.device)
            lut[torch.as_tensor(np.asarray(self._b.movers, np.int64) + 1, device=self.device)] = True
            hit = cache["mover_lut"] = lut
        return hit

    def _upload_button_colours(self, c, chunk, n: int) -> None:
        """Write each world's button colours from its own state bits.

        The tables come from ``swm``'s own ``_apply_button_states`` (see ``_button_tables``) and are
        re-baked here through the context's copy, because that copy has had
        ``bake_unsamplable_textures`` applied to it and its ``mat_rgba`` therefore differs from the
        reference's by the baked button texture.
        """
        b = self._b
        bits = chunk[:, -b.nbuttons:].round().clamp(0, 1).to(torch.int64).cpu().numpy()
        codes = (bits * (1 << np.arange(b.nbuttons))).sum(1)
        geom = c.wp.to_torch(c.wm.geom_rgba)
        mat = c.wp.to_torch(c.wm.mat_rgba)
        for code in np.unique(codes):
            sel = torch.as_tensor(np.nonzero(codes == code)[0], device=self.device)
            g, mt = c.button_tables[int(code)]
            geom.index_copy_(0, sel, g.unsqueeze(0).expand(len(sel), *g.shape))
            mat.index_copy_(0, sel, mt.unsqueeze(0).expand(len(sel), *mt.shape))

    # ---- the reference --------------------------------------------------------------------

    @classmethod
    def reference_env(cls, size: int | None = None):
        return _reference_env(cls.CELL, size, role="render")

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """The official frames for a state stack, through swm's own ``render``.

        The button bits go through ``SceneEnv.set_state``'s required keyword arguments, which is the
        only way swm exposes them; ``mocap_pos``/``mocap_quat`` are written directly because nothing
        in swm's API takes them.
        """
        import mujoco
        u = env.unwrapped
        b = _bundle(cls.CELL)
        out = []
        for s in np.asarray(states, np.float64).reshape(-1, len(b.state_names)):
            i = 0
            qpos = s[i:i + b.nq]; i += b.nq
            qvel = np.zeros(b.nv)
            mp = s[i:i + 3 * b.nmocap].reshape(b.nmocap, 3); i += 3 * b.nmocap
            mq = IDENTITY_QUAT(b.nmocap)
            bits = s[i:i + b.nbuttons]
            if b.nbuttons:
                u.set_state(qpos, qvel,
                            **{f"button_state_{k}": int(round(float(bits[k])))
                               for k in range(b.nbuttons)})
            else:
                u.set_state(qpos, qvel)
            u.data.mocap_pos[:] = mp
            u.data.mocap_quat[:] = mq
            mujoco.mj_forward(u.model, u.data)
            out.append(u.render().copy())
        return np.stack(out)


_PLATE_RIGS: dict = {}

#: Model arrays the plate depends on, by prefix. `geom_rgba` and `mat_rgba` are EXCLUDED because
#: swm's `_apply_button_states` writes them into the live model on every button change -- they are
#: state wearing the shape of appearance, and the rig pins them itself (see `_plate_rig`).
_APPEARANCE_PREFIXES = ("geom_", "mat_", "tex", "light_", "cam_", "mesh_", "body_", "site_",
                        "hfield_", "skin_")
_APPEARANCE_SKIP = ("geom_rgba", "mat_rgba", "geom_group")


def _appearance_digest(m) -> str:
    """A hash of every static-appearance array in an MjModel, so "the plate is bit-static" is CHECKED.

    Phase 0 measured that a dirty reset recompiles this scene to identical appearance arrays and only
    moves the array handles. The plate rig depends on that measurement being true for the life of the
    process, and a measurement that nothing re-checks is an assumption. This turns it into an
    assertion that fires loudly instead of baking a plate from geometry the env has since changed.
    """
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    n = 0
    for name in sorted(dir(m)):
        if name.startswith("_") or name in _APPEARANCE_SKIP:
            continue
        if not name.startswith(_APPEARANCE_PREFIXES):
            continue
        try:
            v = getattr(m, name)
        except Exception:                      # pragma: no cover - MjModel exposes only arrays here
            continue
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(name.encode())
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
            n += 1
    assert n >= 20, (f"appearance digest covered only {n} arrays -- an empty hash would certify "
                     f"staleness it never looked at")
    return h.hexdigest()


class _PlateRig:
    """A PRIVATE (MjModel, MjData, Renderer) triple that bakes this cell's plate, built once.

    ONE EGL context per (cell, resolution), for the life of the process, never released. The previous
    version keyed the renderer on ``id(model)`` and evicted the old entry whenever a dirty reset
    rebound the env's model -- which dropped the last reference to a live ``mujoco.Renderer``, so
    ``__del__`` ran ``close()`` and freed its EGL context. Its docstring said "never closed" while the
    code guaranteed the opposite: the gate run counted 14 contexts created and 8 released, and the
    resource guard failed 11 of the 15 OGB rows because of it. Releasing one context corrupts
    rendering in every other live context in the process, silently, which is the defect
    ``envs/fetch.py`` and ``envs/ogbench.py`` both exist to avoid; ``fetch.py`` avoids it by owning
    its model rather than borrowing one that rebinds, and this is the same answer.

    Owning the model also removes two hazards the borrowed one carried. The rig renders its OWN
    ``MjData``, so a bake no longer mutates the reference env's model and restores it afterwards --
    nothing can leave the reference corrupted if a bake raises in between. And the rig pins the
    button colours at the ALL-ZERO bits, the same convention ``_context`` takes its copy at, so the
    plate is baked at a known appearance instead of at whichever bits the last reset happened to
    choose; today's code baked whatever was live when the cache first filled.
    """

    __slots__ = ("model", "data", "renderer", "appearance", "shape")


def _plate_rig(cell: str, res: int) -> _PlateRig:
    import copy

    import mujoco
    key = (cell, int(res))
    hit = _PLATE_RIGS.get(key)
    if hit is not None:
        return hit
    b = _bundle(cell)
    live = b.model
    rig = _PlateRig()
    rig.appearance = _appearance_digest(live)
    rig.shape = (int(live.ngeom), int(live.nq), int(live.nmocap))
    rig.model = copy.deepcopy(live)
    # Movers are hidden for the life of the rig -- this model exists only to draw the static scene.
    rig.model.geom_group[b.movers] = MW.HIDDEN_GROUP
    zero = (b.button_tables or {}).get(tuple([0] * b.nbuttons))
    if zero is not None:
        rig.model.geom_rgba[:], rig.model.mat_rgba[:] = zero
    else:
        rig.model.geom_rgba[:] = b.pristine["geom_rgba"]
        rig.model.mat_rgba[:] = b.pristine["mat_rgba"]
    rig.data = mujoco.MjData(rig.model)
    rig.renderer = mujoco.Renderer(rig.model, height=int(res), width=int(res))
    _PLATE_RIGS[key] = rig                      # never evicted; the renderer is never closed
    return rig


#: MEASURED, not chosen. Three independent draws of 64 states each on an RTX 3090 (job recorded in
#: ``examples/results/step7/ogb-manipulation/m3/fidelity.json``), against swm's own EGL frames at the
#: registered resolution. The declared value is the WORST of the three draws; the threshold beside it
#: is a round number with margin, set by the same discipline ``envs/fetch.py`` states for its
#: ``parity_interior_min``: a threshold is a contract about what the class guarantees, not a running
#: best, and a threshold equal to a freshly measured value converts a comfortable margin into a
#: tripwire for the next backend change. Filled in from the measurement; ``SE7_OGB_UNMEASURED=1``
#: exists ONLY so the measuring process can import this module before the numbers exist, and it makes
#: the placeholder impossible to mistake for evidence.
import os as _os

#: MEASURED by `scratch/step7/ogb-manipulation/probe_remeasure_chain.py`, job 6407605 on
#: gpu2109 (RTX 3090), three independent 64-state draws per cell against swm's own EGL frames at
#: the registered resolution. The declared value is the WORST of the three and `draw_spread` carries
#: the range; the thresholds were computed by that script under the family rule (x1.3 on the mean,
#: the fractional floors a round 0.05 step below the worst draw), not set by hand here.
#:
#: These supersede the numbers this file carried before the mover-LUT fix. Those described a
#: composite that never used its own baked plate on the floor, and the difference is the measure of
#: that defect: interior-exact 0.351 -> 0.958 and all-exact 0.278 -> 0.872 on the Cube.
_MEASURED: dict = {
        "Cube": {
            "all_exact_frac": 0.870342,
            "draw_spread": {
                "mean_abs": [
                    2.408967,
                    2.441658
                ],
                "ssim": [
                    0.914917,
                    0.923417
                ],
                "within8_frac": [
                    0.915477,
                    0.916856
                ]
            },
            "interior_exact_frac": 0.958007,
            "interior_frac": 0.411172,
            "job": "job 6407605 on gpu2109.oscar.ccv.brown.edu",
            "mean_abs": 2.441658,
            "mean_abs_max": 3.2,
            "ssim": 0.914917,
            "ssim_min": 0.9,
            "within8_frac": 0.915477,
            "within8_min": 0.9
        },
        "Scene": {
            "all_exact_frac": 0.771891,
            "draw_spread": {
                "mean_abs": [
                    5.327423,
                    5.381315
                ],
                "ssim": [
                    0.892819,
                    0.897032
                ],
                "within8_frac": [
                    0.85675,
                    0.857355
                ]
            },
            "interior_exact_frac": 0.917805,
            "interior_frac": 0.2769,
            "job": "job 6407605 on gpu2109.oscar.ccv.brown.edu",
            "mean_abs": 5.381315,
            "mean_abs_max": 7.0,
            "ssim": 0.892819,
            "ssim_min": 0.85,
            "within8_frac": 0.85675,
            "within8_min": 0.85
        }
    }

_UNMEASURED = str(_os.environ.get("SE7_OGB_UNMEASURED", "")) == "1"

#: The REFERENCE side: swm's `render()` goes through `mujoco.Renderer`, so mujoco is the version the
#: certifier must compare. swm and ogbench are deliberately absent -- neither number was measured
#: against a named version of either, and a version nobody measured under is not provenance.
_REFERENCE_VERSIONS = {"mujoco": "3.12.0", "ogbench": "1.2.1"}
#: OUR side: the batch raytracer that drew the movers.
_MEASURED_WITH = {"mujoco_warp": "3.12.0"}


def _fidelity(cell: str) -> dict:
    if cell not in _MEASURED:
        if not _UNMEASURED:
            raise RuntimeError(
                f"ogb_manip: {cell} has no MEASURED visual_fidelity. This module refuses to register "
                f"a visual cell with invented numbers. Run "
                f"scratch/step7/ogb-manipulation/09_m3_fidelity.sbatch, paste its worst-of-three into "
                f"_MEASURED, and re-import. Set SE7_OGB_UNMEASURED=1 only inside that measuring job.")
        return {"mean_abs": -1.0, "within8_frac": -1.0, "ssim": -1.0,
                "mean_abs_max": -1.0, "within8_min": -1.0, "ssim_min": -1.0,
                "measured_at": "UNMEASURED PLACEHOLDER - SE7_OGB_UNMEASURED=1 is set, so these are "
                               "NOT measurements and this build must not ship or be certified."}
    d = _MEASURED[cell]
    spec = SPEC[cell]
    lo_w, hi_w = d["draw_spread"]["within8_frac"]
    lo_s, hi_s = d["draw_spread"]["ssim"]
    return {
        "mean_abs": float(d["mean_abs"]),
        "within8_frac": float(d["within8_frac"]),
        "ssim": float(d["ssim"]),
        # Reported for honesty, threshold nowhere -- the DM Control convention. The interior mask on
        # this scene keeps 0.41 (Cube) / 0.28 (Scene) of pixels, so unlike DM Control's speckled
        # skybox it is defined here, but it is still a statement about the plate rather than about
        # the mover pass.
        # WORST of the three draws, like the three headline metrics -- not the mean. The first
        # version declared means beside a `measured_at` that said "the WORST of the three", which is
        # a small number and a wrong sentence (Cube interior 0.958115 mean vs 0.958007 worst).
        "interior_exact_frac": float(d["interior_exact_frac"]),
        "interior_frac": float(d["interior_frac"]),
        "all_exact_frac": float(d["all_exact_frac"]),
        "draw_spread": d["draw_spread"],
        # THE FAMILY RULE, not a self-invented one: x1.3 on the mean, and the fractional floors a
        # round step below the WORST draw. DM Control sets ceilings this way (mean 5.01 -> 6.50,
        # 8.07 -> 10.5) precisely so a threshold is a contract rather than a tripwire on the next
        # state draw, and the spread over three draws is what it is set clear of.
        # x1.3 on the mean; the fractional floors the largest 0.05-grid value at least 0.04 BELOW
        # the worst draw. The bare `floor(x*20)/20` this used before gave margins of 0.007-0.016
        # whenever the worst draw sat just above a grid line -- a threshold that close to a measured
        # value is the tripwire the family rule exists to avoid, and DM Control's own floors sit
        # 0.04-0.06 under. Stated as a rule rather than nudged by feel.
        "mean_abs_max": round(float(d["mean_abs"]) * 1.3, 1),
        "within8_min": float(np.floor((lo_w - 0.04) * 20) / 20),
        "ssim_min": float(np.floor((lo_s - 0.04) * 20) / 20),
        "measured_at": (
            # The reference version is NAMED here as well as declared in `reference_versions`:
            # `tests/unit/test_declarations.py` requires the two to agree in words, because a
            # version in a field nobody reads next to prose that names a different one is worse
            # provenance than either alone.
            f"mujoco 3.12.0 / ogbench 1.2.1 reference; {spec['size']} px, camera "
            f"{spec['camera']}, 3 independent "
            f"draws of 64 states from the "
            f"env's own pool (reset + 0-25 random actions) on an RTX 3090 under MUJOCO_GL=egl, "
            f"against swm's own render(); the declared value is the WORST of the three and "
            f"draw_spread carries all three, for mean_abs/within8_frac/ssim AND for "
            f"interior_exact_frac, all_exact_frac and interior_frac. The frame is a baked MuJoCo "
            f"plate composited with "
            f"MJWarp movers through the mover segmentation LUT, with gl_rounding and the shared "
            f"out-of-range-spot correction applied. {d['job']}"),
        # The REFERENCE side, which the certifier compares: mujoco is what swm's render() draws with.
        "reference_versions": _REFERENCE_VERSIONS,
        # OUR side, recorded and never compared: a bump of our own renderer that moves these numbers
        # is a regression to be seen, not a reason to block the check.
        "measured_with": _MEASURED_WITH,
    }


_RESIDUAL_NOTE = (
    "THREE declared residuals, all attributed by measurement before this class was written, and all "
    "structural to mujoco_warp rather than to this scene. (1) THE `global` DIRECTIONAL LIGHT is the "
    "largest remaining term: neutralising it on both sides moves interior exactness 0.347 -> 0.702. "
    "It is the castshadow light, and turning shadows OFF makes agreement worse, so the disagreement "
    "is in the light and not in its shadow. (2) THE SPOT'S DISTANCE ATTENUATION: after the shared "
    "out-of-range-cutoff correction, removing the spot entirely still adds +0.113, because MJWarp "
    "takes attenuation = 1/(1,d,d^2).(1,0,0) = 1 and MuJoCo evidently does something else with the "
    "same light_attenuation. (3) THE HEADLIGHT IS A MOVER-PIXEL TERM: interior barely moves (+0.001) "
    "while mover exactness goes 0.757 -> 0.896, which is what a light along the camera axis does. "
    "A fourth thing is NOT a residual and was checked: ambient. Zeroing it makes agreement WORSE, so "
    "MJWarp already agrees with MuJoCo about it. "
    "ALSO DECLARED, AND SEPARATE FROM THE SHADING: the translucent goal marker `target_object_0` "
    "renders at alpha 0.2 and mujoco_warp's shader reads only color[0..2], so it has no alpha path "
    "and would paint the marker opaque. Measured share of the frame: 0.61% (Cube) / 0.15% (Scene), "
    "against the 1.2-1.5% Fetch already declares for the same mechanism. "
    "WHERE THE RESIDUAL IS NOT: the floor. It is the baked plate, composited in wherever the mover "
    "LUT says MJWarp did not hit a mover, and the measured result is interior-exact 0.958 (Cube) / "
    "0.922 (Scene) and whole-frame 0.872 / 0.775 over three independent 64-state draws. An earlier "
    "version of this cell masked with `seg >= 0` -- true wherever MJWarp hit ANYTHING, floor "
    "included -- so the plate reached only the sky and the same scene scored 0.351 interior-exact. "
    "That composite bug was worth more than every lighting term above combined.")


#: The Scene carries one residual the Cube does not, and it is declared per geom because that is what
#: the bake helper's record is for.
_SCENE_BAKE_NOTE = (
    " SCENE ONLY, A FOURTH DECLARED RESIDUAL: the material `btn_metal` wears a 512x512 CUBE map on "
    "two UV-less button meshes (`stopbuttonrim`, geoms //unnamed_geom_8 and //unnamed_geom_20), and "
    "mujoco_warp <= 3.12.0 indexes `mesh_texcoord` unconditionally in its MESH branch, so it dies "
    "with an illegal memory access before the first frame rather than drawing the scene badly. The "
    "texture is baked to its mean colour (0.523, 0.516, 0.512) through the shared "
    "`bake_unsamplable_textures`. MEASURED cost, MuJoCo against MuJoCo over 8 states at 200 px so no "
    "MJWarp term is mixed in: the two button geoms are 36 px/frame (0.090% of the frame), the bake "
    "changes 55.5 px/frame (0.139%), max 37/255, mean 0.023/255 over the frame, and interior_exact "
    "against the official frame is 1.00000 -- every changed pixel fails the 3x3-flat test, so the "
    "residual is invisible to the interior metric. For scale, the alpha note Fetch already declares "
    "is 1.2-1.5% of pixels. THE ZERO-PIXEL ALTERNATIVE EXISTS AND IS NOT TAKEN HERE: upstream "
    "mujoco_warp guards that branch from 3.13.0 (58d8c77, PR #1608), so a venv bump removes the need "
    "for this bake entirely -- to be measured, not assumed, since 3.13.0 moves more than one thing.")


def _make_classes():
    made = {}
    for cell, spec in SPEC.items():
        cname = f"OGB{cell}"
        made[cname] = register(type(cname, (OGBManipBase,), dict(
            CELL=cell,
            name=f"ogb-{cell.lower()}",
            reference_id=spec["swm_ref"],
            default_size=spec["size"],
            visual_fidelity=_fidelity(cell),
            visual_note=_RESIDUAL_NOTE + (_SCENE_BAKE_NOTE if cell == "Scene" else ""),
            parity_interior_min=None,
            __doc__=(
                f"OGBench manipulation {cell} ({spec['swm_ref']}), fidelity_tier='visual'.\n\n"
                f"The frame is a MuJoCo-baked static plate composited with mujoco_warp's batch "
                f"ray-caster for the movers, at swm's registered {spec['size']} px from the model "
                f"camera {spec['camera']!r}. State is (qpos, mocap_pos"
                + (", button bits)" if cell == "Scene" else ")")
                + f", {spec['state_dim']} dimensions. qvel is excluded because it "
                  f"reaches no pixel and mocap_quat because it is the identity everywhere in the "
                  f"registered distribution.\n\n"
                f"THIS IS A VISUAL TIER CELL, NOT A PARITY ONE. Three lighting terms in mujoco_warp "
                f"disagree with MuJoCo and are declared in visual_fidelity and visual_note rather "
                f"than chased: the global directional light, the spot's distance attenuation, and "
                f"the headlight on mover pixels. A fourth, the out-of-range spot cutoff, WAS fixed "
                f"and now lives in the shared render/mjwarp.py path. The translucent goal marker is "
                f"a fifth, separate residual: mujoco_warp has no alpha path.\n\n"
                f"There is NO CPU backend: device='cpu' raises rather than silently using the GPU "
                f"(CPU3D-OGB-1)."),
        )))
    return made


globals().update(_make_classes())
