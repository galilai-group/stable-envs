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

THE NUMBER THAT ACTUALLY SHIPS IS NOT 0.35
-------------------------------------------

The interior mask on this scene is **99.2 % floor**, and the floor is the plate — baked through
MuJoCo and exact BY CONSTRUCTION, asserted per class rather than assumed (``_check_plate_is_constant``
below; phase 0 measured it bit-static over 6 states with every mover hidden). Only the **10-19 %** of
pixels that are movers are MJWarp's to draw. Measured whole-frame exactness of the COMPOSITE is
**0.92-0.95**, and that is what a caller gets. The 0.35 is a statement about a floor this class never
asks MJWarp to draw, and quoting it as the cell's fidelity would be as wrong as quoting the 0.95 as
parity.

STATE
-----

``(qpos, mocap_pos, mocap_quat)`` — and for the Scene, two extra dimensions for the button bits.
Measured in phase 0: the frame is a bit-exact function of exactly that, verified by imposing a state
from one episode onto a DIFFERENT episode (different seed, recompiled model) and getting an identical
frame. The mocap pose is what positions the translucent goal marker, and it is NOT in swm's info
dict, so a caller has to read it off ``data``.

``qvel`` is deliberately NOT in the state, though swm's own ``reset(options={'state'})`` contract is
``concat(qpos, qvel)``. It reaches no pixel — the mover pass runs ``kinematics``, which reads ``qpos``
and the mocap pose — and Gate A measured exactly that: carrying it, 20 of 48 dimensions moved no
pixel under a 0.05 perturbation, which ``check_state_dims_matter`` would correctly fail. ``fetch.py``
made the same call when it dropped mocap from its own state.

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
#: MJWarp worlds per launch. Same constant and the same reason as Fetch: a context is per (res,
#: nworld) and an unbounded batch would build one per batch size.
MJW_CHUNK = 256
_CTX_CACHE: dict = {}
_CTX_CACHE_MAX = 4
_PLATE_CACHE: dict = {}
_PLATE_CACHE_MAX = 8
_BUNDLE_CACHE: dict = {}
_REF_CACHE: dict = {}


def _reference_env(cell: str, size: int | None = None):
    """The swm-registered env itself. Needs ``stable_worldmodel``, ``ogbench`` and a GL backend.

    Cached per (cell, size) because each ``mujoco.Renderer`` holds a GL context and this family
    would otherwise open one per call. The env is reset once so the model is compiled.
    """
    key = (cell, int(size) if size else None)
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
    """The model, the mover set and the state layout, resolved once per cell."""

    __slots__ = ("model", "data", "cam", "movers", "nq", "nv", "nmocap", "nbuttons",
                 "state_names", "pristine", "size", "unwrapped", "button_tables")


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
    env = _reference_env(cell)
    u = env.unwrapped
    m, d = u.model, u.data
    b = _Bundle()
    b.model, b.data, b.unwrapped = m, d, u
    b.size = int(u._render_height)
    b.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, spec["camera"])
    assert b.cam >= 0, f"{cell}: no camera {spec['camera']!r}"
    b.movers = _movers(m)
    b.nq, b.nv, b.nmocap = int(m.nq), int(m.nv), int(m.nmocap)
    b.nbuttons = int(getattr(u, "_num_buttons", 0) or 0)
    # qvel is NOT in the state. It reaches no pixel -- MJWarp's mover pass runs `kinematics`, which
    # reads qpos and the mocap pose and nothing else -- and Gate A measured exactly that: with qvel
    # carried, 20 of 48 dimensions moved no pixel under a 0.05 perturbation. `check_state_dims_matter`
    # would correctly fail that, and fetch.py made the same call for the same reason when it dropped
    # mocap from ITS state. swm's own `reset(options={'state'})` contract is concat(qpos, qvel), which
    # is why the split helper still understands qvel; the RENDER state is the part that reaches a pixel.
    names = ([f"qpos[{i}]" for i in range(b.nq)]
             + [f"mocap_pos[{i}]" for i in range(3 * b.nmocap)]
             + [f"mocap_quat[{i}]" for i in range(4 * b.nmocap)]
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
    m = copy.deepcopy(b.model)
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
    c.button_tables = _ctx_button_tables(b, m, res)
    _CTX_CACHE[key] = c
    return c


def _ctx_button_tables(b: "_Bundle", m, res: int) -> dict | None:
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
    base_g = np.array(m.geom_rgba, copy=True)
    base_m = np.array(m.mat_rgba, copy=True)
    zero = tuple([0] * b.nbuttons)
    ref_g0, ref_m0 = b.button_tables[zero]
    out = {}
    for bits, (rg, rm) in b.button_tables.items():
        dg, dm = rg - ref_g0, rm - ref_m0
        # Disjointness, asserted: whatever the bake changed must not also be what the bits change.
        baked_rows = np.nonzero(np.any(base_m != b.pristine["mat_rgba"], axis=1))[0]
        if len(baked_rows):
            assert not np.any(dm[baked_rows]), (
                "the texture bake and the button colour rule touch the same material rows; applying "
                "one as a delta over the other would undo it")
        code = sum(bit << k for k, bit in enumerate(bits))
        out[code] = (torch.as_tensor(base_g + dg, dtype=torch.float32, device="cuda"),
                     torch.as_tensor(base_m + dm, dtype=torch.float32, device="cuda"))
    return out


#: Per-cell facts. ``swm_id`` is the registration under test; ``camera`` is what swm's own ``render``
#: defaults to (both subclasses override the ogbench base to use ``front_pixels``).
SPEC = {
    "Cube": {"swm_id": "swm/OGBCube-v0", "camera": "front_pixels", "size": 224,
             "swm_ref": "swm/OGBCube-v0"},
    "Scene": {"swm_id": "swm/OGBScene-v0", "camera": "front_pixels", "size": 200,
              "swm_ref": "swm/OGBScene-v0"},
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
            raise ValueError(
                f"{type(self).__name__} has no CPU backend (CPU3D-OGB-1): this scene is 193,113 "
                f"visible triangles at a median projected edge of 0.70 px, and the certification "
                f"CPU bar leaves 0.75 ns/triangle against a 24 ns/triangle vertex-setup floor -- "
                f"12.9x short before any fill. See "
                f"examples/results/step7/ogb-manipulation/findings.md. Use device='cuda'.")
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
        mq = s[i:i + 4 * b.nmocap].reshape(b.nmocap, 4); i += 4 * b.nmocap
        bits = s[i:i + b.nbuttons]
        return qpos, mp, mq, bits

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States a dataset would actually hold: reset, then 0-25 random actions.

        Reset states alone do not exercise this env -- every one has the arm at its home pose and
        only the cube moved -- so a fidelity number computed from them would say nothing about the
        14 arm joints that are most of the picture. The same argument fetch.py records.
        """
        import mujoco
        rng = np.random.default_rng(0) if rng is None else rng
        b = self._b
        env, u = _reference_env(self.CELL), None
        u = env.unwrapped
        out = []
        for k in range(int(batch)):
            env.reset(seed=int(rng.integers(0, 1 << 30)))
            env.action_space.seed(int(rng.integers(0, 1 << 30)))
            m, d = u.model, u.data
            for _ in range(int(rng.integers(0, 26))):
                env.step(env.action_space.sample())
            bits = [float(x) for x in list(getattr(u, "_cur_button_states", []))[:b.nbuttons]]
            out.append(np.concatenate([d.qpos.copy(), d.mocap_pos.ravel().copy(),
                                       d.mocap_quat.ravel().copy(), np.asarray(bits, np.float64)]))
        return np.stack(out).astype(np.float64)

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
        """MuJoCo's own raster of the static scene: the floor plane and the skybox. Exact by
        construction, which is the whole reason the composite is worth 0.92-0.95 while MJWarp alone
        is worth 0.35.

        Every mutation is undone from a pristine snapshot rather than by inverse edit, so baking
        twice gives the same model.
        """
        import mujoco
        b = self._b
        m, d, u = b.model, b.data, b.unwrapped
        pr = b.pristine
        keep = np.array(m.geom_group, copy=True)
        m.geom_group[b.movers] = MW.HIDDEN_GROUP
        if states is not None:
            qpos, mp, mq, _ = self._split(np.asarray(states, np.float64))
            d.qpos[:] = qpos
            d.mocap_pos[:] = mp
            d.mocap_quat[:] = mq
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r = _renderer(self.CELL, int(res))
        r.update_scene(d, camera=b.cam, scene_option=u._scene_option)
        img = r.render().copy()
        m.geom_group[:] = keep
        m.geom_rgba[:] = pr["geom_rgba"]
        m.mat_rgba[:] = pr["mat_rgba"]
        mujoco.mj_forward(m, d)
        return torch.as_tensor(img, device=self.device)

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
                mp = chunk[:, nq: nq + 3 * nmoc].reshape(n, nmoc, 3)
                mq = chunk[:, nq + 3 * nmoc: nq + 7 * nmoc].reshape(n, nmoc, 4)
                wp.to_torch(c.wd.mocap_pos)[:n] = mp
                wp.to_torch(c.wd.mocap_quat)[:n] = mq
            if b.nbuttons:
                self._upload_button_colours(c, chunk, n)
            # torch wrote the per-world arrays on its stream; warp replays the graph on its own.
            # The two syncs bracket the launch so a frame can never be rendered from half-written
            # state -- microseconds against the difference between deterministic and nearly so.
            torch.cuda.synchronize()
            wp.capture_launch(c.graph)
            wp.synchronize()
            rgb, seg = c.bgra[:n][..., [2, 1, 0]], c.seg[:n]
            mv = seg >= 0
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
        return _reference_env(cls.CELL, size)

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
            mq = s[i:i + 4 * b.nmocap].reshape(b.nmocap, 4); i += 4 * b.nmocap
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


_RENDERERS: dict = {}


def _renderer(cell: str, res: int):
    """One ``mujoco.Renderer`` per (cell, resolution), kept forever.

    Never closed, for the reason ``envs/ogbench.py`` records at length: releasing an EGL context
    corrupts rendering in every other live renderer in the process.
    """
    key = (cell, int(res))
    hit = _RENDERERS.get(key)
    if hit is not None:
        return hit
    import mujoco
    b = _bundle(cell)
    out = _RENDERERS[key] = mujoco.Renderer(b.model, height=int(res), width=int(res))
    return out


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

_MEASURED: dict = {
    # cell: (mean_abs, within8_frac, ssim)  -- worst of three 64-state draws
}
_UNMEASURED = str(_os.environ.get("SE7_OGB_UNMEASURED", "")) == "1"


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
    mean_abs, within8, ss = _MEASURED[cell]
    spec = SPEC[cell]
    return {
        "mean_abs": float(mean_abs),
        "within8_frac": float(within8),
        "ssim": float(ss),
        # Ceilings/floors with margin, rounded, NOT tracking the measurements.
        "mean_abs_max": float(np.ceil(mean_abs * 1.5 * 10) / 10),
        "within8_min": float(np.floor(within8 * 20) / 20 - 0.05),
        "ssim_min": float(np.floor(ss * 20) / 20 - 0.05),
        "measured_at": (
            f"{spec['size']} px, camera {spec['camera']}, 3 independent draws of 64 states from the "
            f"env's own pool (reset + 0-25 random actions) on an RTX 3090 under MUJOCO_GL=egl, "
            f"against swm's own render(); the declared value is the WORST of the three. The frame is "
            f"a baked MuJoCo plate composited with MJWarp movers by MJWarp's segmentation buffer, "
            f"with gl_rounding and the shared out-of-range-spot correction applied."),
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
    "WHAT THESE NUMBERS ARE NOT: the 0.35 interior figure quoted above is 99.2% a statement about "
    "the FLOOR, which this class bakes through MuJoCo and never asks MJWarp to draw. Whole-frame "
    "exactness of the composite that actually ships is 0.92-0.95.")


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
                f"camera {spec['camera']!r}. State is (qpos, qvel, mocap_pos, mocap_quat"
                + (", button bits)" if cell == "Scene" else ")") + ".\n\n"
                f"THIS IS A VISUAL TIER CELL, NOT A PARITY ONE. Three lighting terms in mujoco_warp "
                f"disagree with MuJoCo and are declared in visual_fidelity and parity_note rather "
                f"than chased: the global directional light, the spot's distance attenuation, and "
                f"the headlight on mover pixels. A fourth, the out-of-range spot cutoff, WAS fixed "
                f"and now lives in the shared render/mjwarp.py path. The translucent goal marker is "
                f"a fifth, separate residual: mujoco_warp has no alpha path.\n\n"
                f"There is NO CPU backend: device='cpu' raises rather than silently using the GPU "
                f"(CPU3D-OGB-1)."),
        )))
    return made


globals().update(_make_classes())
