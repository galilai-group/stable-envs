"""Shared GPU rendering for OGBench Cube and Scene.

MJWarp draws articulated objects over a cached MuJoCo background. Bundles are
keyed by compiled configuration; per-world colors and poses are uploaded on
every render. Mocap orientation is explicit, with identity as its default.

Plate rigs own private models. Their GL contexts remain alive for the process
because releasing one can invalidate other live contexts on supported drivers."""

from __future__ import annotations

import copy

import numpy as np
import torch

from ...env import Env
from ...registry import register
from ...render import _resources as _RES
from ...render import mjwarp as MW
from ...spaces import Box, Dict
from .._plate_texels import quantise as _quantise
from .._plate_texels import substitute as _substitute
from .._plate_texels import texel_key as _texel_key
from .._plate_texels import texel_slots as _texel_slots

#: Groups ``mujoco.Renderer`` draws at the env's own ``MjvOption`` (``geomgroup = [1,1,1,0,0,0]``).
VISIBLE_GROUPS = (0, 1, 2)


def IDENTITY_QUAT(nmocap: int) -> np.ndarray:
    """Return (nmocap, 4) identity quaternions."""
    q = np.zeros((int(nmocap), 4), np.float64)
    q[:, 0] = 1.0
    return q


#: MJWarp worlds per launch. Same constant and the same reason as Fetch: a context is per (res,
#: nworld) and an unbounded batch would build one per batch size.
MJW_CHUNK = 256
# Reset variation leaves sampled explicitly from the caller's RNG.
DEFAULT_VARIATION_PATHS = {
    "Cube": ("cube.start_position", "cube.start_yaw", "cube.goal_position", "cube.goal_yaw"),
    "Scene": ("cube.start_position", "cube.start_yaw", "cube.goal_position", "cube.goal_yaw"),
}
# Caches share compiled configuration, mutable contexts, and static backgrounds.
# Per-world appearance must be uploaded on every draw, including default values.
from .context_cache import ContextCache, allocation_bytes


def _release_context(context):
    # A previous compositor may still have queued reads of the Warp buffers.
    torch.cuda.synchronize(context.rc.rgb_data.device.ordinal)
    context.wp.synchronize_device(context.rc.rgb_data.device)
    MW.release_cached_ctx(context)


_CTX_CACHE = ContextCache(on_evict=_release_context, memory_info=torch.cuda.mem_get_info)
_PLATE_CACHE_MAX = 8
_PLATE_CACHE = _RES.BoundedCache(_PLATE_CACHE_MAX, name="ogbench.plate")
_BUNDLE_CACHE: dict = {}
_REF_CACHE: dict = {}


import contextlib as _contextlib


@_contextlib.contextmanager
def _seeded_action_space(u, seed: int):
    """Temporarily seed the upstream action-space descriptor for reproducible sampling.

    Upstream creates a fresh space on each property access. The class-wide patch
    must not overlap state sampling in another thread."""
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
    """Return a cached upstream environment for rendering or state sampling."""
    key = (cell, int(size) if size else None, str(role))
    hit = _REF_CACHE.get(key)
    if hit is not None:
        return hit
    kw = {}
    if size:
        kw = {"width": int(size), "height": int(size)}
    env = _construct_reference(cell, kw)
    env.reset(seed=0)
    _REF_CACHE[key] = env
    return env


def _construct_reference(cell: str, kw: dict):
    """Build the certified reference env from OGBench constructors."""
    return _ogbench_make(cell, kw)


def _ogbench_make(cell: str, kw: dict):
    """Upstream OGBench construction at the certified size / observation type."""
    size = int(kw.get("height") or kw.get("width") or SPEC[cell]["size"])
    if cell == "Cube":
        from ogbench.manipspace.envs.cube_env import CubeEnv

        return CubeEnv(env_type="single", ob_type="states", height=size, width=size)
    if cell == "Scene":
        from ogbench.manipspace.envs.scene_env import SceneEnv

        return SceneEnv(ob_type="states", height=size, width=size)
    raise KeyError(f"no OGBench constructor for {cell!r}")


class _Bundle:
    """Compiled model and derived render metadata, shared by configuration key."""

    __slots__ = (
        "cam",
        "movers",
        "nq",
        "nv",
        "nmocap",
        "nbuttons",
        "state_names",
        "pristine",
        "size",
        "unwrapped",
        "button_tables",
        "cube_geoms",
        "grid_tex",
        "var_defaults",
        "plate_checked",
    )

    @property
    def model(self):
        return self.unwrapped.model

    @property
    def data(self):
        return self.unwrapped.data


def _movers(m) -> list[int]:
    """Find geoms attached to articulated or mocap bodies, including their descendants."""
    out = []
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i])
        while b > 0:
            if int(m.body_dofnum[b]) > 0 or int(m.body_mocapid[b]) >= 0:
                out.append(i)
                break
            b = int(m.body_parentid[b])
    return out


def render_state_names(nq, nmocap, nbuttons):
    goals = ("goal_x", "goal_y", "goal_z") if nmocap == 1 else tuple(f"mocap_pos_{i}" for i in range(3 * nmocap))
    return tuple(f"qpos_{i}" for i in range(nq)) + goals + tuple(f"button_{i}" for i in range(nbuttons))


def build_bundle(env, *, camera, size, nbuttons):
    """Derive renderer metadata from an already compiled environment, without stepping it."""
    import mujoco

    u = env.unwrapped
    m = u.model
    b = _Bundle()
    b.unwrapped = u
    b.size = int(size)
    b.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if b.cam < 0:
        raise ValueError(f"render model has no camera {camera!r}")
    b.movers = _movers(m)
    b.nq, b.nv, b.nmocap = int(m.nq), int(m.nv), int(m.nmocap)
    b.nbuttons = int(nbuttons)
    b.state_names = render_state_names(b.nq, b.nmocap, b.nbuttons)
    b.pristine = {name: np.array(getattr(m, name), copy=True) for name in ("geom_group", "geom_rgba", "mat_rgba")}
    b.button_tables = _button_tables(u, m, b.nbuttons)
    b.var_defaults = _variation_defaults(u)
    b.cube_geoms = _cube_geoms(u, m)
    b.grid_tex = _grid_texture(m, b.var_defaults["floor.color"])
    b.plate_checked = False
    return b


def get_bundle(cache_key):
    return _BUNDLE_CACHE.get(cache_key)


def register_bundle(cache_key, bundle):
    existing = get_bundle(cache_key)
    if existing is not None and existing is not bundle:
        raise ValueError(f"render bundle key already registered: {cache_key}")
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def _bundle(cell: str) -> _Bundle:
    hit = get_bundle(cell)
    if hit is not None:
        return hit
    env = _reference_env(cell, role="sample")
    return register_bundle(
        cell,
        build_bundle(
            env,
            camera=SPEC[cell]["camera"],
            size=env.unwrapped._render_height,
            nbuttons=int(getattr(env.unwrapped, "_num_buttons", 0) or 0),
        ),
    )


# MuJoCo stores texture colors by truncating 255 * value, not rounding.
mj_quantise = _quantise


def _variation_defaults(u) -> dict:
    """Read default cube RGB (3,) and checker RGB (2, 3) from the upstream variation space."""
    vs = getattr(u, "variation_space", None)
    ncube = int(getattr(u, "_num_cubes", 1) or 1)
    assert ncube == 1, (
        f"this cell has {ncube} cubes, and the (3,) cube.color contract assumes one. A multi-cube "
        f"configuration needs the leaf widened to (num_cubes, 3) on BOTH sides, plus per-cube geom "
        f"groups in _cube_geoms; it is not a shape that can be fixed at the Box alone"
    )
    out = {}
    for path, fallback in (
        ("cube.color", [[0.96, 0.26, 0.33]]),
        ("floor.color", [[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]]),
    ):
        val = None
        if vs is not None:
            node = vs
            try:
                for part in path.split("."):
                    node = node[part]
                val = np.asarray(node.value, np.float64)
            except Exception:
                val = None
        out[path] = np.asarray(fallback, np.float64) if val is None else val
    # upstream (num_cubes, 3) -> our flat (3,); asserted one cube above, so this cannot silently drop
    # a second cube's colour.
    out["cube.color"] = np.asarray(out["cube.color"], np.float64).reshape(-1)[:3]
    out["floor.color"] = np.asarray(out["floor.color"], np.float64).reshape(2, 3)
    return out


def _cube_geoms(u, m) -> np.ndarray:
    """Return geom IDs colored by cube.color, including the translucent target."""
    import mujoco

    ids = []
    for lst in getattr(u, "_cube_geom_ids_list", []) or []:
        ids.extend(int(g) for g in lst)
    for i in range(int(getattr(u, "_num_cubes", 1) or 1)):
        for nm in (f"object_{i}", f"target_object_{i}"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if g >= 0 and g not in ids:
                ids.append(int(g))
    return np.asarray(sorted(set(ids)), np.int64)


def _grid_texture(m, floor_default) -> dict | None:
    """Locate checker texture texels for the two floor colors."""
    return _texel_slots(m, "grid", np.asarray(floor_default, np.float64).reshape(2, 3))


def _button_tables(u, m, nbuttons: int) -> dict | None:
    """Snapshot geom and material colors for each button combination, restoring the initial state."""
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
    assert any(not np.array_equal(first[0], v[0]) or not np.array_equal(first[1], v[1]) for v in out.values()), (
        "button states produce identical colour tables: either SceneEnv's rule changed or this cell "
        "is capturing the wrong arrays, and carrying the bits in the state would be a lie either way"
    )
    return out


class _Ctx:
    """An MJWarp render context plus the device views a render writes through."""

    __slots__ = (
        "wm",
        "wd",
        "rc",
        "bgra",
        "seg",
        "graph",
        "nworld",
        "res",
        "wp",
        "mjw",
        "prepared",
        "baked",
        "button_tables",
    )


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


def _context(cell: str, res: int, nworld: int) -> _Ctx:
    import warp as wp

    key = (torch.cuda.current_device(), cell, int(res), int(nworld))
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
    _CTX_CACHE.prepare(key)
    free_before, _ = torch.cuda.mem_get_info()
    c = _build_context(cell, res, nworld)
    wp.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    try:
        return _CTX_CACHE.put(key, c, max(allocation_bytes(c), free_before - free_after))
    except BaseException:
        MW.release_cached_ctx(c)
        raise


def _build_context(cell: str, res: int, nworld: int) -> _Ctx:
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    b = _bundle(cell)
    # Prepare a private model copy so rendering cannot alter the reference.
    # Copy at all-zero button bits, the base used by the per-world color deltas.
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
    # Bake unsupported UV-less textures before MJWarp can access them.
    baked = MW.bake_unsamplable_textures(m, VISIBLE_GROUPS)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    c = _Ctx()
    c.wp, c.mjw, c.nworld, c.res, c.prepared, c.baked = (wp, mjw, int(nworld), int(res), prepared, baked)
    # Per-world color arrays allow independent button states within one launch.
    c.wm = mjw.put_model(m, batch_sizes={"geom_rgba": int(nworld), "mat_rgba": int(nworld)})
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # Capture inside gl_rounding so the graph retains the patched quantizer.
    with MW.gl_rounding():
        c.rc = MW.create_render_context(
            m,
            nworld=int(nworld),
            cam_res=(int(res), int(res)),
            render_rgb=True,
            render_seg=True,
            cam_active=[i == b.cam for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS),
            use_textures=True,
            use_shadows=True,
            render_skybox=True,
        )
        _frame(c)
        wp.synchronize()
        # Keep views of Warp buffers; advanced channel indexing here would freeze a copied frame.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
        c.seg = wp.to_torch(c.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)[..., 0]
        with wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    c.button_tables = _ctx_button_tables(b, m, res, device="cuda")
    return c


def _ctx_button_tables(b: _Bundle, m, res: int, device: str = "cuda") -> dict | None:
    """Build device button-color tables as deltas from the all-zero reference state.

    Apply deltas to the prepared context model so button changes preserve texture
    bakes; overlapping material edits are rejected."""
    if not b.nbuttons or b.button_tables is None:
        return None
    # `m` was deep-copied from the reference while it was forced to all-zero bits (see `_context`),
    # so the all-zero table is the right delta base. Asserted below rather than assumed.
    base_g = np.array(m.geom_rgba, copy=True)
    base_m = np.array(m.mat_rgba, copy=True)
    zero = tuple([0] * b.nbuttons)
    ref_g0, ref_m0 = b.button_tables[zero]
    touched = sorted(
        {int(i) for bits, (rg, rm) in b.button_tables.items() for i in np.nonzero(np.any(rg != ref_g0, axis=1))[0]}
    )
    assert np.array_equal(base_g[touched], ref_g0[touched]), (
        "the context model was copied at button bits other than all-zero, so every delta below "
        "would be applied to the wrong base; see _context's forced-zero block"
    )
    # Compare prepared materials against the same all-zero button baseline to isolate texture bakes.
    baked_rows = np.nonzero(np.any(base_m != ref_m0, axis=1))[0]
    out = {}
    for bits, (rg, rm) in b.button_tables.items():
        dg, dm = rg - ref_g0, rm - ref_m0
        if len(baked_rows):
            assert not np.any(dm[baked_rows]), (
                f"the texture bake and the button colour rule both write material rows "
                f"{sorted(set(baked_rows) & set(np.nonzero(np.any(dm, axis=1))[0]))}; applying one "
                f"as a delta over the other would undo it"
            )
        code = sum(bit << k for k, bit in enumerate(bits))
        out[code] = (
            torch.as_tensor(base_g + dg, dtype=torch.float32, device=device),
            torch.as_tensor(base_m + dm, dtype=torch.float32, device=device),
        )
    return out


#: Per-cell facts. Native OGBench constructors are the certified reference.
SPEC = {
    # Registered camera, image size, and packed render-row width.
    "Cube": {
        "camera": "front_pixels",
        "size": 224,
        "visible_tri": 193113,
        "state_dim": 24,
    },
    "Scene": {
        "camera": "front_pixels",
        "size": 200,
        "visible_tri": 196633,
        "state_dim": 30,
    },
}


class OGBManipBase(Env):
    """Base for the two manipulation cells. ``CELL`` is set by the generated subclasses."""

    family = "ogbench"
    requires_cuda = True

    CELL: str = ""
    default_supersample = 1
    fidelity_tier = "visual"

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str | None = None,
        backend: str = "auto",
        seed: int = 0,
        *,
        bundle: _Bundle | None = None,
        cache_key: str | None = None,
    ):
        """Construct the GPU renderer, optionally using a registered compiled bundle."""
        device = Env.resolve_device(device)
        if self.requires_cuda and not str(device).startswith("cuda"):
            raise ValueError("The MJWarp manipulation renderer requires device='cuda'.")
        if (bundle is None) != (cache_key is None):
            raise ValueError("bundle and cache_key must be supplied together")
        if bundle is not None:
            self.CELL = cache_key
            register_bundle(cache_key, bundle)
        b = _bundle(self.CELL)
        self.state_dim = len(b.state_names)
        self.state_names = b.state_names
        # The packed row contains qpos, mocap position, and button bits in that order.
        self.state_layout = ((("qpos",), b.nq, 0), (("mocap", "position"), 3 * b.nmocap, b.nq)) + (
            ((("buttons",), b.nbuttons, b.nq + 3 * b.nmocap),) if b.nbuttons else ()
        )
        super().__init__(
            num_envs=num_envs,
            size=size or b.size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        self._b = b
        if self.requires_cuda:
            self._check_plate_is_constant()

    def _close(self) -> None:
        """Drop instance-owned index and style caches; shared contexts remain alive."""
        self.__dict__.pop("_idx_cache", None)
        self.__dict__.pop("_style_host_cache", None)

    # ---- the assumption the whole design rests on, checked rather than assumed -------------

    def _check_plate_is_constant(self) -> None:
        """Check once per compiled bundle that the baked background is pose-independent."""
        if self._b.plate_checked:
            return
        import mujoco

        b = self._b
        state = np.concatenate([b.data.qpos, b.data.mocap_pos.ravel(), np.zeros(b.nbuttons)])
        other = state.copy()
        for joint in range(b.model.njnt):
            if b.model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                other[b.model.jnt_qposadr[joint]] += 0.1
        other[b.nq : b.nq + 3 * b.nmocap] += 0.1
        a = self._bake_plate(128, states=state)
        c = self._bake_plate(128, states=other)
        if not torch.equal(a, c):
            raise ValueError("baked static background depends on articulated or mocap pose")
        b.plate_checked = True

    # ---- state ----------------------------------------------------------------------------

    def _split(self, s: np.ndarray):
        """Split a packed render row into qpos, mocap position, default orientation, and buttons."""
        b = self._b
        i = 0
        qpos = s[i : i + b.nq]
        i += b.nq
        mp = s[i : i + 3 * b.nmocap].reshape(b.nmocap, 3)
        i += 3 * b.nmocap
        bits = s[i : i + b.nbuttons]
        return qpos, mp, IDENTITY_QUAT(b.nmocap), bits

    def _variation_values(self, rng: np.random.Generator) -> dict:
        """Sample reset variation leaves explicitly from rng, avoiding upstream entropy-based seeding."""
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
        """Return float32 render rows from seeded resets followed by 0–25 random actions."""
        rng = np.random.default_rng(0) if rng is None else rng
        b = self._b
        env = _reference_env(self.CELL, role="sample")  # reset many times; never rendered
        u = env.unwrapped
        out = []
        # Upstream reset also consumes NumPy's global RNG; restore its state after sampling.
        gstate = np.random.get_state()
        try:
            for k in range(int(batch)):
                sd = int(rng.integers(0, 1 << 30))
                np.random.seed(int(rng.integers(0, 1 << 31)))
                # Seed both the environment generator and its action-space descriptor before reset.
                u.np_random = np.random.default_rng(sd)
                with _seeded_action_space(u, sd):
                    env.reset(seed=sd, options={"variation_values": self._variation_values(rng)})
                    _m, d = u.model, u.data
                    for _ in range(int(rng.integers(0, 26))):
                        env.step(u.action_space.sample())
                bits = [float(x) for x in list(getattr(u, "_cur_button_states", []))[: b.nbuttons]]
                out.append(np.concatenate([d.qpos.copy(), d.mocap_pos.ravel().copy(), np.asarray(bits, np.float64)]))
        finally:
            np.random.set_state(gstate)
        # Match float32 render inputs before rendering either candidate or reference.
        return np.stack(out).astype(np.float32)

    # ---- variation ------------------------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Expose per-world cube RGB and per-background checker RGB."""
        d = _bundle(self.CELL).var_defaults
        return Dict(
            {
                "cube": Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=d["cube.color"])}),
                "floor": Dict({"color": Box(low=0.0, high=1.0, shape=(2, 3), init_value=d["floor.color"])}),
            }
        )

    # Floor color changes the static background and therefore the plate grouping.
    PLATE_PATHS = ("floor.color",)

    def _style_host(self, sty) -> dict:
        """Copy a complete style row to host once; cache by tensor identity and mutation version."""
        v = sty.values
        # Cache identity and mutation version together; retain the tensor to prevent ID reuse.
        hit = self.__dict__.get("_style_host_cache")
        if hit is not None and hit[0] is v and hit[1] == v._version:
            return hit[2]
        host = np.asarray(v.detach().cpu(), np.float64)
        out = {p: host[:, o : o + w] for p, (o, w) in sty.layout.items()}
        self.__dict__["_style_host_cache"] = (v, v._version, out)
        return out

    # ---- the plate ------------------------------------------------------------------------

    def _plate(self, res: int, floor=None) -> torch.Tensor:
        """Return the background cached by configuration, resolution, device, and quantized floor RGB."""
        fq = self._floor_key(floor)
        key = (self.CELL, int(res), str(self.device), fq)
        hit = _PLATE_CACHE.get(key)
        if hit is not None:
            return hit
        return _PLATE_CACHE.put(key, self._bake_plate(res, floor=floor))

    def _floor_key(self, floor):
        """Return the checker texture byte key, or None for the default appearance."""
        return _texel_key(self._b.grid_tex, floor)

    def _bake_plate(self, res: int, states=None, floor=None) -> torch.Tensor:
        """Render the static background through a private model, checking its shape and appearance."""
        import mujoco

        b = self._b
        rig = _plate_rig(self.CELL, int(res))
        live = b.model
        shape = (int(live.ngeom), int(live.nq), int(live.nmocap))
        assert shape == rig.shape, (
            f"{self.CELL}: the env recompiled to a different model shape {shape} than the plate rig "
            f"was built for {rig.shape}; the cached plate would be a picture of the old scene"
        )
        now = _appearance_digest(live, b.unwrapped._scene_option.sitegroup)
        assert now == rig.appearance, (
            f"{self.CELL}: an appearance array changed since the plate rig was built "
            f"({rig.appearance} -> {now}). The cached plate is a picture of the scene as it was, so "
            f"it is now wrong; phase 0 measured these arrays identical across resets and this "
            f"assertion is what keeps that measurement honest"
        )
        self._apply_floor(rig, floor)
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

    def _apply_floor(self, rig, floor) -> None:
        """Write checker texels and upload them in the rig's current GL context.

        The upload is unconditional: shared mutable textures cannot infer their current
        contents from another caller's cached floor value."""
        g = self._b.grid_tex
        if g is None:
            return
        want = self._b.var_defaults["floor.color"] if floor is None else np.asarray(floor, np.float64)
        want = want.reshape(2, 3)
        _substitute(rig.model, rig.renderer, g, want)
        rig.floor = want.copy()

    # ---- render ---------------------------------------------------------------------------

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
        *,
        mocap_quat=None,
    ) -> torch.Tensor:
        """Render (..., state_dim) rows to (..., H, W, 3) uint8, with optional (B, nmocap, 4) orientations."""
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        if mocap_quat is not None:
            mocap_quat = torch.as_tensor(mocap_quat, device=self.device, dtype=torch.float32)
            if mocap_quat.shape != (B, self._b.nmocap, 4):
                raise ValueError("mocap_quat must have shape (B, nmocap, 4)")
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        sty = self.resolve_style(style, variation, batch=B)
        # Use one floor grouping at both resolutions so each row receives the same background.
        host = self._style_host(sty)
        reps, gidx = self._floor_plan(host, B)
        plates = self._floor_plates(reps, res)
        plates1 = plates if ss == 1 else self._floor_plates(reps, size)
        plate = plates[0] if gidx is None else None
        plate1 = (plate if ss == 1 else plates1[0]) if gidx is None else None
        cube = self._cube_colours(host, B)
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
                mp = chunk[:, nq : nq + 3 * nmoc].reshape(n, nmoc, 3)
                wp.to_torch(c.wd.mocap_pos)[:n] = mp
                wp.to_torch(c.wd.mocap_quat)[:n] = (
                    self._identity_quat(nmoc, n) if mocap_quat is None else mocap_quat[lo:hi]
                )
            if b.nbuttons:
                self._upload_button_colours(c, chunk, n)
            # Upload buttons first: their whole-row color write would otherwise overwrite cube RGB.
            if cube is not None:
                self._upload_cube_colours(c, cube[lo:hi], n)
            # Synchronize Torch uploads and Warp graph replay before reading shared buffers.
            torch.cuda.synchronize()
            wp.capture_launch(c.graph)
            wp.synchronize()
            rgb, seg = c.bgra[:n][..., [2, 1, 0]], c.seg[:n]
            # Use the mover mask: a nonnegative segmentation ID may instead identify the static floor.
            mv = lut.index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
            if gidx is None:
                pl, pl1 = plate, plate1
            else:
                g = gidx[lo:hi]
                pl = plates.index_select(0, g)
                pl1 = pl if ss == 1 else plates1.index_select(0, g)
            comp = torch.where(mv.unsqueeze(-1), rgb, pl)
            if ss == 1:
                buf[lo:hi] = comp
            else:
                small = (
                    comp.to(torch.float32)
                    .reshape(n, size, ss, size, ss, 3)
                    .mean((2, 4))
                    .round_()
                    .clamp_(0, 255)
                    .to(torch.uint8)
                )
                any_mv = mv.reshape(n, size, ss, size, ss).any(4).any(2)
                buf[lo:hi] = torch.where(any_mv.unsqueeze(-1), small, pl1)
        return buf.view(*lead, size, size, 3) if lead else buf

    def _identity_quat(self, nmoc: int, n: int) -> torch.Tensor:
        """Return cached (n, nmoc, 4) identity quaternions on the renderer device."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        key = ("idq", int(nmoc), int(n))
        hit = cache.get(key)
        if hit is None:
            q = torch.zeros((int(n), int(nmoc), 4), dtype=torch.float32, device=self.device)
            q[..., 0] = 1.0
            hit = cache[key] = q
        return hit

    def _mover_lut(self) -> torch.Tensor:
        """Map segmentation IDs plus one to a boolean mover mask; slot zero represents a miss."""
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("mover_lut")
        if hit is None:
            n = int(self._b.model.ngeom)
            lut = torch.zeros(n + 1, dtype=torch.bool, device=self.device)
            lut[torch.as_tensor(np.asarray(self._b.movers, np.int64) + 1, device=self.device)] = True
            hit = cache["mover_lut"] = lut
        return hit

    def _floor_plan(self, host, B: int):
        """Group environments by quantized floor color; return original float rows and group indices.

        Use bytes for grouping and original floats for baking: decoding and requantizing
        bytes can lose a level. A uniform batch returns None for its group indices."""
        f = host["floor.color"]  # (B, 6)
        q = mj_quantise(f.reshape(B, 2, 3)).reshape(B, 6)
        uniq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        reps = [f[i].reshape(2, 3) for i in first]
        if len(uniq) == 1:
            return reps, None
        inv = np.asarray(inv, np.int64).reshape(-1)  # numpy >= 2.0 shapes this per input, not flat
        return reps, torch.as_tensor(inv, device=self.device)

    def _floor_plates(self, reps, res: int) -> torch.Tensor:
        """Return (G, res, res, 3) cached backgrounds for the floor groups."""
        if len(reps) == 1:
            return self._plate(res, floor=reps[0]).unsqueeze(0)  # a VIEW: no copy on the hot path
        return torch.stack([self._plate(res, floor=r) for r in reps])

    def _cube_colours(self, host, B: int):
        """Return (B, len(cube_geoms), 4) colors, preserving each geom's original alpha.

        Always upload defaults too, since render contexts retain per-world colors
        between calls. Return None only if the model has no cube geoms."""
        b = self._b
        gids = b.cube_geoms
        if gids is None or not len(gids):
            return None
        rgb = host["cube.color"]  # (B, 3)
        base = np.asarray(b.pristine["geom_rgba"], np.float64)[gids]  # (G, 4), own alphas
        out = np.broadcast_to(base, (B, len(gids), 4)).copy()
        out[:, :, :3] = rgb[:, None, :]
        return torch.as_tensor(out, dtype=torch.float32, device=self.device)

    def _cube_geom_index(self) -> torch.Tensor:
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get("cube_geoms")
        if hit is None:
            hit = cache["cube_geoms"] = torch.as_tensor(np.asarray(self._b.cube_geoms, np.int64), device=self.device)
        return hit

    def _upload_cube_colours(self, c, rgba, n: int) -> None:
        """Write each world's cube colors into the shared render context."""
        geom = c.wp.to_torch(c.wm.geom_rgba)
        idx = self._cube_geom_index()
        geom[:n, idx] = rgba[:n]

    def _upload_button_colours(self, c, chunk, n: int) -> None:
        """Upload each world's button colors from its state bits."""
        b = self._b
        bits = chunk[:, -b.nbuttons :].round().clamp(0, 1).to(torch.int64).cpu().numpy()
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
        """Render packed rows through the upstream CPU renderer."""
        import mujoco

        u = env.unwrapped
        b = _bundle(cls.CELL)
        out = []
        for s in np.asarray(states, np.float64).reshape(-1, len(b.state_names)):
            i = 0
            qpos = s[i : i + b.nq]
            i += b.nq
            qvel = np.zeros(b.nv)
            mp = s[i : i + 3 * b.nmocap].reshape(b.nmocap, 3)
            i += 3 * b.nmocap
            mq = IDENTITY_QUAT(b.nmocap)
            bits = s[i : i + b.nbuttons]
            if b.nbuttons:
                u.set_state(qpos, qvel, **{f"button_state_{k}": int(round(float(bits[k]))) for k in range(b.nbuttons)})
            else:
                u.set_state(qpos, qvel)
            u.data.mocap_pos[:] = mp
            u.data.mocap_quat[:] = mq
            mujoco.mj_forward(u.model, u.data)
            out.append(u.render().copy())
        return np.stack(out)


_PLATE_RIGS: dict = {}

# Retain GL renderers until process exit; releasing one may corrupt other live contexts.
_KEEPALIVE: list = []

# Static appearance arrays; per-world geom/material colors are handled separately.
_APPEARANCE_PREFIXES = ("geom_", "mat_", "tex", "light_", "cam_", "mesh_", "body_", "site_", "hfield_", "skin_")
_APPEARANCE_SKIP = ("geom_rgba", "mat_rgba", "geom_group")


def _appearance_digest(m, sitegroup=None) -> str:
    """Hash static appearance arrays and visible sites for background-cache validation."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    n = 0
    # Ignore hidden sites, whose poses change without affecting the rendered background.
    nsite = int(getattr(m, "nsite", 0) or 0)
    drawn = None
    if nsite and sitegroup is not None:
        flags = np.asarray(sitegroup).ravel()
        g = np.clip(np.asarray(m.site_group, int), 0, len(flags) - 1)
        drawn = flags[g] != 0
    for name in sorted(dir(m)):
        if name.startswith("_") or name in _APPEARANCE_SKIP:
            continue
        if not name.startswith(_APPEARANCE_PREFIXES):
            continue
        try:
            v = getattr(m, name)
        except Exception:  # pragma: no cover - MjModel exposes only arrays here
            continue
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            if drawn is not None and name.startswith("site_") and a.shape[:1] == (nsite,):
                a = np.ascontiguousarray(a[drawn])
            h.update(name.encode())
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
            n += 1
    assert n >= 20, (
        f"appearance digest covered only {n} arrays -- an empty hash would certify staleness it never looked at"
    )
    return h.hexdigest()


class _PlateRig:
    """Private model, data, and retained GL renderer for one cached background resolution."""

    __slots__ = ("model", "data", "renderer", "appearance", "shape", "floor")


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
    rig.floor = None  # which floor.color the GL context currently holds
    rig.appearance = _appearance_digest(live, b.unwrapped._scene_option.sitegroup)
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
    # Enlarge the private framebuffer before constructing a renderer at the requested resolution.
    rig.model.vis.global_.offwidth = max(int(rig.model.vis.global_.offwidth), int(res))
    rig.model.vis.global_.offheight = max(int(rig.model.vis.global_.offheight), int(res))
    rig.renderer = mujoco.Renderer(rig.model, height=int(res), width=int(res))
    _KEEPALIVE.append(rig.renderer)  # the rule, enforced directly: never released
    _PLATE_RIGS[key] = rig  # never evicted; the renderer is never closed
    if __debug__ and _os.environ.get("SE7_OGB_RIGDEBUG") == "1":  # diagnostics, off unless asked
        import sys

        mods = [n for n, m in sys.modules.items() if getattr(m, "__name__", None) == __name__ and m is not None]
        print(
            f"[rigdebug] built {key}: rigs={len(_PLATE_RIGS)} keepalive={len(_KEEPALIVE)} "
            f"dict_id={id(_PLATE_RIGS)} module_copies={len(mods)} {mods}",
            flush=True,
        )
    return rig


# Recorded fidelity measurements and declared acceptance margins.
import os as _os

_MEASURED: dict = {
    "Cube": {
        "all_exact_frac": 0.870012,
        "draw_spread": {
            "mean_abs": [3.265277, 3.309378],
            "ssim": [0.919219, 0.926569],
            "within8_frac": [0.91581, 0.917723],
        },
        "gate_draw_mean_abs": 3.295483,
        "interior_exact_frac": 0.9578,
        "interior_frac": 0.412552,
        "mean_abs": 3.309378,
        "ssim": 0.919219,
        "within8_frac": 0.91581,
    },
    "Scene": {
        "all_exact_frac": 0.773417,
        "draw_spread": {
            "mean_abs": [6.493273, 6.691589],
            "ssim": [0.892792, 0.898809],
            "within8_frac": [0.855483, 0.860345],
        },
        "gate_draw_mean_abs": 6.602667,
        "interior_exact_frac": 0.912032,
        "interior_frac": 0.278728,
        "mean_abs": 6.691589,
        "ssim": 0.892792,
        "within8_frac": 0.855483,
    },
}
#: OUR side: the batch raytracer that drew the movers.
_UNMEASURED = str(_os.environ.get("SE7_OGB_UNMEASURED", "")) == "1"

# Pin the upstream renderer and scene package independently of our rendering backend.
_REFERENCE_VERSIONS = {"mujoco": "3.12.0", "ogbench": "1.2.1"}

_MEASURED_WITH = {"mujoco_warp": "3.12.0"}


def _fidelity(cell: str) -> dict:
    if cell not in _MEASURED:
        if not _UNMEASURED:
            raise RuntimeError(
                f"ogb_manip: {cell} has no MEASURED visual_fidelity. This module refuses to register "
                f"a visual cell with invented numbers. Re-measure Gate-F fidelity, paste its "
                f"worst-of-eight into _MEASURED, and re-import. Set SE7_OGB_UNMEASURED=1 only "
                f"inside that measuring job."
            )
        return {
            "mean_abs": -1.0,
            "within8_frac": -1.0,
            "ssim": -1.0,
            "mean_abs_max": -1.0,
            "within8_min": -1.0,
            "ssim_min": -1.0,
            "measured_at": "UNMEASURED PLACEHOLDER - SE7_OGB_UNMEASURED=1 is set, so these are "
            "NOT measurements and this build must not ship or be certified.",
        }
    d = _MEASURED[cell]
    spec = SPEC[cell]
    lo_w, hi_w = d["draw_spread"]["within8_frac"]
    lo_s, hi_s = d["draw_spread"]["ssim"]
    return {
        "mean_abs": float(d["mean_abs"]),
        "within8_frac": float(d["within8_frac"]),
        "ssim": float(d["ssim"]),
        # Interior statistics are reported separately from the whole-frame acceptance thresholds.
        "interior_exact_frac": float(d["interior_exact_frac"]),
        "interior_frac": float(d["interior_frac"]),
        "all_exact_frac": float(d["all_exact_frac"]),
        "draw_spread": d["draw_spread"],
        # Margins are 1.3 times mean error and at least 0.04 below the worst fractional score.
        "mean_abs_max": round(float(d["mean_abs"]) * 1.3, 1),
        "within8_min": float(np.floor((lo_w - 0.04) * 20) / 20),
        "ssim_min": float(np.floor((lo_s - 0.04) * 20) / 20),
        "measured_at": (
            f"mujoco 3.12.0 / ogbench 1.2.1 reference; {spec['size']} px, "
            f"camera {spec['camera']}; worst of eight draws of 64 states, each from "
            "a reset followed by 0–25 random actions. Measured by check_visual_fidelity; "
            "draw_spread records the observed range."
        ),
        # The REFERENCE side, which the certifier compares: mujoco is what OGBench's render() draws with.
        "reference_versions": _REFERENCE_VERSIONS,
        # OUR side, recorded and never compared: a bump of our own renderer that moves these numbers
        # is a regression to be seen, not a reason to block the check.
        "measured_with": _MEASURED_WITH,
    }


_RESIDUAL_NOTE = (
    "The mjwarp path uses reference-rendered static backgrounds with MJWarp movers. "
    "Mover lighting and target transparency can differ from MuJoCo; visual_fidelity "
    "records the measured error and acceptance margins."
)
_SCENE_BAKE_NOTE = (
    " Scene's UV-less button textures are replaced by mean colors because the pinned "
    "MJWarp version cannot sample them safely."
)


def _make_classes():
    made = {}
    for cell, spec in SPEC.items():
        cname = f"OGB{cell}"
        made[cname] = register(
            type(
                cname,
                (OGBManipBase,),
                dict(
                    CELL=cell,
                    domain=cell.lower(),
                    name=f"ogb-{cell.lower()}",
                    reference_id=None,
                    default_size=spec["size"],
                    visual_fidelity=_fidelity(cell),
                    visual_note=_RESIDUAL_NOTE + (_SCENE_BAKE_NOTE if cell == "Scene" else ""),
                    parity_interior_min=None,
                    __doc__=(
                        f"Visual tier GPU renderer for OGBench {cell}, using camera {spec['camera']!r}. "
                        f"Packed render rows have {spec['state_dim']} columns; optional mocap "
                        "orientations are supplied separately. Lighting and target transparency can "
                        "differ from the reference renderer."
                    ),
                ),
            )
        )
    return made


#: ``{name: class}`` for the two registered cells; see ``maze.CELL_CLASSES``.
CELL_CLASSES = _make_classes()
globals().update(CELL_CLASSES)
