from __future__ import annotations

import os

import numpy as np
import torch

from .._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

import importlib.util as _ilu

import mujoco as _mj

# Opt in while coincident geometry can produce slot-dependent pixels.
if os.environ.get("STABLE_ENVS_HUMANOIDBENCH", "") != "1":  # pragma: no cover
    raise ImportError(
        "the HumanoidBench family is measured and complete on fidelity (17 cells, VISUAL tier, mean "
        "2.6-8.4/255, SSIM 0.845-0.931, 437x the official renderer at batch 64) but is NOT registered "
        "by default: mujoco_warp's batched render is not bit-reproducible inside a long conformance "
        "session, so 2-3 of ~660 tests fail every run on test_determinism and "
        "test_frame_i_is_independent_of_batch_size, by 1-2/255, on a different cell each time. Set "
        "STABLE_ENVS_HUMANOIDBENCH=1 to register it anyway."
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
        f" pulls the renderer half. The envs were measured in {_VENV}."
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
# Keep a fixed world count to limit context variants and small-batch overhead.
MJW_CHUNK = 64

_MODEL_CACHE: dict = {}
_CTX_CACHE = _RES.BoundedCache(32, name="humanoidbench.ctx", on_evict=MW.release_cached_ctx)
# Retain contexts across task switches; eviction releases their render resources.
_CTX_CACHE_MAX = 32
_POOL_CACHE: dict = {}


def _hb_compat() -> list:
    """Patch source wrappers for newer MuJoCo fields and Gymnasium rendering.

    Missing named-indexer fields return None; camera selection stays on the
    renderer constructor. Patches are process-global and idempotent. Return
    the names of unavailable fields.
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
    from humanoid_bench.env import TASKS

    return TASKS[task].camera_name


class _Bundle:
    __slots__ = ("model", "data", "info", "pristine", "state_names", "nq", "robot", "task")


def _build_bundle(robot: str, task: str) -> _Bundle:
    """Compile a render model with resolved colors and a flat floor approximation.

    Collision margins are cleared only in this rendering model.
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

    # Resolve source geom/material precedence into one material per geom.
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

    # Approximate the floor texture with its dominant texel, exposed as a per-world color.
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

    # Exclude invisible geometry before building the context; MJWarp does not apply alpha.
    prepared = MW.prepare_model(m, VISIBLE_GROUPS)
    # This render-only model needs zero margins for MJWarp MULTICCD validation.
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
    # Keep truncating output for the flat-floor approximation; GL rounding increases its bias.
    # Capture inside specialised_lights so replay uses the specialized kernel.
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
        # Keep a view: advanced channel indexing here would freeze a copy of the first frame.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
        with wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    _CTX_CACHE.put(key, c)
    return c


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    if c.needs_com:
        # TRACKCOM reads subtree_com, which kinematics alone does not update.
        c.mjw.com_pos(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


class HumanoidBenchBase(Env):
    """Render HumanoidBench qpos through MJWarp using task-specific cameras.

    VISUAL TIER: the floor texture is replaced by its dominant color. This
    family is opt-in because coincident geometry can yield slot-dependent
    pixels. State contains qpos only; it is not a simulator continuation snapshot.
    """

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
        # The render state is qpos only; this family does not store simulator continuation state.
        self.state_layout = ((("qpos",), self.state_dim, 0),)
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        self._b = b

    def make_variation_space(self) -> Dict:
        """Expose float-valued material colors, light intensity, and camera offsets.

        Keep material colors as floats to preserve source defaults. The skybox is
        a context texture and cannot vary per lane.
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
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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

        campos, cammat = self._camera(sty, 0, B)
        # Pin the context world count: changing it can change rounding and BVH tie outcomes.
        c = _context(self.ROBOT, self.TASK, res, MJW_CHUNK)
        # Upload, launch, and copy on one stream before reusing shared context buffers.
        # Concurrent callers on other streams must not drive this context.
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
        """Write all context worlds, padding the tail with the last requested row.

        Uninitialized worlds affect the shared BVH and can change near-tie traversal
        even in requested worlds. Padding removes stale input, but does not resolve
        slot-dependent ties between exactly coincident surfaces.
        """
        info, n = self._b.info, hi - lo
        nw = c.nworld
        wp = c.wp
        pad = lambda t: t if n >= nw else torch.cat([t, t[-1:].expand(nw - n, *t.shape[1:])], 0)

        wp.to_torch(c.wd.qpos)[:] = pad(s)
        mat = wp.to_torch(c.wm.mat_rgba)
        m0 = info["mat0"]
        mat[:, m0 + info["floor"], :3] = pad(sty["floor.color"][lo:hi])
        mat[:, self._idx("robot.mats", m0 + np.asarray(info["robot"], np.int64)), :3] = pad(
            sty["robot.color"][lo:hi]
        ).unsqueeze(1)
        wp.to_torch(c.wm.light_diffuse)[:] = self._light0().unsqueeze(0) * pad(
            sty["light.diffuse"][lo:hi].reshape(n, 1, 1)
        )
        wp.to_torch(c.wm.cam_poscom0)[:, info["cam"]] = pad(campos)
        wp.to_torch(c.wm.cam_mat0)[:, info["cam"]] = pad(cammat)

    def _idx(self, key: str, values) -> torch.Tensor:
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
        """Orbit both the TRACKCOM offset and orientation about the tracked center of mass."""
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
        # Warp mat33 views require (n, 3, 3), not flattened (n, 9) assignments.
        mat = torch.as_tensor(np.stack(mats)[inv], device=self.device, dtype=torch.float32)
        return pos, mat

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample qpos from cached source resets and up to 60 random actions."""
        rng = rng or self.rng
        pool = _state_pool(self.ROBOT, self.TASK, self.state_dim)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    @classmethod
    def reference_env(cls, size: int = 256):
        """Construct a source environment and open its renderer. The caller owns its lifetime."""
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
        """Set source qpos, run forward kinematics, and render.

        Reject qpos changes above 1e-5 after forward; smaller quaternion
        normalization differences are tolerated.
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
    """Cache source qpos samples with local, global NumPy, and action RNGs seeded.

    Task placement uses global NumPy randomness, so reset(seed=...) alone
    does not reproduce the pool.
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

# Per-task visual thresholds and their source comparison measurements.
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

# Register only measured tasks whose declared qpos dimensions affect the image.
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
