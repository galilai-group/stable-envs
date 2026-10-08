"""Render cached RoboCasa kitchens without importing RoboCasa in the render process.

Build the kitchen XML and trajectory with RoboCasa's own dependencies using
``examples/robocasa_visual/dump_trajectory.py``. Point
``STABLE_ENVS_ROBOCASA_CACHE`` at that cache. Rendering uses a separate MuJoCo/
MJWarp installation; only cached, declared kitchen/camera pairs register.

Plain and learned-corrector classes are separate visual-tier registrations.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from .._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

# Require the raytracer and a separately compiled kitchen before registration.
import importlib.util as _ilu

import mujoco as _mj

#: Where a compiled kitchen is cached. One directory per ``(layout, style)``.
CACHE_ROOT = os.path.expanduser(os.environ.get("STABLE_ENVS_ROBOCASA_CACHE", "~/.cache/stable-envs/robocasa"))

_BUILD_CMD = (
    "conda run -n robocasa python examples/robocasa_visual/dump_trajectory.py "
    "--kitchen {kitchen} --steps 200 --res 256 --out " + CACHE_ROOT
)

_missing = [_m for _m in ("warp", "mujoco_warp") if _ilu.find_spec(_m) is None]
_mjver = tuple(int(x) for x in str(_mj.__version__).split(".")[:2])
if _missing or _mjver < (3, 11):  # pragma: no cover
    raise ImportError(
        "the RoboCasa family renders its kitchens with mujoco_warp's batch raytracer, which this "
        "interpreter cannot run: "
        + (f"missing {', '.join(_missing)}; " if _missing else "")
        + f"mujoco is {_mj.__version__} and mujoco_warp requires >= 3.11. Note that robocasa itself "
        f"pins mujoco == 3.3.1, so the interpreter that BUILDS a kitchen is never the one that "
        f"renders it; see the module docstring."
    )

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import mjwarp as MW
from ..render import rendernet as RN
from ..spaces import Box, Dict

# Each registration selects one cached kitchen and camera.
CELLS = (
    ("RoboCasaCounterToSink", "L52S03", "robot0_agentview_left"),
    ("RoboCasaCounterToSinkRight", "L52S03", "robot0_agentview_right"),
    ("RoboCasaCounterToSinkWrist", "L52S03", "robot0_eye_in_hand"),
    ("RoboCasaCounterToSinkL45S33", "L45S33", "robot0_agentview_left"),
)

# Match robosuite visual geometry on both sides; group 0 contains collision geometry.
VISIBLE_GROUPS = (1,)
# Limit per-context image buffers and amortize per-chunk overhead.
MJW_CHUNK = 256
_CTX_CACHE = _RES.BoundedCache(4, name="robocasa.ctx", on_evict=MW.release_cached_ctx)
_CTX_CACHE_MAX = 4
_SCENE_CACHE: dict = {}
_REF_CACHE: dict = {}
_POOL_CACHE: dict = {}
#: ``(class, size) -> a CPU env`` borrowed by ``reference_render(style=)`` for its camera arithmetic.
_STYLE_PROBE: dict = {}


def available_kitchens() -> list[str]:
    """List cache directories containing scene.xml and qpos.npy."""
    if not os.path.isdir(CACHE_ROOT):
        return []
    return sorted(
        k
        for k in os.listdir(CACHE_ROOT)
        if os.path.exists(os.path.join(CACHE_ROOT, k, "scene.xml"))
        and os.path.exists(os.path.join(CACHE_ROOT, k, "qpos.npy"))
    )


if not available_kitchens():  # pragma: no cover
    raise ImportError(
        f"no compiled RoboCasa kitchen is cached under {CACHE_ROOT}. robocasa pins mujoco == 3.3.1 "
        f"and mujoco_warp needs >= 3.11, so the kitchen has to be built by a different interpreter "
        f"and cached:\n\n    " + _BUILD_CMD.format(kitchen="L52S03") + "\n\n"
        "It needs robocasa's 24 GB asset library. Set STABLE_ENVS_ROBOCASA_CACHE to point somewhere "
        "else."
    )


def _joint_qpos_names(m) -> list[str]:
    import mujoco

    suffix = {
        int(mujoco.mjtJoint.mjJNT_FREE): ("x", "y", "z", "qw", "qx", "qy", "qz"),
        int(mujoco.mjtJoint.mjJNT_BALL): ("qw", "qx", "qy", "qz"),
    }
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
    __slots__ = ("model", "kitchen", "xml", "sha", "qpos_names", "pool", "pinned", "prepared", "gaps", "meta")


def _scene(kitchen: str) -> _Scene:
    """Compile and permanently prepare the cached render model.

    Reference rendering compiles its own pristine model, so temporary visibility
    changes here cannot affect source frames.
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
            + _BUILD_CMD.format(kitchen=kitchen)
        )
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
        raise ValueError(
            f"{kitchen}: cached qpos has {pool.shape[1]} columns, model nq is "
            f"{s.model.nq}; the cache and the MJCF are from different builds"
        )
    s.pool = pool
    # Pin invisible coordinates to the same recorded row in every process.
    s.pinned = pool[0].copy()
    _SCENE_CACHE[kitchen] = s
    return s


class _Ctx:
    __slots__ = ("wm", "wd", "rc", "bgra", "graph", "nworld", "res", "wp", "mjw", "cam")


def _context(
    kitchen: str, camera: str, res: int, nworld: int, tex_lod: int = 1, backface_cull: bool | None = None
) -> _Ctx:
    """Cache a graph by kitchen, camera, resolution, world count, and render profile.

    Texture LOD and culling belong in the key: plain and corrected classes can
    coexist. None leaves culling at the backend default.
    """
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    key = (kitchen, camera, int(res), int(nworld), int(tex_lod), backface_cull)
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
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
    c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in ("cam_pos", "cam_quat", "light_diffuse")})
    # Use a 32-thread render block for this mesh-heavy scene.
    c.wm.block_dim.render = 32
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # Capture inside both patches; replay otherwise retains the stock generated kernel.
    from ..render.mjwarp import lightspec as LS

    profile = {} if backface_cull is None else {"enable_backface_culling": bool(backface_cull)}
    with MW.gl_rounding(), LS.specialised_lights():
        c.rc = MW.create_render_context(
            m,
            tex_lod=int(tex_lod),
            nworld=int(nworld),
            cam_res=(int(res), int(res)),
            render_rgb=True,
            cam_active=[i == cid for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS),
            use_textures=True,
            use_shadows=True,
            render_skybox=True,
            **profile,
        )
        _frame(c)
        wp.synchronize()
        # Keep a live buffer view; swapping channels here would cache a frozen copy.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
        with wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    _CTX_CACHE.put(key, c)
    return c


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


def _reference(kitchen: str, size: int):
    """Cache a pristine source model and process-lifetime renderer showing geom group 1."""
    import mujoco

    key = (kitchen, int(size))
    hit = _REF_CACHE.get(key)
    if hit is None:
        m = mujoco.MjModel.from_xml_path(_scene(kitchen).xml)
        vopt = mujoco.MjvOption()
        vopt.geomgroup[:] = 0
        vopt.geomgroup[1] = 1
        hit = _REF_CACHE[key] = (m, mujoco.MjData(m), mujoco.Renderer(m, width=int(size), height=int(size)), vopt)
    return hit


class RoboCasaBase(Env):
    """VISUAL TIER: MJWarp rendering for one cached kitchen and camera.

    State stores the camera-visible qpos subset; remaining coordinates are
    pinned to the recorded pool's first row. It is a render pose, not a complete
    simulator snapshot. Textured primitives and edge filtering can differ from source.
    """

    family = "robocasa"

    KITCHEN: str = ""
    CAMERA: str = ""
    default_size = 256
    fidelity_tier = "visual"
    parity_interior_min = None
    # Plain profile; corrected subclasses take their profile from the checkpoint.
    TEX_LOD: int = 1
    BACKFACE_CULL: bool | None = None

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int = 1,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        s = _scene(self.KITCHEN)
        live = _live_dims(self.KITCHEN, self.CAMERA, int(size or self.default_size))
        self._live_idx = live
        self.state_dim = len(live)
        self.state_names = tuple(s.qpos_names[i] for i in live)
        # Public qpos contains only visible indices; _full_qpos restores pinned coordinates.
        self.state_layout = ((("qpos",), self.state_dim, 0),)
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        self._s = s
        self._live_t = torch.as_tensor(live, device=self.device, dtype=torch.long)
        self._pinned_t = torch.as_tensor(s.pinned, device=self.device, dtype=torch.float32)

    def _full_qpos(self, states: torch.Tensor) -> torch.Tensor:
        q = self._pinned_t.unsqueeze(0).repeat(states.shape[0], 1)
        q[:, self._live_t] = states
        return q

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample recorded poses with seeded random yaw on free-joint objects."""
        rng = rng or self.rng
        pool = _yawed_pool(self.KITCHEN)
        idx = rng.integers(0, len(pool), int(batch))
        return pool[idx][:, self._live_idx].astype(np.float32)

    def make_variation_space(self) -> Dict:
        """Expose per-world camera offsets and diffuse-light tint/intensity.

        Textures and material colors stay fixed in the shared render context.
        """
        return Dict(
            {
                "camera": Dict(
                    {
                        # Translation is expressed in the camera mount's local frame.
                        "offset": Box(low=-0.06, high=0.06, shape=(3,), init_value=[0.0, 0.0, 0.0]),
                        "pan": Box(low=-8.0, high=8.0, shape=(1,), init_value=[0.0]),
                        "tilt": Box(low=-8.0, high=8.0, shape=(1,), init_value=[0.0]),
                    }
                ),
                "light": Dict(
                    {
                        "scale": Box(low=0.45, high=1.6, shape=(1,), init_value=[1.0]),
                        "tint": Box(low=0.6, high=1.4, shape=(3,), init_value=[1.0, 1.0, 1.0]),
                    }
                ),
            }
        )

    def _camera_pose(self, offset: np.ndarray, pan: np.ndarray, tilt: np.ndarray):
        """Apply camera-local translation and right-multiplied pan/tilt quaternions.

        Pan rotates about local y, tilt about local x; the camera looks along -z.
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
        # Flatten (B, 1) angles before scalar iteration.
        pan, tilt = np.asarray(pan).reshape(-1), np.asarray(tilt).reshape(-1)
        # Deduplicate exactly equal angle pairs without quantizing their values.
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

        qpos = self._full_qpos(s)
        offset = sty["camera.offset"].reshape(B, 3).cpu().numpy().astype(np.float64)
        pan = sty["camera.pan"].reshape(B, 1).cpu().numpy().astype(np.float64)
        tilt = sty["camera.tilt"].reshape(B, 1).cpu().numpy().astype(np.float64)
        cpos, cquat = self._camera_pose(offset, pan, tilt)
        cpos = torch.as_tensor(cpos, device=self.device, dtype=torch.float32)
        cquat = torch.as_tensor(cquat, device=self.device, dtype=torch.float32)
        base_light = torch.as_tensor(
            np.asarray(self._s.model.light_diffuse, np.float64), device=self.device, dtype=torch.float32
        )
        light = base_light.unsqueeze(0) * sty["light.scale"].reshape(B, 1, 1) * sty["light.tint"].reshape(B, 1, 3)

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
            # Order Torch writes, Warp replay, and buffer reads across their streams.
            torch.cuda.synchronize()
            wp.capture_launch(c.graph)
            wp.synchronize()
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

    @classmethod
    def reference_env(cls, size: int = 256):
        """Return a cached MuJoCo reference showing robosuite's visual geom group 1."""
        return _reference(cls.KITCHEN, int(size))

    @classmethod
    def reference_render(cls, ref, states, style=None) -> np.ndarray:
        """Render source frames with optional packed ``(B, style_dim)`` appearance.

        Use the same camera transform as the fast renderer and restore camera/light
        model fields in finally; the reference model is shared across calls.
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
            probe = _STYLE_PROBE.get((cls, int(r.height)))
            if probe is None:
                probe = _STYLE_PROBE[(cls, int(r.height))] = cls(num_envs=1, size=int(r.height), device="cpu")
            packed = np.asarray(style, np.float64).reshape(len(st), -1)
            leaf = {p: packed[:, off : off + w] for p, off, w in probe.variation_space.flat_layout()}
            cpos, cquat = probe._camera_pose(leaf["camera.offset"], leaf["camera.pan"], leaf["camera.tilt"])
            lights = base_light[None] * leaf["light.scale"].reshape(-1, 1, 1) * leaf["light.tint"].reshape(-1, 1, 3)
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
        """Identify corrector weights by class, family, kitchen, camera, and XML hash."""
        return {
            "env_class": cls.__name__,
            "family": "robocasa",
            "scene": cls.KITCHEN,
            "camera": cls.CAMERA,
            "scene_sha": _scene(cls.KITCHEN).sha,
        }


def _yawed_pool(kitchen: str) -> np.ndarray:
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
        pool[:, adr : adr + 4] = np.stack(
            [np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], 1
        )
    _POOL_CACHE[kitchen] = pool
    return pool


def _live_dims(kitchen: str, camera: str, size: int) -> np.ndarray:
    """Probe qpos visibility using source rendering and cache indices by XML hash.

    Keep a dimension only when donor values or nudges move pixels for every
    probe state. Pin omitted coordinates when reconstructing full qpos.
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

    # Probe visibility with the source renderer so approximations cannot hide live dimensions.
    probes = 4
    base_states = pool[rng.integers(0, len(pool), probes)]
    donors = pool[rng.integers(0, len(pool), 4)]
    live = []
    for dim in range(nq):
        alive_everywhere = True
        for q0 in base_states:
            base = draw(q0)
            moved = False
            for cand in list(donors[:, dim]) + [q0[dim] + 25.0, q0[dim] - 25.0, q0[dim] + 1.0, q0[dim] + 0.5]:
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
        json.dump(
            {
                "sha": s.sha,
                "camera": camera,
                "size": int(size),
                "nq": nq,
                "live": live.tolist(),
                "live_names": [names[i] for i in live],
            },
            f,
            indent=1,
        )
    return live


# Recorded source comparisons and fixed visual acceptance thresholds.
_AT = (
    "256 px, 64 states x 4 independent draws from sample_states (fresh env per draw, seeds 0-3), "
    "H200, mujoco 3.11.0 / mujoco-warp 3.11.0 / warp 1.16.0, graph captured inside gl_rounding; "
    "the measured values below are the WORST of the four and draw_spread is [min, max]"
)
FIDELITY = {
    ("L52S03", "robot0_agentview_left"): {
        "mean_abs": 6.41,
        "within8_frac": 0.833,
        "ssim": 0.907,
        "interior_mean_abs": 1.27,
        "interior_exact_frac": 0.408,
        "draw_spread": {
            "mean_abs": [6.36, 6.41],
            "within8_frac": [0.833, 0.834],
            "ssim": [0.907, 0.909],
            "interior_exact_frac": [0.408, 0.430],
        },
        "mean_abs_max": 8.5,
        "within8_min": 0.78,
        "ssim_min": 0.86,
        "measured_at": _AT,
    },
    ("L52S03", "robot0_agentview_right"): {
        "mean_abs": 5.52,
        "within8_frac": 0.846,
        "ssim": 0.921,
        "interior_mean_abs": 0.63,
        "interior_exact_frac": 0.672,
        "draw_spread": {
            "mean_abs": [5.37, 5.52],
            "within8_frac": [0.846, 0.850],
            "ssim": [0.921, 0.925],
            "interior_exact_frac": [0.672, 0.687],
        },
        "mean_abs_max": 7.5,
        "within8_min": 0.79,
        "ssim_min": 0.88,
        "measured_at": _AT,
    },
    ("L52S03", "robot0_eye_in_hand"): {
        "mean_abs": 9.69,
        "within8_frac": 0.826,
        "ssim": 0.857,
        "interior_mean_abs": 4.02,
        "interior_exact_frac": 0.433,
        "draw_spread": {
            "mean_abs": [4.30, 9.69],
            "within8_frac": [0.826, 0.868],
            "ssim": [0.857, 0.953],
            "interior_exact_frac": [0.433, 0.458],
        },
        "mean_abs_max": 14.0,
        "within8_min": 0.76,
        "ssim_min": 0.80,
        "measured_at": _AT,
    },
    ("L45S33", "robot0_agentview_left"): {
        "mean_abs": 32.01,
        "within8_frac": 0.364,
        "ssim": 0.822,
        "interior_mean_abs": 45.53,
        "interior_exact_frac": 0.153,
        "draw_spread": {
            "mean_abs": [31.90, 32.01],
            "within8_frac": [0.364, 0.370],
            "ssim": [0.822, 0.824],
            "interior_exact_frac": [0.153, 0.154],
        },
        "mean_abs_max": 38.0,
        "within8_min": 0.32,
        "ssim_min": 0.77,
        "measured_at": _AT,
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
        made[cname] = register(
            type(
                cname,
                (RoboCasaBase,),
                dict(
                    KITCHEN=kitchen,
                    CAMERA=camera,
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
                        f"when a policy will be evaluated on official RoboCasa pixels. See visual_note."
                    ),
                ),
            )
        )
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
    "policy."
)


# Each cached corrector owns a CUDA graph and fixed input/output buffers.
_CORRECTOR_CACHE = _RES.BoundedCache(4, name="robocasa.correctors", on_evict=lambda c: c.close())


class RoboCasaCorrectedBase(RoboCasaBase):
    """VISUAL TIER, LEARNED decoder for a specific kitchen and camera.

    A checkpoint supplies the low-resolution MJWarp profile and the network
    that maps it to the output resolution. State and appearance use the plain
    class's contracts; synthesized detail is not source renderer output.
    """

    render_decoder = "learned"
    # Training may supply an unsealed corrector to measure its eventual declaration.
    _corrector_override = None

    def corrector(self):
        """Return the cached verified corrector, or the explicit training override."""
        if self._corrector_override is not None:
            return self._corrector_override
        cls = type(self)
        hit = _CORRECTOR_CACHE.get(cls.__name__)
        if hit is None:
            hit = _CORRECTOR_CACHE.put(cls.__name__, RN.load(cls.corrector_cell_key()))
        return hit

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
        """Apply the corrector at its checkpoint output size and supersample 1.

        Reject other sizes rather than silently resampling the network output.
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
                f"this cell, which renders at any size."
            )
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        b = int(s.shape[0])
        if out is None:
            out = torch.empty((b, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(b, size, size, 3)
        low = super()._render(s, size=corr.render_size, supersample=1, variation=variation, style=style)
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
        "UNDERLYING RENDER'S OWN RESIDUAL, unchanged and still present under the correction: " + base_note
    )


def _make_corrected_classes() -> dict:
    """Register separate corrected classes only for available sealed checkpoints.

    Read rendering profiles and fidelity declarations from checkpoint headers;
    missing weights leave the plain classes unchanged.
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
            continue  # the loader will say why; do not register a wrong class
        vf = dict(hdr["visual_fidelity"])
        made[cname + "Corrected"] = register(
            type(
                cname + "Corrected",
                (RoboCasaCorrectedBase,),
                dict(
                    KITCHEN=kitchen,
                    CAMERA=camera,
                    TEX_LOD=int(hdr["render"]["tex_lod"]),
                    BACKFACE_CULL=hdr["render"]["backface_cull"],
                    default_size=int(hdr["render"]["out_size"]),
                    name=f"robocasa-{kitchen.lower()}-{camera.replace('robot0_', '').replace('_', '-')}-corrected",
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
                        f"stable_envs/render/rendernet.py."
                    ),
                ),
            )
        )
    return made


def missing_kitchens() -> dict[str, str]:
    """Map missing declared kitchens to commands that build their local caches."""
    have = set(available_kitchens())
    return {k: _BUILD_CMD.format(kitchen=k) for _, k, _ in CELLS if k not in have}


globals().update(_make_classes())
globals().update(_make_corrected_classes())
