from __future__ import annotations

import os

import numpy as np
import torch

from ... import tuning
from ..._core.mujoco_gl import configure_mujoco_gl, mujoco_renderer
from ..._core.warp_cuda import require_warp_cuda

configure_mujoco_gl()


def _require_gpu_renderer():
    import importlib.util

    import mujoco

    missing = [name for name in ("warp", "mujoco_warp") if importlib.util.find_spec(name) is None]
    if missing or tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 11):
        raise ImportError(
            "Fetch GPU rendering requires MuJoCo >=3.11, warp and mujoco_warp; "
            "CPU simulation is supported independently. Missing: " + ", ".join(missing)
        )


from ...env import Env
from ...registry import register
from ...render import _resources as _RES
from ...render import mjwarp as MW
from ...render.nvdiff import effective_rgba
from ...spaces import Box, Dict, Discrete

SCENE_XML = {"Reach": "reach.xml", "Push": "push.xml", "Slide": "slide.xml", "PickAndPlace": "pick_and_place.xml"}

# Joint order must match the compiled qpos layout; _build_bundle checks it.
ROBOT_JOINTS = (
    "slide0",
    "slide1",
    "slide2",
    "torso_lift",
    "head_pan",
    "head_tilt",
    "shoulder_pan",
    "shoulder_lift",
    "upperarm_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "r_gripper_finger",
    "l_gripper_finger",
)
OBJECT_JOINT = ("object_x", "object_y", "object_z", "object_quat_w", "object_quat_x", "object_quat_y", "object_quat_z")
GOAL_NAMES = ("goal_x", "goal_y", "goal_z")

# Compiled robot0:blue color identifies the geoms controlled by arm.color.
BLUE = (0.086, 0.506, 0.767)
# MuJoCo defaults for geoms without a material (mjv_initGeom).
DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION = 0.5, 0.5, 0.0
HIDDEN_GROUP = MW.HIDDEN_GROUP
VISIBLE_GROUPS = MW.DEFAULT_GROUPS
# Alpha stable-worldmodel writes into every robot0: geom for rendering.transparent_arm.
ARM_ALPHA = 0.3


class _Cache(dict):
    """Process-lifetime cache for shared models, reference renderers and state pools."""

    def get_or_build(self, key, build):
        hit = self.get(key)
        if hit is None:
            hit = self[key] = build()
        return hit


# MJWarp contexts may be evicted; shared EGL renderers remain alive for the process.
_CTX_CACHE = _RES.BoundedCache(6, name="fetch.ctx", on_evict=MW.release_cached_ctx)
_PLATE_CACHE = _RES.BoundedCache(32, name="fetch.plate")
_MODEL_CACHE = _Cache()
_RENDERER_CACHE = _Cache()
_POOL_CACHE = _Cache()
# Cache free-camera poses to avoid repeated MjvScene allocation and scene updates.
_CAM_CACHE = _RES.BoundedCache(256, name="fetch.cam")


def _asset_dir() -> str:
    import gymnasium_robotics as gr

    return os.path.join(os.path.dirname(gr.__file__), "envs", "assets", "fetch")


def _skybox(m) -> int:
    import mujoco

    return next(t for t in range(m.ntex) if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX))


def _sky_texels(m) -> np.ndarray:
    """A writable ``(texels, nchannel)`` view of the skybox texture."""
    t = _skybox(m)
    adr, nc = int(m.tex_adr[t]), int(m.tex_nchannel[t])
    return m.tex_data[adr : adr + int(m.tex_width[t]) * int(m.tex_height[t]) * nc].reshape(-1, nc)


def sky_color(m) -> np.ndarray:
    """The registered skybox's first texel; ``sky.color`` at this value draws the registered gradient."""
    return _sky_texels(m)[0, :3].astype(np.float64) / 255.0


def _flat_sky(m, color) -> bool:
    """Fill the skybox with ``color`` unless it is the registered default; return whether it did.

    Channels are truncated to uint8 as stable-worldmodel's ``background.color`` does.
    """
    color = np.asarray(color, np.float64).reshape(3)
    if np.array_equal(np.round(color, 6), np.round(sky_color(m), 6)):
        return False
    _sky_texels(m)[:, :3] = (np.clip(color, 0.0, 1.0) * 255.0).astype(np.uint8)
    return True


def _restore(m, pr) -> None:
    """Restore shared model colors, lighting, camera poses and skybox before and after baking."""
    _sky_texels(m)[:] = pr["sky"]
    m.mat_rgba[:] = pr["mat_rgba"]
    m.geom_rgba[:] = pr["geom_rgba"]
    m.light_diffuse[:] = pr["light_diffuse"]
    m.cam_pos[:] = pr["cam_pos"]
    m.cam_quat[:] = pr["cam_quat"]


def _style_groups(sty, paths, B: int):
    """Group appearance rows after rounding to six decimal places.

    Return unique rows ``(G, sum(widths))``, a group index per environment,
    and ``(offset, width)`` pairs for splitting each row into the requested paths.
    """
    offs, o = [], 0
    cols = []
    for path in paths:
        w = int(sty.layout[path][1])
        cols.append(sty[path].reshape(-1, w)[:B])
        offs.append((o, w))
        o += w
    key = torch.cat(cols, 1).cpu().numpy()
    uniq, inv = np.unique(np.round(key.astype(np.float64), 6), axis=0, return_inverse=True)
    return uniq, inv, offs


def _free_cam_pose(m, d, lookat, distance, azimuth, elevation):
    """Return the world position and quaternion of a MuJoCo free camera.

    Average the two stereo eye positions to match mono rendering.
    """
    import mujoco

    key = (
        id(m),
        tuple(np.round(np.asarray(lookat, np.float64), 9).tolist()),
        round(float(distance), 9),
        round(float(azimuth), 9),
        round(float(elevation), 9),
    )
    hit = _CAM_CACHE.get(key)
    if hit is not None:
        return hit
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, np.float64)
    cam.distance, cam.azimuth, cam.elevation = float(distance), float(azimuth), float(elevation)
    scn = mujoco.MjvScene(m, maxgeom=int(m.ngeom) + int(m.nsite) + 8)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_STATIC, scn)
    e0, e1 = scn.camera[0], scn.camera[1]
    pos = 0.5 * (np.array(e0.pos, np.float64) + np.array(e1.pos, np.float64))
    fwd = np.array(e0.forward, np.float64)
    up = np.array(e0.up, np.float64)
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(fwd, up / np.linalg.norm(up))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, z], axis=1).reshape(-1))
    return _CAM_CACHE.put(key, (pos, q))


class _Bundle:
    """Compiled Fetch model, geometry indices and unmodified appearance values."""

    __slots__ = (
        "model",
        "data",
        "info",
        "pristine",
        "cam_defaults",
        "static",
        "movers",
        "nq",
        "has_object",
        "state_names",
        "blender",
    )


def _build_bundle(scene: str) -> _Bundle:
    """Compile a render model with a fixed camera, movable goal geom and per-geom materials."""
    import mujoco

    hit = _MODEL_CACHE.get(scene)
    if hit is not None:
        return hit

    xml = os.path.join(_asset_dir(), SCENE_XML[scene])
    spec = mujoco.MjSpec.from_file(xml)
    base = spec.compile()
    d0 = mujoco.MjData(base)
    mujoco.mj_forward(base, d0)
    from gymnasium_robotics.envs.fetch.fetch_env import DEFAULT_CAMERA_CONFIG as CC

    cam_defaults = dict(
        lookat=np.asarray(CC["lookat"], np.float64),
        distance=float(CC["distance"]),
        azimuth=float(CC["azimuth"]),
        elevation=float(CC["elevation"]),
        fovy=float(base.vis.global_.fovy),
    )
    pos, quat = _free_cam_pose(
        base, d0, cam_defaults["lookat"], cam_defaults["distance"], cam_defaults["azimuth"], cam_defaults["elevation"]
    )

    for i in range(base.ngeom + 1):
        spec.add_material(name=f"se:mat{i}")
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = cam_defaults["fovy"]
    # Put the goal on a mocap body: MJWarp kinematics does not update world-body geom positions.
    gb = spec.worldbody.add_body()
    gb.name = "se:goalbody"
    gb.mocap = True
    tg = gb.add_geom()
    tg.name = "se:target"
    tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tsid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_SITE, "target0")
    tsize = float(base.site_size[tsid][0])
    tg.size = np.array([tsize, tsize, tsize])
    tg.rgba = np.asarray(base.site_rgba[tsid], np.float32)
    tg.contype, tg.conaffinity = 0, 0
    m = spec.compile()

    gname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    bname = lambda i: mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    # Match geoms by their order within each body, including unnamed geoms.
    # Compilation must preserve that order; names and counts are checked below.
    b_by_body: dict[int, list[int]] = {}
    for g in range(base.ngeom):
        b_by_body.setdefault(int(base.geom_bodyid[g]), []).append(g)
    target = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "se:target")
    s_by_body: dict[int, list[int]] = {}
    for g in range(m.ngeom):
        if g != target:
            s_by_body.setdefault(int(m.geom_bodyid[g]), []).append(g)
    back = {}
    for body, gs in s_by_body.items():
        bs = b_by_body[body]
        assert len(gs) == len(bs), f"{scene}: geom count changed in body {body}"
        for a, b in zip(gs, bs):
            assert gname(a) == bname(b), f"{scene}: geom {a} is {gname(a)!r}, base {b} is {bname(b)!r}"
            back[a] = b

    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")
    arm, translucent = [], []
    floor = obj = table = -1
    for g in range(m.ngeom):
        k = mat0 + g
        if g == target:
            rgba = np.asarray(tg.rgba, np.float64)
            spec_, shin, emis = DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION
        else:
            b = back[g]
            rgba = effective_rgba(base, b)
            mid = int(base.geom_matid[b])
            if mid >= 0:
                spec_ = float(base.mat_specular[mid])
                shin = float(base.mat_shininess[mid])
                emis = float(base.mat_emission[mid])
            else:
                spec_, shin, emis = DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION
            nm = bname(b)
            if nm == "floor0":
                floor = g
            elif nm == "object0":
                obj = g
            elif np.allclose(rgba[:3], BLUE, atol=2e-3):
                arm.append(g)
            elif (
                int(base.geom_matid[b]) >= 0
                and mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_MATERIAL, int(base.geom_matid[b])) == "table_mat"
            ):
                table = g
            if rgba[3] < 1.0:
                translucent.append(g)
        m.mat_rgba[k] = rgba
        m.mat_specular[k] = spec_
        m.mat_shininess[k] = shin
        m.mat_emission[k] = emis
        m.mat_reflectance[k] = 0.0
        m.mat_texid[k] = -1
        m.geom_matid[g] = k
        m.geom_rgba[g] = rgba
    assert floor >= 0 and table >= 0, f"{scene}: floor/table geom not found"
    assert arm, f"{scene}: no geom carries the robot0:blue colour"

    # MJWarp cannot draw sites. Hide them from the plate and render the goal as a moving geom.
    m.site_group[:] = HIDDEN_GROUP
    # Exclude translucent markers from this render model; this MJWarp path does not blend alpha.
    m.geom_group[translucent] = HIDDEN_GROUP
    # This model is only for rendering; MJWarp MULTICCD rejects nonzero collision margins.
    m.geom_margin[:] = 0.0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 4 * FetchBase.default_size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 4 * FetchBase.default_size)

    b = _Bundle()
    b.model, b.data = m, mujoco.MjData(m)
    b.nq = int(m.nq)
    b.has_object = obj >= 0
    b.info = dict(
        cam=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam"),
        target=target,
        mat0=mat0,
        floor=floor,
        table=table,
        object=obj,
        arm=arm,
        goal_mocap=int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "se:goalbody")]),
        robot=[g for g in range(m.ngeom) if "robot0:" in gname(g)],
        sky=_skybox(m),
    )
    b.cam_defaults = cam_defaults
    b.pristine = dict(
        mat_rgba=np.array(m.mat_rgba, copy=True),
        geom_rgba=np.array(m.geom_rgba, copy=True),
        light_diffuse=np.array(m.light_diffuse, copy=True),
        cam_pos=np.array(m.cam_pos, copy=True),
        cam_quat=np.array(m.cam_quat, copy=True),
        sky=np.array(_sky_texels(m), copy=True),
    )
    b.blender = None
    # Only geometry welded to the world can be baked; the goal must stay in the moving pass.
    b.static = [
        g
        for g in range(m.ngeom)
        if g != target and int(m.body_weldid[m.geom_bodyid[g]]) == 0 and int(m.body_mocapid[m.geom_bodyid[g]]) < 0
    ]
    b.movers = np.array([g for g in range(m.ngeom) if g not in b.static], np.int64)

    names = list(ROBOT_JOINTS) + (list(OBJECT_JOINT) if b.has_object else []) + list(GOAL_NAMES)
    got = []
    for j in range(m.njnt):
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").replace("robot0:", "")
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            got += list(OBJECT_JOINT)
        else:
            got.append(nm.replace("_joint", ""))
    assert tuple(got) == tuple(names[:-3]), (
        f"{scene}: declared joint names {tuple(names[:-3])} do not match the model's layout {tuple(got)}"
    )
    b.state_names = tuple(names)
    _MODEL_CACHE[scene] = b
    return b


def _renderer(scene: str, res: int):
    """Return the shared reference renderer for a scene and resolution.

    Keep its EGL context alive for the process; releasing it can invalidate
    other live renderers on supported drivers.
    """
    return _RENDERER_CACHE.get_or_build(
        (scene, int(res)), lambda: mujoco_renderer(_build_bundle(scene).model, width=int(res), height=int(res))
    )


class _Ctx:
    __slots__ = ("wm", "wd", "rc", "bgra", "seg", "graph", "nworld", "res", "wp", "mjw")


def _context(scene: str, res: int, nworld: int, translucent: bool = False) -> _Ctx:
    """A batched MJWarp context; ``translucent`` leaves the robot out and also renders depth."""
    import mujoco
    import mujoco_warp as mjw

    wp = require_warp_cuda("Fetch GPU rendering")
    key = (str(wp.get_device()), scene, int(res), int(nworld), bool(translucent))
    hit = _CTX_CACHE.get(key)
    if hit is not None:
        return hit
    b = _build_bundle(scene)
    m = b.model
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    c = _Ctx()
    c.wp, c.mjw, c.nworld, c.res = wp, mjw, int(nworld), int(res)
    batched = {k: int(nworld) for k in ("mat_rgba", "light_diffuse", "cam_pos", "cam_quat")}
    groups = np.array(m.geom_group, copy=True)
    if translucent:
        m.geom_group[b.info["robot"]] = HIDDEN_GROUP
    try:
        c.wm = mjw.put_model(m, batch_sizes=batched)
        c.wd = mjw.put_data(m, d0, nworld=int(nworld))
        # Capture inside gl_rounding so graph replay uses the rounding kernel, not truncation.
        with MW.gl_rounding():
            c.rc = MW.create_render_context(
                m,
                nworld=int(nworld),
                cam_res=(int(res), int(res)),
                render_rgb=True,
                render_seg=True,
                render_depth=bool(translucent),
                cam_active=[i == b.info["cam"] for i in range(m.ncam)],
                enabled_geom_groups=list(VISIBLE_GROUPS),
                use_textures=True,
                use_shadows=False,
                render_skybox=True,
                enable_specular=True,
            )
    finally:
        m.geom_group[:] = groups
    with MW.gl_rounding():
        _frame(c)
        wp.synchronize()
        # Keep a view of the live Warp buffer. Channel-swapping here would copy and freeze the frame.
        c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
        c.seg = wp.to_torch(c.rc.seg_data).reshape(int(nworld), int(res), int(res), 2)[..., 0]
        with c.wp.ScopedCapture() as cap:
            _frame(c)
        c.graph = cap.graph
    _CTX_CACHE.put(key, c)
    return c


def _frame(c: _Ctx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)


class FetchBase(Env):
    """Fetch renderer shared by Reach, Push, Slide and PickAndPlace.

    Rendering uses a private ``qpos + goal`` projection. Registered classes
    combine this renderer with ``FetchSemantics`` for simulation and full-State
    restoration. Reward and observation variants share the scene renderer.
    """

    family = "fetch"

    SCENE: str = ""
    REF_ID: str = ""
    default_size = 480  # gymnasium-robotics' own DEFAULT_SIZE
    default_supersample = 1
    # Changing these leaves requires a new static plate; mover colors are uploaded per world.
    PLATE_PATHS = (
        "floor.color",
        "table.color",
        "sky.color",
        "light.diffuse",
        "camera.distance_scale",
        "camera.azimuth_delta",
        "camera.elevation_delta",
    )
    parity_interior_min = 0.99
    parity_note = (
        "Two declared residuals, both structural to the MJWarp backend. (1) The robot's 18 meshes are "
        "shaded per pixel while MuJoCo shades them per vertex (Gouraud), so they differ by a level or two "
        "where they differ at all: 0.8% of interior pixels, mean 0.009/255. The robot is 8.9% of the "
        "frame. (2) The translucent mocap markers (a 1 cm cube at alpha 0.7, three 2 m crosshair bars at "
        "alpha 0.1) are not drawn: mujoco_warp has no alpha path and would paint them opaque. They cover "
        "1.2-1.5% of all pixels (mean 0.32/255) but 0.000002 of interior pixels, since a translucent bar "
        "over a flat surface leaves no flat 3x3 neighbourhood. The static 91% of the frame (floor, table, "
        "skybox) is baked through MuJoCo itself and is exact."
    )

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        """Initialize the compact renderer layout from the compiled scene model."""
        _require_gpu_renderer()
        b = _build_bundle(self.SCENE)
        self.state_dim = len(b.state_names)
        self.state_names = b.state_names
        # Reach omits the object joint, so the compact projection width depends on the scene.
        self.state_layout = ((("qpos",), int(b.nq), 0), (("goal",), 3, int(b.nq)))
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        self._b = b
        assert b.nq + 3 == self.state_dim, f"{type(self).__name__}: model nq={b.nq} but state_dim={self.state_dim}"
        self._check_plate_is_constant()

    def _close(self) -> None:
        """Drop instance tensors while retaining caches shared with other environments."""
        self.__dict__.pop("_idx_cache", None)

    def _check_plate_is_constant(self) -> None:
        """Check that hiding all moving geometry makes the baked frame independent of pose."""
        st = self.sample_states(2, rng=np.random.default_rng(0))
        a = self._bake_plate(128, None, states=st[0])
        c = self._bake_plate(128, None, states=st[1])
        d = int((a.to(torch.int16) - c.to(torch.int16)).abs().max())
        assert d == 0, (
            f"{type(self).__name__}: the baked static plate differs by {d}/255 between two states with "
            f"every mover hidden, so the floor/table/skybox depend on pose"
        )

    def make_variation_space(self) -> Dict:
        """Define colors, lighting and camera settings from the compiled model defaults.

        Colors remain floating point to preserve MuJoCo material values. ``sky.color`` at
        its default draws the registered gradient skybox; any other value fills it flat.
        ``rendering.transparent_arm`` draws every robot geom at alpha 0.3.
        """
        b = _build_bundle(self.SCENE)
        info = b.info
        base = b.pristine["mat_rgba"]
        space = {
            "floor": Dict(
                {
                    "color": Box(
                        low=0.0, high=1.0, shape=(3,), init_value=base[info["mat0"] + info["floor"], :3].tolist()
                    )
                }
            ),
            "table": Dict(
                {
                    "color": Box(
                        low=0.0, high=1.0, shape=(3,), init_value=base[info["mat0"] + info["table"], :3].tolist()
                    )
                }
            ),
            "goal": Dict(
                {
                    "color": Box(
                        low=0.0, high=1.0, shape=(3,), init_value=base[info["mat0"] + info["target"], :3].tolist()
                    )
                }
            ),
            "arm": Dict(
                {
                    "color": Box(
                        low=0.0, high=1.0, shape=(3,), init_value=base[info["mat0"] + info["arm"][0], :3].tolist()
                    )
                }
            ),
            "sky": Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=sky_color(b.model).tolist())}),
            "rendering": Dict({"transparent_arm": Discrete(2, init_value=0)}),
            "light": Dict({"diffuse": Box(low=0.0, high=2.0, shape=(1,), init_value=[1.0])}),
            "camera": Dict(
                {
                    "distance_scale": Box(low=0.6, high=1.6, shape=(1,), init_value=[1.0]),
                    "azimuth_delta": Box(low=-25.0, high=25.0, shape=(1,), init_value=[0.0]),
                    "elevation_delta": Box(low=-20.0, high=20.0, shape=(1,), init_value=[0.0]),
                }
            ),
        }
        if b.has_object:
            space["object"] = Dict(
                {
                    "color": Box(
                        low=0.0, high=1.0, shape=(3,), init_value=base[info["mat0"] + info["object"], :3].tolist()
                    )
                }
            )
        return Dict(space)

    def _var(self, variation, path) -> np.ndarray:
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)

    def _camera(self, variation):
        b = self._b if hasattr(self, "_b") else _build_bundle(self.SCENE)
        c = b.cam_defaults
        return _free_cam_pose(
            b.model,
            b.data,
            c["lookat"],
            c["distance"] * float(self._var(variation, "camera.distance_scale")[0]),
            c["azimuth"] + float(self._var(variation, "camera.azimuth_delta")[0]),
            c["elevation"] + float(self._var(variation, "camera.elevation_delta")[0]),
        )

    def _plate_key(self, res, variation):
        return (
            self.SCENE,
            int(res),
            str(self.device),
            *[tuple(np.round(self._var(variation, p), 6).tolist()) for p in self.PLATE_PATHS],
        )

    def _plate(self, res, variation) -> torch.Tensor:
        key = self._plate_key(res, variation)
        hit = _PLATE_CACHE.get(key)
        if hit is not None:
            return hit
        return _PLATE_CACHE.put(key, self._bake_plate(res, variation))

    def _bake_plate(self, res, variation, states=None) -> torch.Tensor:
        """Bake the static scene with MuJoCo, restoring shared model appearance afterward."""
        import mujoco

        b = _build_bundle(self.SCENE)
        m, d, info = b.model, b.data, b.info
        pr = b.pristine
        _restore(m, pr)
        # Write geom and material colors together: MuJoCo and MJWarp resolve them differently.
        for path, g in (("floor.color", info["floor"]), ("table.color", info["table"])):
            col = self._var(variation, path)
            m.mat_rgba[info["mat0"] + g, :3] = col
            m.geom_rgba[g, :3] = col
        m.light_diffuse[:] = pr["light_diffuse"] * float(self._var(variation, "light.diffuse")[0])
        pos, quat = self._camera(variation)
        m.cam_pos[info["cam"]] = pos
        m.cam_quat[info["cam"]] = quat
        r = _renderer(self.SCENE, int(res))
        flat = _flat_sky(m, self._var(variation, "sky.color"))
        if flat:
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r._mjr_context, info["sky"])

        keep = np.array(m.geom_group, copy=True)
        m.geom_group[b.movers] = HIDDEN_GROUP
        if states is not None:
            d.qpos[:] = np.asarray(states, np.float64)[: b.nq]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=info["cam"])
        img = r.render().copy()
        m.geom_group[:] = keep
        _restore(m, pr)
        if flat:
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r._mjr_context, info["sky"])
        return torch.as_tensor(img, device=self.device)

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
        """Render packed states ``(..., state_dim)`` to uint8 RGB ``(..., H, W, 3)``.

        Ray-cast the moving geometry in batches and composite it over static plates
        shared by appearance. Output stays on the environment device.
        """
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        res = size * ss
        sty = self.resolve_style(style, variation, batch=B)
        tuning.on_render("fetch-mjwarp", size, B, self)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        uniq, inv, offs = _style_groups(sty, self.PLATE_PATHS, B)
        plates, plates1 = [], []
        for row in uniq:
            over = {p: row[a : a + w] for p, (a, w) in zip(self.PLATE_PATHS, offs)}
            plates.append(self._plate(res, over))
            plates1.append(plates[-1] if ss == 1 else self._plate(size, over))
        plates_t = torch.stack(plates)  # (G, res, res, 3)
        plates1_t = plates_t if ss == 1 else torch.stack(plates1)
        grp = torch.as_tensor(inv, device=self.device, dtype=torch.long)
        # Camera grouping is batch-wide; repeating it per chunk adds host transfers and synchronization.
        campos, camquat = self._camera_batch(sty, B)
        arm = self._translucent_lanes(sty, B)
        # More worlds per context reduce launches and increase framebuffer memory.
        chunk = tuning.get("fetch_mjw_worlds", path="fetch-mjwarp", size=size, batch=B, device=self.device)
        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            n = hi - lo
            nw = 1 if n <= 1 else min(chunk, 1 << (n - 1).bit_length())
            g = grp[lo:hi]
            pl = plates_t[0] if len(uniq) == 1 else plates_t[g]
            flags = None if arm is None else arm[lo:hi]
            comp = mv = None
            if flags is None or not flags.all():
                comp, mv = self._composite(_context(self.SCENE, res, nw), s, sty, lo, hi, nw, campos, camquat, pl)
            if flags is not None and flags.any():
                c = _context(self.SCENE, res, nw, translucent=True)
                comp_t, mv_t = self._composite(c, s, sty, lo, hi, nw, campos, camquat, pl)
                self._blend_arm(c, comp_t, mv_t, flags)
                if comp is None:
                    comp, mv = comp_t, mv_t
                else:
                    lane = torch.as_tensor(flags, device=self.device).view(n, 1, 1)
                    comp = torch.where(lane.unsqueeze(-1), comp_t, comp)
                    mv = torch.where(lane, mv_t, mv)
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
                pl1 = plates1_t[0] if len(uniq) == 1 else plates1_t[g]
                buf[lo:hi] = torch.where(any_mv.unsqueeze(-1), small, pl1)
        return buf.view(*lead, size, size, 3) if lead else buf

    def _composite(self, c: _Ctx, s, sty, lo, hi, nw, campos, camquat, pl):
        """Ray-cast one chunk and composite its movers over the plate; return the frame and mover mask."""
        n = hi - lo
        self._upload(c, s[lo:hi], sty, lo, hi, nw, campos, camquat)
        # Torch writes and Warp replay use different streams; synchronize before and after replay.
        torch.cuda.synchronize()
        c.wp.capture_launch(c.graph)
        c.wp.synchronize()
        rgb, seg = c.bgra[:n][..., [2, 1, 0]], c.seg[:n]
        # Index with int32 segmentation IDs to avoid a full-frame int64 temporary.
        mv = self._mover_lut().index_select(0, (seg + 1).reshape(-1)).view(seg.shape)
        return torch.where(mv.unsqueeze(-1), rgb, pl), mv

    def _translucent_lanes(self, sty, B: int):
        """Host bool mask of lanes drawing a transparent arm, or None when no lane does."""
        if "rendering.transparent_arm" not in sty.layout:
            return None
        flags = sty["rendering.transparent_arm"].reshape(-1)[:B].cpu().numpy() > 0.5
        return flags if flags.any() else None

    def _blend_arm(self, c: _Ctx, comp: torch.Tensor, mv: torch.Tensor, flags: np.ndarray) -> None:
        """Blend the translucent robot into ``comp`` in place and add the pixels it covers to ``mv``."""
        from .transparency import ArmBlender

        b = self._b
        if b.blender is None:
            b.blender = ArmBlender(b.model, b.info["robot"])
        n = int(comp.shape[0])
        frame = comp.view(n, -1, 3)
        touched = torch.empty(frame.shape[:2], dtype=torch.uint8, device=self.device)
        lanes = torch.as_tensor(flags.astype(np.int32), device=self.device)
        torch.cuda.synchronize()
        wp = c.wp
        b.blender.launch(
            c.wm,
            c.wd,
            c.rc,
            b.info["cam"],
            wp.from_torch(lanes),
            wp.from_torch(frame),
            wp.from_torch(touched),
            ARM_ALPHA,
        )
        wp.synchronize()
        mv |= touched.view(mv.shape).bool()

    def _camera_batch(self, sty, B: int):
        """Return device camera positions ``(B, 3)`` and quaternions ``(B, 4)`` for a batch."""
        keys = ("camera.distance_scale", "camera.azimuth_delta", "camera.elevation_delta")
        uniq, inv, offs = _style_groups(sty, keys, B)
        poses = [self._camera({k: r[a : a + w] for k, (a, w) in zip(keys, offs)}) for r in uniq]
        pos = torch.as_tensor(np.stack([p for p, _ in poses])[inv], device=self.device, dtype=torch.float32)
        quat = torch.as_tensor(np.stack([q for _, q in poses])[inv], device=self.device, dtype=torch.float32)
        return pos, quat

    def _upload(
        self, c: _Ctx, s: torch.Tensor, sty, lo: int, hi: int, nw: int, campos: torch.Tensor, camquat: torch.Tensor
    ) -> None:
        """Copy state, appearance and camera poses into per-world MJWarp arrays."""
        b = self._b
        info = b.info
        n = hi - lo
        wp = c.wp
        qpos = wp.to_torch(c.wd.qpos)
        qpos[:n] = s[:, : b.nq]
        wp.to_torch(c.wd.mocap_pos)[:n, info["goal_mocap"]] = s[:, b.nq : b.nq + 3]
        mat = wp.to_torch(c.wm.mat_rgba)
        m0 = info["mat0"]
        mat[:n, m0 + info["floor"], :3] = sty["floor.color"][lo:hi]
        mat[:n, m0 + info["table"], :3] = sty["table.color"][lo:hi]
        mat[:n, m0 + info["target"], :3] = sty["goal.color"][lo:hi]
        mat[:n, self._idx("arm.mats", m0 + np.asarray(info["arm"], np.int64)), :3] = sty["arm.color"][lo:hi].unsqueeze(
            1
        )
        if b.has_object:
            mat[:n, m0 + info["object"], :3] = sty["object.color"][lo:hi]
        wp.to_torch(c.wm.light_diffuse)[:n] = self._light0().unsqueeze(0) * sty["light.diffuse"][lo:hi].reshape(
            n, 1, 1
        )
        cam = info["cam"]
        wp.to_torch(c.wm.cam_pos)[:n, cam] = campos[lo:hi]
        wp.to_torch(c.wm.cam_quat)[:n, cam] = camquat[lo:hi]

    def _cached(self, key: str, build) -> torch.Tensor:
        cache = self.__dict__.setdefault("_idx_cache", {})
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = build()
        return hit

    def _idx(self, key: str, values) -> torch.Tensor:
        return self._cached(key, lambda: torch.as_tensor(np.asarray(values, np.int64), device=self.device))

    def _light0(self) -> torch.Tensor:
        return self._cached(
            "light0",
            lambda: torch.as_tensor(self._b.pristine["light_diffuse"], device=self.device, dtype=torch.float32),
        )

    def _mover_lut(self) -> torch.Tensor:
        """Return a boolean lookup table indexed by segmentation ID plus one."""

        def build():
            lut = torch.zeros(int(self._b.model.ngeom) + 1, dtype=torch.bool, device=self.device)
            lut[torch.as_tensor(np.asarray(self._b.movers, np.int64) + 1, device=self.device)] = True
            return lut

        return self._cached("mover_lut", build)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample compact render states from a cached pool of source rollouts.

        The pool includes 0–25 actions after reset to vary the arm pose; reset
        alone leaves the arm unchanged. It is shared by variants of each scene.
        """
        rng = rng or self.rng
        pool = _state_pool(self.SCENE, self.REF_ID, self.state_dim, self.state_dim - 3)
        return pool[rng.integers(0, len(pool), int(batch))].copy()

    @classmethod
    def reference_env(cls, size: int = 480):
        """Construct a Gymnasium Robotics reference at the requested render resolution."""
        import gymnasium
        import gymnasium_robotics  # noqa: F401

        gymnasium.register_envs(gymnasium_robotics)
        return gymnasium.make(cls.REF_ID, render_mode="rgb_array", width=int(size), height=int(size))

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Render compact states through the reference after restoring joints and goal.

        Align mocap markers to the gripper before drawing so their positions do
        not depend on previous renders.
        """
        import mujoco
        from gymnasium_robotics.utils import mujoco_utils

        u = env.unwrapped
        m, d = u.model, u.data
        nq = int(m.nq)
        out = []
        checked = False
        for st in np.asarray(states, np.float64):
            d.qpos[:] = st[:nq]
            d.qvel[:] = 0.0
            u.goal = st[nq : nq + 3].copy()
            mujoco.mj_forward(m, d)
            mujoco_utils.reset_mocap2body_xpos(m, d)
            mujoco.mj_forward(m, d)
            got = np.concatenate([d.qpos[:nq], np.asarray(u.goal, np.float64)])
            # mj_forward renormalizes free-joint quaternions, so float32 snapshots need a small tolerance.
            err = float(np.abs(got - st).max())
            if err > 1e-5:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state (max {err:.3e})"
                )
            out.append(np.ascontiguousarray(u.render()))
            if not checked:
                checked = True
                cls._check_camera(u)
        return np.stack(out)

    @classmethod
    def _check_camera(cls, u) -> None:
        """Verify that the reference camera matches the camera used to compile the render model."""
        c = u.mujoco_renderer.viewer.cam
        want = _build_bundle(cls.SCENE).cam_defaults
        got = (float(c.distance), float(c.azimuth), float(c.elevation))
        exp = (want["distance"], want["azimuth"], want["elevation"])
        bad = (
            int(c.type) != 0
            or int(c.fixedcamid) != -1
            or max(abs(a - b) for a, b in zip(got, exp)) > 1e-9
            or float(np.abs(np.asarray(c.lookat) - want["lookat"]).max()) > 1e-9
        )
        if bad:
            raise AssertionError(
                f"{cls.__name__}: the official renderer's camera is "
                f"(type {int(c.type)}, fixed {int(c.fixedcamid)}, distance {got[0]}, azimuth "
                f"{got[1]}, elevation {got[2]}, lookat {np.asarray(c.lookat).tolist()}) but se:cam was "
                f"built from (FREE, {exp}, lookat {want['lookat'].tolist()})"
            )


def _state_pool(scene: str, ref_id: str, state_dim: int, nq: int, n: int = 512) -> np.ndarray:
    """Build compact states from seeded source rollouts without creating a renderer."""
    hit = _POOL_CACHE.get(scene)
    if hit is not None:
        return hit
    import gymnasium
    import gymnasium_robotics  # noqa: F401

    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make(ref_id)
    u = env.unwrapped
    # reset(seed=...) does not seed action_space; seed it separately for reproducible pools.
    env.action_space.seed(0)
    rng = np.random.default_rng(0)
    out = np.empty((n, state_dim), np.float32)
    for i in range(n):
        env.reset(seed=int(i))
        for _ in range(int(rng.integers(0, 26))):
            env.step(env.action_space.sample())
        q = np.array(u.data.qpos, np.float64)
        if len(q) > len(ROBOT_JOINTS):
            # Vary object yaw so quaternion perturbations do not all normalize to the same identity pose.
            yaw = float(rng.uniform(-np.pi, np.pi))
            adr = len(ROBOT_JOINTS)
            q[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        out[i, :nq] = q
        out[i, nq : nq + 3] = np.asarray(u.goal, np.float64)
    env.close()
    _POOL_CACHE[scene] = out
    return out


CELLS = (
    [(f"Fetch{t}", t, f"Fetch{t}-v4") for t in SCENE_XML]
    + [(f"Fetch{t}Dense", t, f"Fetch{t}Dense-v4") for t in SCENE_XML]
    + [(f"Fetch{t}Dict", t, f"Fetch{t}-v4") for t in SCENE_XML]
)


from .semantics import FetchSemantics as _FetchSemantics


def _make_classes() -> dict[str, type]:
    made = {}
    for cname, scene, ref in CELLS:
        suffix = cname[len(f"Fetch{scene}") :]
        made[cname] = register(
            type(
                cname,
                (_FetchSemantics, FetchBase),
                dict(
                    SCENE=scene,
                    REF_ID=ref,
                    name=f"fetch-{scene.lower()}" + (f"-{suffix.lower()}" if suffix else ""),
                    domain=scene.lower(),
                    reference_id=ref,
                    __module__=__name__,
                    __doc__=(f"{ref}: CPU simulation with restorable State and batched Fetch {scene} rendering."),
                ),
            )
        )
    return made


globals().update(_make_classes())

__all__ = ["FetchBase", *(c for c, _, _ in CELLS)]
