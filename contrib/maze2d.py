from __future__ import annotations

import numpy as np
import torch

from stable_envs._core.mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.render import _resources as _RES
from stable_envs.render import raycast3d as R3
from stable_envs.spaces import Box, Dict, Discrete

# Source material specular default.
MAT_SPECULAR = 0.5
# Source headlight ambient/diffuse and material-scaled specular constants.
SHADE_AMBIENT, SHADE_DIFFUSE, SHADE_SPECULAR, SHADE_EXPONENT = 0.4, 0.8, 0.05, 40.0
#: ``maze_v4.Maze.make_maze``: every block geom is written with this rgba and no material
WALL_RGB = (0.7, 0.5, 0.3)
#: ``point.xml``: ``<site name="particle_site" size="0.2" rgba="0.3 0.6 0.3 1">``
BALL_RGB = (0.3, 0.6, 0.3)
#: ``make_maze``: ``<site name="target" size="0.2*scaling" rgba="1 0 0 0.7">``
TARGET_RGB, TARGET_ALPHA = (1.0, 0.0, 0.0), 0.699999988079071
#: ``MazeEnv.position_noise_range``, the uniform jitter added to a cell centre at reset
POSITION_NOISE = 0.25

STATE_NAMES = ("ball_x", "ball_y", "goal_x", "goal_y")

_ENV_CACHE: dict = {}
_SCENE_CACHE_MAX = 64
_SCENE_CACHE = _RES.BoundedCache(_SCENE_CACHE_MAX, name="maze2d.scene")


def _offsets(paths, sty):
    out, off = [], 0
    for p in paths:
        w = int(sty.layout[p][1])
        out.append((off, w))
        off += w
    return out


class _Scene:
    """Cached plate, depth, camera and primitive definitions for one appearance."""

    __slots__ = (
        "plate",
        "plate_t",
        "cam",
        "lights",
        "size",
        "free_xy",
        "target_z",
        "scaling",
        "prims_ball",
        "prims_target",
        "prims_both",
        "radius",
    )


def _build_reference(maze: str, size: int):
    """Construct a source viewer with the free camera initialized at zero qpos.

    Gymnasium derives lookat from geom positions when the viewer is first created.
    Pin that state so reference frames and baked plates share one camera."""
    import gymnasium
    import gymnasium_robotics  # noqa: F401
    import mujoco

    gymnasium.register_envs(gymnasium_robotics)
    env = gymnasium.make(
        f"PointMaze_{maze}-v3", render_mode="rgb_array", width=int(size), height=int(size), continuing_task=False
    )
    env.reset(seed=0)
    u = env.unwrapped
    u.data.qpos[:] = 0.0
    u.data.qvel[:] = 0.0
    mujoco.mj_forward(u.model, u.data)
    u.point_env.mujoco_renderer._get_viewer("rgb_array")  # freezes lookat at the pinned state
    return env


def _scene_env(maze: str, size: int):
    """Cache one source environment per maze and size for the process lifetime.

    Retain its GL context: destroying it can corrupt other live render contexts."""
    key = (maze, int(size))
    hit = _ENV_CACHE.get(key)
    if hit is not None:
        return hit
    env = _build_reference(maze, size)
    u = env.unwrapped
    m = u.model
    m.vis.quality.offsamples = 4  # MuJoCo's default; what the datasets hold
    pristine = dict(
        site_rgba=np.array(m.site_rgba, copy=True),
        geom_rgba=np.array(m.geom_rgba, copy=True),
        mat_rgba=np.array(m.mat_rgba, copy=True),
        geom_matid=np.array(m.geom_matid, copy=True),
        cam_distance=float(u.point_env.mujoco_renderer.default_cam_config["distance"]),
    )
    _ENV_CACHE[key] = (env, u, pristine)
    return _ENV_CACHE[key]


def _official_frame(u) -> np.ndarray:
    """Bind the source viewer’s GL context and framebuffer before rendering.

    Another renderer may have made its own context current since the last call."""
    v = u.point_env.mujoco_renderer._get_viewer("rgb_array")
    v.make_context_current()
    v._set_mujoco_buffer()
    return u.point_env.mujoco_renderer.render("rgb_array").copy()


def _mat_and_tex(m, g):
    """Resolve geom/material RGB and mean texture RGB in MuJoCo modulation order."""
    mid = int(m.geom_matid[g])
    mat = np.asarray(m.geom_rgba[g], np.float64)[:3]
    tex = np.ones(3)
    if mid >= 0:
        if np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
            mat = np.asarray(m.mat_rgba[mid], np.float64)[:3]
        tid = m.mat_texid[mid]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        if tid >= 0:
            attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
            buf = getattr(m, attr)
            a0 = int(m.tex_adr[tid])
            h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
            reg = np.asarray(buf[a0 : a0 + 3 * h * w], np.float64).reshape(h, w, 3) / 255.0
            uq = np.unique(reg.reshape(-1, 3), axis=0)
            tex = uq[0] if len(uq) == 1 else reg.reshape(-1, 3).mean(0)
    return mat, tex


class PointMaze(Env):
    """Render PointMaze from ball and goal xy positions over a cached source plate.

    These four coordinates determine appearance, not simulator continuation.
    The free-camera initialization convention is shared with the source bridge."""

    family = "maze2d"

    MAZE: str = ""
    state_dim = 4
    state_names = STATE_NAMES
    # Pose-only layout; velocity and episode state are not represented.
    state_layout = ((("ball", "position"), 2, 0), (("goal", "position"), 2, 2))
    default_size = 224
    # Supersample silhouettes to approximate the source’s four-sample edges.
    default_supersample = 2
    # Screen-space cast window; must cover the scaled moving sphere.
    TILE = 48
    parity_interior_min = 1.0
    parity_note = (
        "MOVING-PRIMITIVE PARITY (supersample 2, against MuJoCo at its default offsamples=4). The "
        "baked plate carries every static pixel, so the residual is confined to the two spheres, "
        "which together cover 0.52% of a 224x224 frame. Whole-frame agreement is 99.53% of pixels "
        "exact at mean 0.024/255; on the pixels the two sites actually touch it is 41-47% within "
        "1/255 at mean 4.4/255. Two mechanisms, both measured and neither closable by fitting: "
        "(1) MuJoCo's 4x multisampling at the silhouette of a 12-pixel sphere, which an analytic "
        "caster reproduces only as coverage-free hard edges -- the same effect ogbench.py records "
        "for its agent; and (2) the fixed-function shading of a saturated primary colour, where the "
        "target's red channel clips at 1.0 and carries no information to fit against. A grid search "
        "over (ambient, diffuse, specular, exponent) around the model's declared headlight moved "
        "within-1 agreement from 0.410 to 0.468 and whole-frame exactness from 0.995326 to 0.995455, "
        "i.e. by less than the multisampling costs, so the DECLARED headlight constants are used "
        "rather than fitted ones. INTERIOR agreement, which is what parity_interior_min measures, is "
        "1.000000: the interior mask keeps only pixels whose 3x3 neighbourhood in the reference is "
        "one flat colour, and a shaded 12-pixel sphere has none."
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
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        sc = self._scene()
        assert sc.radius > 0, f"{type(self).__name__}: the particle site has no radius"

    def make_variation_space(self) -> Dict:
        """Expose float RGB materials, agent scale, target alpha, camera and floor style.

        Keep material colours in [0, 1]; uint8 quantization changes source shading.
        Set target alpha to zero to hide the goal marker."""
        return Dict(
            {
                "agent": Dict(
                    {
                        "color": Box(low=0.0, high=1.0, shape=(3,), init_value=list(BALL_RGB)),
                        "scale": Box(low=0.25, high=4.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "target": Dict(
                    {
                        "color": Box(low=0.0, high=1.0, shape=(3,), init_value=list(TARGET_RGB)),
                        "alpha": Box(low=0.0, high=1.0, shape=(1,), init_value=[TARGET_ALPHA]),
                    }
                ),
                "floor": Dict({"tint": Box(low=0.0, high=1.0, shape=(3,), init_value=[1.0, 1.0, 1.0])}),
                "walls": Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=list(WALL_RGB))}),
                "camera": Dict(
                    {
                        "distance_scale": Box(low=0.6, high=1.8, shape=(1,), init_value=[1.0]),
                        "elevation": Box(low=-90.0, high=-20.0, shape=(1,), init_value=[-45.0]),
                        "azimuth": Box(low=0.0, high=360.0, shape=(1,), init_value=[90.0]),
                    }
                ),
                # 0 keeps source checkerboard; 1 replaces its material with a flat colour.
                "rendering": Dict({"decolor_floor": Discrete(2, init_value=0)}),
            }
        )

    #: Style leaves that change the STATIC scene and therefore the baked plate.
    PLATE_PATHS = (
        "floor.tint",
        "walls.color",
        "camera.distance_scale",
        "camera.elevation",
        "camera.azimuth",
        "rendering.decolor_floor",
    )

    def _var(self, variation, path):
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)

    def _plate_key(self, size, variation):
        return (
            self.MAZE,
            int(size),
            str(self.device),
            *(tuple(np.round(self._var(variation, p), 6)) for p in self.PLATE_PATHS),
        )

    def _scene(self, size=None, variation=None) -> _Scene:
        size = int(size or self.size)
        key = self._plate_key(size, variation)
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            return hit
        return _SCENE_CACHE.put(key, self._bake(size, variation))

    def _bake(self, size: int, variation) -> _Scene:
        """Hide moving sites and render the static plate and depth for one appearance.

        Apply absolute values from the pristine model so styles do not accumulate.
        The source model/viewer is shared; baking calls must be serialized."""
        import mujoco

        env, u, pr = _scene_env(self.MAZE, size)
        m, d = u.model, u.data
        m.site_rgba[:] = pr["site_rgba"]
        m.geom_rgba[:] = pr["geom_rgba"]
        m.mat_rgba[:] = pr["mat_rgba"]

        wall = self._var(variation, "walls.color")
        tint = self._var(variation, "floor.tint")
        for g in range(m.ngeom):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("block_"):
                m.geom_rgba[g, :3] = wall
        gmat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
        if gmat >= 0:
            m.mat_rgba[gmat, :3] = tint

        # Detach the floor material via geom_matid, which scene updates reread.
        # Editing texture bytes alone would leave the context’s uploaded texture unchanged.
        m.geom_matid[:] = pr["geom_matid"]
        if int(self._var(variation, "rendering.decolor_floor")[0]):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            if gid >= 0:
                m.geom_matid[gid] = -1
                m.geom_rgba[gid, :3] = tint * np.array([60, 64, 72]) / 255.0

        viewer = u.point_env.mujoco_renderer._get_viewer("rgb_array")
        cam = viewer.cam
        cam.distance = pr["cam_distance"] * float(self._var(variation, "camera.distance_scale")[0])
        cam.elevation = float(self._var(variation, "camera.elevation")[0])
        cam.azimuth = float(self._var(variation, "camera.azimuth")[0])

        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        keep = m.site_rgba[:, 3].copy()
        m.site_rgba[:, 3] = 0.0

        plate = _official_frame(u)
        m.site_rgba[:, 3] = keep
        _official_frame(u)  # repopulate scn with the real appearance

        sc = _Scene()
        sc.size = size
        sc.cam = R3.Camera.from_mjv(viewer.scn.camera[0], viewer.scn.camera[1])
        sc.plate = torch.tensor(plate, device=self.device, dtype=torch.uint8)

        pid = 1 - int(u.target_site_id)
        sc.radius = float(m.site_size[pid][0])
        sc.scaling = float(u.maze.maze_size_scaling)
        sc.target_z = float(u.maze.maze_height / 2 * sc.scaling)
        sc.free_xy = np.stack(
            [np.asarray(p, np.float64) for p in (u.maze.unique_goal_locations or u.maze.combined_locations)]
        )

        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
        static = [g for g, n in enumerate(names) if n != "particle_geom"]
        sprims = R3.Primitives(
            kind=np.array([int(m.geom_type[g]) for g in static]),
            size=np.stack([m.geom_size[g] for g in static]),
            mat_rgb=np.stack([_mat_and_tex(m, g)[0] for g in static]),
            tex_rgb=np.stack([_mat_and_tex(m, g)[1] for g in static]),
            spec=np.full(len(static), MAT_SPECULAR),
        )
        sc.plate_t = R3.static_depth(
            sc.cam,
            sprims,
            np.stack([d.geom_xpos[g] for g in static]),
            np.stack([d.geom_xmat[g] for g in static]),
            size,
            size,
            self.device,
        )

        def prim(n):
            return R3.Primitives(
                kind=np.full(n, R3.SPHERE),
                size=np.tile([sc.radius, 0.0, 0.0], (n, 1)),
                mat_rgb=np.zeros((n, 3)),
                tex_rgb=np.ones((n, 3)),
                spec=np.full(n, MAT_SPECULAR),
            )

        sc.prims_ball, sc.prims_target, sc.prims_both = prim(1), prim(1), prim(2)

        L = int(viewer.scn.nlight)
        ldir = np.stack([-np.asarray(viewer.scn.lights[i].dir, np.float64) for i in range(L)])
        ldir = ldir / np.linalg.norm(ldir, axis=1, keepdims=True)
        sc.lights = R3.Lights(
            dirs=ldir,
            ambient=np.full(L, SHADE_AMBIENT),
            diffuse=np.full(L, SHADE_DIFFUSE),
            specular=np.full(L, SHADE_SPECULAR),
            exponent=SHADE_EXPONENT,
        )
        return sc

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
        """Render ``(..., 4)`` rows to uint8 frames, grouping by plate-affecting style."""
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = s.shape[0]
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        key = torch.cat([sty[p].reshape(B, -1) for p in self.PLATE_PATHS], 1).cpu().numpy()
        uniq, inv = np.unique(np.round(key.astype(np.float64), 6), axis=0, return_inverse=True)
        for gi in range(len(uniq)):
            idx = np.flatnonzero(inv == gi)
            over = {p: uniq[gi][o : o + w] for p, (o, w) in zip(self.PLATE_PATHS, _offsets(self.PLATE_PATHS, sty))}
            sc = self._scene(size, over)
            sub = torch.as_tensor(idx, device=self.device, dtype=torch.long)
            got = self._render_group(sc, s[sub], sty, sub, size, ss, buf if len(uniq) == 1 else None)
            if len(uniq) != 1:
                buf[sub] = got
        return buf.view(*lead, size, size, 3) if lead else buf

    def _render_group(self, sc: _Scene, s: torch.Tensor, sty, sub, size: int, ss: int, dst):
        """Cast ball and target separately; jointly cast rows where their depths interact."""
        b = s.shape[0]
        dev = self.device
        eye = torch.eye(3, device=dev).view(1, 1, 3, 3)
        scale = sty["agent.scale"][sub].to(torch.float32).reshape(-1)
        acol = sty["agent.color"][sub].to(torch.float32).view(b, 1, 3)
        tcol = sty["target.color"][sub].to(torch.float32).view(b, 1, 3)
        alpha = sty["target.alpha"][sub].to(torch.float32).reshape(b, 1)

        xb = torch.zeros(b, 1, 3, device=dev)
        xb[:, 0, :2] = s[:, 0:2]
        xt = torch.zeros(b, 1, 3, device=dev)
        xt[:, 0, :2] = s[:, 2:4]
        xt[:, 0, 2] = sc.target_z

        # Write ball-over-plate directly into the destination; the target blend edits it in place.
        base = R3.composite(
            sc.plate,
            sc.plate_t,
            sc.cam,
            sc.prims_ball,
            xb,
            eye.expand(b, 1, 3, 3).contiguous(),
            sc.lights,
            size,
            size,
            supersample=ss,
            tile=self.TILE,
            mat_rgb=acol,
            size_scale=scale,
            out=dst,
        )
        with_t = R3.composite(
            sc.plate,
            sc.plate_t,
            sc.cam,
            sc.prims_target,
            xt,
            eye.expand(b, 1, 3, 3).contiguous(),
            sc.lights,
            size,
            size,
            supersample=ss,
            tile=self.TILE,
            mat_rgb=tcol,
            size_scale=scale,
        )
        # Independent casts cannot resolve ball/target depth. Exclude overlapping
        # rows here and recast both primitives together below.
        occl = self._ball_occludes_target(sc, xb, xt, scale)
        mask = (with_t != sc.plate.view(1, size, size, 3)).any(-1) & ~occl.view(b, 1, 1)
        self._blend(base, with_t, mask, alpha)

        # Only overlapping rows need a joint cast; one tile covers both spheres there.
        j = occl.nonzero(as_tuple=True)[0]
        if j.numel():
            sel = base[j]
            joint = R3.composite(
                sc.plate,
                sc.plate_t,
                sc.cam,
                sc.prims_both,
                torch.cat([xb[j], xt[j]], 1),
                eye.expand(len(j), 2, 3, 3).contiguous(),
                sc.lights,
                size,
                size,
                supersample=ss,
                tile=self.TILE,
                mat_rgb=torch.cat([acol[j], tcol[j]], 1),
                size_scale=scale[j],
            )
            self._blend(sel, joint, (joint != sel).any(-1), alpha[j])
            base[j] = sel
        return base

    @staticmethod
    def _blend(base: torch.Tensor, over: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor) -> None:
        """Blend masked pixels in place with effective alpha ``2a - a²``.

        This models the source’s two transparent-face blends. Restrict float
        intermediates to the mask to avoid full-frame allocations."""
        who = mask.nonzero(as_tuple=True)
        if not who[0].numel():
            return
        a = alpha[who[0]]
        v = (2 * a - a * a) * over[who].float() + (1 - a) ** 2 * base[who].float()
        base[who] = v.round_().clamp_(0, 255).to(torch.uint8)

    def _ball_occludes_target(self, sc: _Scene, xb, xt, scale) -> torch.Tensor:
        """Select projected-sphere overlaps with the ball centre nearer the camera."""
        o = torch.tensor(sc.cam.pos, device=self.device, dtype=torch.float32).view(1, 3)
        db = xb[:, 0] - o
        dt = xt[:, 0] - o
        nb = db.norm(dim=-1).clamp_min(1e-9)
        nt = dt.norm(dim=-1).clamp_min(1e-9)
        cos = ((db * dt).sum(-1) / (nb * nt)).clamp(-1, 1)
        rb = (sc.radius * scale / nb).clamp(max=1.0)
        rt = (sc.radius * scale / nt).clamp(max=1.0)
        return (torch.arccos(cos) < torch.arcsin(rb) + torch.arcsin(rt)) & (nb < nt)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample ball and goal positions independently from the cached eligible cells.

        Add source-style uniform reset noise scaled by maze size."""
        rng = rng or self.rng
        sc = self._scene()
        cells = sc.free_xy
        pick = cells[rng.integers(0, len(cells), (batch, 2))]  # (batch, 2, 2)
        noise = rng.uniform(-POSITION_NOISE, POSITION_NOISE, (batch, 2, 2)) * sc.scaling
        return (pick + noise).reshape(batch, 4).astype(np.float32)

    @classmethod
    def reference_env(cls, size: int = 224):
        """Construct a fresh source viewer with the same pinned camera convention."""
        return _build_reference(cls.MAZE, size)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore ball and goal positions and render source frames with velocity zeroed."""
        import mujoco

        u = env.unwrapped
        m, d = u.model, u.data
        out = []
        for st in np.asarray(states, np.float64):
            d.qpos[:] = st[:2]
            d.qvel[:] = 0.0
            u.goal = st[2:4].copy()
            u.update_target_site_pos()
            mujoco.mj_forward(m, d)
            got = np.array(d.qpos[:2], np.float64)
            if np.abs(got - st[:2]).max() > 1e-9:
                raise AssertionError(
                    f"PointMaze reference did not reach the requested ball position: asked {st[:2]}, got {got}"
                )
            out.append(_official_frame(u))
        return np.stack(out)


def _make_classes():
    """Register one renderer per maze layout; reward variants share its geometry."""
    made = {}
    for maze in ("UMaze", "Open", "Medium", "Large"):
        cname = f"PointMaze{maze}"
        cls = register(
            type(
                cname,
                (PointMaze,),
                dict(
                    MAZE=maze,
                    domain=maze.lower(),
                    name=f"pointmaze-{maze.lower()}",
                    reference_id=f"PointMaze_{maze}-v3",
                    __doc__=(
                        f"gymnasium-robotics PointMaze_{maze}-v3 (D4RL maze2d-{maze.lower()}). State is "
                        f"the ball's (x, y) and the goal's (x, y); the Dense and Diverse variants of "
                        f"this id differ only in reward and goal sampling and share this renderer."
                    ),
                ),
            )
        )
        made[cname] = cls
    return made


globals().update(_make_classes())
