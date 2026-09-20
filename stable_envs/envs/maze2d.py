"""D4RL maze2d, as it survives today: gymnasium-robotics' PointMaze, four scenes, four numbers a frame.

Reference: ``gymnasium_robotics`` 1.4.2's ``PointMaze_{UMaze,Open,Medium,Large}-v3`` at
``render_mode='rgb_array'``. D4RL's own ``maze2d-*-v1`` is the same environment one generation back
and is not installable here (``d4rl`` needs ``mujoco_py``, which needs MuJoCo 2.1 binaries;
``import d4rl`` raises ``ModuleNotFoundError`` in this environment), so the maintained port is the
reference. The picture is the same: a green ball in a top-lit maze of brown boxes on a checkerboard
floor, with a translucent red ball marking the goal.

**The state is four numbers**: the ball's two slide joints and the goal's ``(x, y)``. Measured, not
asserted: reaching the same ``(qpos, goal)`` through a different history
changes **0 pixels**; scaling ``qvel`` by 100 changes **0 pixels**; a FRESH env instance driven to the
same state changes **0 pixels**. So 16 bytes of float32 stand against a 150,528-byte 224x224 frame —
**9,408x** — the same ratio OGBench's pointmaze reports, which is not a coincidence: it is the same
body under a different camera.

The goal is the difference, and it is the reason this env exists beside ``ogbench``'s pointmaze. OGBench
renders no goal marker, so its state is ``qpos`` alone; here the target is drawn, moves per episode, and
is therefore state. It is also **translucent**, which turned out to be the one genuinely new thing in
the family.

How MuJoCo composites the translucent target, measured rather than read
-----------------------------------------------------------------------

The obvious model — source-over, ``out = a*C + (1-a)*bg`` with ``a = 0.7`` — is **wrong by a mean of
18.7/255 and exact on 0% of target pixels**. Sweeping the site's alpha and fitting shows why: the pixel
is **quadratic in alpha** (linear fit max residual 31.5/255, quadratic 0.84/255), because MuJoCo draws a
transparent geom's two faces and blends it over the background **twice** with the same front-face
colour. So

    out = (2a - a^2) * C_front + (1 - a)^2 * background

which sums to 1 and is therefore the identity wherever the target is absent — the whole frame can be
blended without a mask. Checked on individual pixels: front (240,0,0) over background (49,73,98) at
a = 0.7 predicts (222.8, 6.6, 8.8) and the reference draws (222, 7, 9).

What that buys is a renderer made of two ordinary opaque casts: one of the ball over the baked plate and
one of the target over the same plate, combined by the formula above. Both are TILED — the moving
primitives cover 0.5% of the frame — which is where the throughput comes from; casting the two together
would need one window covering both, and at opposite ends of a maze that is the whole frame.

The one case two independent casts cannot express is the target being hidden BEHIND the ball, since
neither cast knows about the other's depth. It is detected analytically (do the two projected discs
overlap, and is the ball nearer?) and those environments are re-rendered with both primitives in one
untiled cast, which resolves depth correctly. The test is exact for spheres; the fallback is exact; and
it fires only when the ball is sitting on its own goal.

The plate
---------

Floor, walls and the gradient skybox are constant for a given (scene, appearance, size) and are baked
once through MuJoCo itself, exactly as ``ogbench`` does — so the static 99.5% of the frame inherits the
reference's own rasterisation rather than being re-derived. That matters more here than it does for
OGBench: the camera sits at elevation -45 rather than straight down, so the frame contains a
``builtin="gradient"`` skybox with ``mark="random"`` speckles and a 20x20 checkerboard receding to the
horizon — the exact scene content that makes DM Control a NO-GO for a hand-written 2-D path. Baking
turns it into a constant.

Two defects in the reference that only a second maze family would have found
---------------------------------------------------------------------------

Both are in ``gymnasium``'s MuJoCo renderer rather than in this code, both are invisible until you own
two implementations to compare, and both are recorded at the function that works around them:

  * **The free camera is a function of the env's construction history, not of the state.**
    ``BaseRender.__init__`` sets ``cam.lookat[i] = np.median(data.geom_xpos[:, i])`` and the viewer is
    built lazily on the first render, so the camera depends on where the ball was standing then. Two
    fresh ``PointMaze_Medium-v3`` instances at the same state came out **37.9% of pixels apart, max
    212/255**. See ``_build_reference``, which pins it.
  * **The viewer never re-binds its GL context.** ``make_context_current()`` is called once in
    ``__init__`` and ``render()`` goes straight to ``mjr_render``, so in a process that also holds a
    ``mujoco.Renderer`` (which binds on every call, as ``ogbench`` does) the gymnasium viewer draws
    into somebody else's context. These four envs passed the cross-process reproducibility gate alone
    and failed it in a run that also contained the OGBench mazes. See ``_official_frame``.

Parity, and where the residual lives
------------------------------------

Measured over 4 scenes x 32 states at 224 px against the official renderer: **99.49-99.77% of all pixels exact, mean |diff| 0.013-0.023/255**, and
**interior agreement 1.000000 on every scene**, so nothing is relaxed. The two spheres cover
0.25-0.57% of the frame; the plate carries the rest and is **99.997% bit-identical**. Rendering the
ball on top of its own goal — the state that exercises the occlusion fallback — is 99.67-99.82% exact.
``parity_note`` carries the moving-pixel breakdown.

Throughput on one H200 (PointMazeMedium, 224 px, supersample 2): **48,648 frames/s at batch 4,096**
against **2,293 f/s** for MuJoCo's own EGL renderer, i.e. **21x**. Two tiled casts and a masked blend;
the untiled form of the same renderer measured 7,985 f/s, and taking the whole batch down the
occlusion fallback measured the same, which is why both are per-row.

The 18 registered ``PointMaze_*`` ids collapse to these four scenes. ``Dense`` changes the reward,
``_Diverse_G`` / ``_Diverse_GR`` change which cells goals and resets are drawn from; none of the three
reaches the renderer, and the sampler here draws from every free cell, which is the union of them.
"""

from __future__ import annotations

import numpy as np
import torch

from .._mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import raycast3d as R3
from ..spaces import Box, Dict, Discrete

#: MuJoCo's own default; what a PointMaze frame is drawn with
MAT_SPECULAR = 0.5
#: ``point.xml``'s ``<headlight ambient=".4" diffuse=".8" specular="0.1">``, and the only light in the
#: model (``nlight = 0``, so ``mjv_updateScene`` adds the headlight and nothing else). Used as
#: measured: a grid search over ambient/diffuse/specular/exponent moved target-and-ball agreement by
#: 6 points and the nominal values are what the model declares, so they are not "fitted" here. The
#: residual is coverage, not shading — see ``parity_note``.
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
    """Everything a render needs that does not depend on the state."""

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
    """The official env at the requested frame size, with its free camera PINNED. Needs EGL.

    The pinning is not a convenience and it is the single hardest thing about this family.
    ``gymnasium.envs.mujoco.mujoco_rendering.BaseRender.__init__`` sets the free camera as

        for i in range(3): self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])

    and the viewer is constructed lazily, on the FIRST render call, from whatever ``data`` happens to
    hold at that instant. ``geom_xpos`` includes ``particle_geom``, which moves with the ball. So the
    camera of a PointMaze env is a function of **where the ball was standing the first time anybody
    rendered** — after which it is frozen for the life of the viewer.

    Measured: two freshly built ``PointMaze_Medium-v3`` instances, driven to the SAME state, disagree
    on **37.9% of pixels, max 212/255**, purely because their ``lookat`` came out ``(-0.25, 0.25,
    0.2)`` and ``(0.25, 0.25, 0.2)``. Moving one of 40 geoms shifts a median only when it crosses it,
    which is why UMaze, Open and Large agree at 0/255 and Medium does not — the defect is invisible
    on three quarters of the family.

    That makes "the official appearance" ill-defined unless the construction is pinned, so both the
    plate bake and the parity reference build their viewer at ``qpos = 0``: forward the model with the
    ball at the origin, then force the viewer into existence. Any convention would do; having one is
    what makes the comparison mean anything, and it is stated here rather than buried.
    """
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
    """One official env per (scene, size), kept forever.

    Never closed and never pooled, for the reason ``ogbench.py`` records at length: each one owns an
    EGL context, releasing one corrupts rendering in every other live context, and the corruption
    reads exactly like a state-sufficiency violation.
    """
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
    """One frame from the official viewer, with its GL context BOUND first.

    ``gymnasium.envs.mujoco.mujoco_rendering.OffScreenViewer`` calls ``make_context_current()`` exactly
    once, in ``__init__``, and never again — ``render()`` goes straight to ``mjr_render`` against
    whatever EGL context happens to be current. MuJoCo's own ``mujoco.Renderer`` (which ``ogbench``
    uses) binds on every call, so a process that holds both hands the gymnasium viewer someone else's
    context and it renders into it.

    That is not theoretical and it is not visible in isolation. The four PointMaze envs pass the
    cross-process reproducibility gate on their own and FAIL it as soon as the run also contains the
    OGBench mazes, because the pytest process and the render-worker process bind contexts in a
    different order. Binding here costs one EGL call per bake and makes the plate a function of the
    scene rather than of what else is loaded.
    """
    v = u.point_env.mujoco_renderer._get_viewer("rgb_array")
    v.make_context_current()
    v._set_mujoco_buffer()
    return u.point_env.mujoco_renderer.render("rgb_array").copy()


def _mat_and_tex(m, g):
    """(mat_rgb, tex_rgb) for geom ``g``, following MuJoCo's modulation order. As ``ogbench``."""
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
    """Base for the four scenes. ``MAZE`` is set by the generated subclasses."""

    family = "maze2d"

    MAZE: str = ""
    state_dim = 4
    state_names = STATE_NAMES
    #: Public state: the four columns are two positions in a plane, and saying so is
    #: the whole of this family's state structure. The packed row is unchanged -- ball then goal, the
    #: order ``STATE_NAMES`` has always named -- and every column is covered, so nothing is hidden,
    #: ``_hidden_state`` is never consulted and the frozen layout does not move. The ball's velocity
    #: is deliberately NOT here: the official env integrates it, this package renders from a pose,
    #: and a column the frame never reads would be a dimension the conformance suite reports as dead.
    state_layout = ((("ball", "position"), 2, 0), (("goal", "position"), 2, 2))
    default_size = 224
    #: The reference is multisampled at 4 samples; a hard silhouette is the wrong answer at the edge
    #: of a 12-pixel sphere. As ``ogbench``, and for the same reason.
    default_supersample = 2
    #: Screen-space window each moving primitive is cast in. Both are ~0.2% of the frame at the
    #: default camera, so 48 output pixels holds either with room for a scaled-up one.
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

    # ---- variation -------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose defaults are the official picture exactly.

        Colours are float Boxes in [0,1] rather than ``RGBBox`` for the reason ``ogbench`` gives:
        MuJoCo's material colours are floats and a uint8 channel cannot hold 0.7 — the nearest,
        179/255, shifts every wall by one level.

        ``target.alpha`` is a real knob rather than a constant because the whole compositing law is a
        function of it, and because a leak-free variant of this env (goal hidden from the pixels) is
        exactly ``alpha = 0``.
        """
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
                # the checkerboard is what lets a pixel agent read its own position off the floor; flat
                # slate is the leak-free condition. 0 is the official appearance.
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

    # ---- scene -----------------------------------------------------------

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
        """Hide the two sites, let MuJoCo draw everything else, and read the camera off its scene.

        Every mutation is an absolute assignment from the pristine snapshot, never an increment, so
        baking appearance B after appearance A gives the same model as baking B first — the camera
        distance is a MULTIPLIER and would otherwise compound.
        """
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

        # Detaching the floor GEOM from its material, not editing the texture's bytes. A GL context
        # uploads texture data when it is built and never re-reads it, so writing flat slate into
        # ``model.tex_rgb`` after the fact changes nothing at all — measured: the knob was inert and
        # ``check_variations_do_something`` said so. Clearing ``mat_texid`` is not enough either: the
        # material table is baked into the MjrContext. ``geom_matid`` IS read by ``mjv_updateScene``
        # on every frame, so clearing it takes effect immediately and leaves the geom at its own
        # ``geom_rgba`` — set here to slate, tinted by ``floor.tint`` so the two knobs compose.
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
        # ONE renderer per env, never closed. A texture edit needs the context rebuilt, so the
        # decolor_floor knob is applied before the first render of that key and re-applied
        # identically afterwards; the renderer is reused because the bytes it uploads are the same.
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

    # ---- rendering -------------------------------------------------------

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
        """``(...,4)`` -> ``(...,H,W,3)`` uint8. Style may differ per environment.

        The batch is grouped by the plate-affecting half of the style, as ``ogbench`` does, so
        randomising the ball's colour never splits the batch and randomising the camera splits it once
        per distinct camera.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
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
        """One plate's worth of environments: two tiled casts and the transparency blend."""
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

        # the ball over the plate, written straight into the caller's buffer: it is the base image and
        # the blend below only edits the handful of pixels the target covers
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
        # ``with_t`` is the target over the PLATE alone, so it differs from the plate exactly where the
        # target is drawn. The environments where the ball stands in front of the target are excluded
        # here and redone below, because two independent casts cannot resolve depth between them.
        occl = self._ball_occludes_target(sc, xb, xt, scale)
        mask = (with_t != sc.plate.view(1, size, size, 3)).any(-1) & ~occl.view(b, 1, 1)
        self._blend(base, with_t, mask, alpha)

        # Excluding the occluded environments per ROW rather than per BATCH is what makes the
        # fallback affordable: at B = 4096 independent (ball, goal) pairs about 0.5% of rows overlap,
        # and taking the whole batch down the untiled path for them measured 7,985 f/s against
        # 48,648 for the row-wise form. The joint cast is still TILED — the two spheres overlap on
        # screen by construction here, so one window holds both.
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
        """``base[mask] = (2a - a^2) * over[mask] + (1 - a)^2 * base[mask]``, in place.

        Masked rather than whole-frame. The formula is a convex combination and is the identity
        wherever ``over`` equals ``base``, so blending the entire frame would be correct — and it would
        also be the entire cost of the renderer: two float temporaries of ``(B, 224, 224, 3)`` are
        2.4 GB apiece at B = 4096.
        """
        who = mask.nonzero(as_tuple=True)
        if not who[0].numel():
            return
        a = alpha[who[0]]
        v = (2 * a - a * a) * over[who].float() + (1 - a) ** 2 * base[who].float()
        base[who] = v.round_().clamp_(0, 255).to(torch.uint8)

    def _ball_occludes_target(self, sc: _Scene, xb, xt, scale) -> torch.Tensor:
        """``(b,)`` bool: do the two projected discs overlap with the ball in front?

        Exact for spheres: two spheres overlap on screen iff the angle between their centre directions
        is less than the sum of their angular radii, and the near one is the one whose centre is
        closer to the eye. Conservative only in that it ignores the plate, which cannot make the
        target visible where the ball hides it.
        """
        o = torch.tensor(sc.cam.pos, device=self.device, dtype=torch.float32).view(1, 3)
        db = xb[:, 0] - o
        dt = xt[:, 0] - o
        nb = db.norm(dim=-1).clamp_min(1e-9)
        nt = dt.norm(dim=-1).clamp_min(1e-9)
        cos = ((db * dt).sum(-1) / (nb * nt)).clamp(-1, 1)
        rb = (sc.radius * scale / nb).clamp(max=1.0)
        rt = (sc.radius * scale / nt).clamp(max=1.0)
        return (torch.arccos(cos) < torch.arcsin(rb) + torch.arcsin(rt)) & (nb < nt)

    # ---- states ----------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """``MazeEnv.reset``'s own distribution: a free cell centre plus U(-0.25, 0.25) * scaling.

        Ball and goal are drawn independently from every free cell, which is the UNION of what the
        plain, ``_Diverse_G`` and ``_Diverse_GR`` variants draw from — they differ only in which cells
        are eligible, and that never reaches the renderer.
        """
        rng = rng or self.rng
        sc = self._scene()
        cells = sc.free_xy
        pick = cells[rng.integers(0, len(cells), (batch, 2))]  # (batch, 2, 2)
        noise = rng.uniform(-POSITION_NOISE, POSITION_NOISE, (batch, 2, 2)) * sc.scaling
        return (pick + noise).reshape(batch, 4).astype(np.float32)

    # ---- reference bridge -------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 224):
        """A FRESH official env. Not cached and never closed — see ``ogbench.py``'s note."""
        return _build_reference(cls.MAZE, size)

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for the given (ball, goal). ``qvel`` is zero and provably irrelevant."""
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
    """One registered class per maze layout.

    Four, not eighteen. ``Dense`` changes the reward and ``_Diverse_G`` / ``_Diverse_GR`` change which
    cells the goal and the reset are drawn from; neither reaches the renderer, so registering them
    would put fourteen identical renderers into a conformance run whose reference leg costs a MuJoCo
    render apiece.
    """
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
