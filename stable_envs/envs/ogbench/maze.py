from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.mujoco_gl import configure_mujoco_gl
from ..._execution.protocol import StepPipeline, step_pipeline

configure_mujoco_gl()

from ..._adapters.ogbench.execution import make_maze_adapter
from ..._adapters.ogbench.maze import OGB_MAZE_FIELDS, OGBMazeAdapter
from ...env import Env
from ...registry import register
from ...render import _resources as _RES
from ...render import raycast3d as R3
from ...spaces import Box, Dict, Discrete

# Shading totals fitted on point-agent interiors, not baked floor pixels.
SHADE_AMBIENT, SHADE_DIFFUSE, SHADE_SPECULAR, SHADE_EXPONENT = 0.078438, 1.416189, 0.230554, 45.0
# Split fitted totals by source headlight/global proportions when directions diverge.
_NOMINAL = dict(amb=(0.1, 0.0), dif=(0.4, 1.0), spec=(0.5, 0.1))
MAT_SPECULAR = 0.5  # every material in every OGBench maze scene

#: locomotion body -> (state_dim, state_names). Verified against the compiled model at construction.
_FREE = ("root_x", "root_y", "root_z", "root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z")
_ANT_JOINTS = ("hip_1", "ankle_1", "hip_2", "ankle_2", "hip_3", "ankle_3", "hip_4", "ankle_4")
_HUMANOID_JOINTS = (
    "abdomen_z",
    "abdomen_y",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "right_ankle_y",
    "right_ankle_x",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "left_ankle_y",
    "left_ankle_x",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
)
BODY_STATE = {
    "point": ("x", "y"),
    "ant": _FREE + _ANT_JOINTS,
    "humanoid": _FREE + _HUMANOID_JOINTS,
}

#: the 25 release cells: (loco, maze, dataset). Only (loco, maze) reaches the renderer.
CELLS = (
    [("point", m, d) for m in ("medium", "large", "giant", "teleport") for d in ("navigate", "stitch")]
    + [("ant", m, d) for m in ("medium", "large") for d in ("navigate", "stitch", "explore")]
    + [("ant", "giant", d) for d in ("navigate", "stitch")]
    + [("ant", "teleport", d) for d in ("navigate", "stitch", "explore")]
    + [("humanoid", m, d) for m in ("medium", "large", "giant") for d in ("navigate", "stitch")]
)

# Retain recently used plates: rebaking can change a few source GL rounding bits.
_SCENE_CACHE_MAX = 256  # a plate is 150 kB and a depth buffer 200 kB; re-baking costs far more
_SCENE_CACHE = _RES.BoundedCache(_SCENE_CACHE_MAX, name="ogbench.maze")


def _offsets(paths, sty):
    """Offsets and widths within a key formed by concatenating the selected style leaves."""
    out, off = [], 0
    for p in paths:
        w = int(sty.layout[p][1])
        out.append((off, w))
        off += w
    return out


class _Scene:
    """Cached appearance resources; model and data remain shared mutable objects."""

    __slots__ = (
        "plate",
        "plate_t",
        "cam",
        "model",
        "data",
        "agent_ids",
        "lights",
        "size",
        "geom_local_pos",
        "maze_map",
        "maze_unit",
        "offset",
        "agent_base",
        "agent_is_self",
        "prims_base",
    )


def _mat_and_tex(m, g):
    """Resolve geom/material RGB and mean texture RGB in MuJoCo modulation order."""
    mid = int(m.geom_matid[g])
    mat = np.asarray(m.geom_rgba[g], np.float64)[:3]
    tex = np.ones(3)
    if mid >= 0:
        if np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):  # compiler default -> material colour
            mat = np.asarray(m.mat_rgba[mid], np.float64)[:3]
        tid = m.mat_texid[mid]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        if tid >= 0:
            attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
            buf = getattr(m, attr)
            a0 = int(m.tex_adr[tid])
            h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
            reg = np.asarray(buf[a0 : a0 + 3 * h * w], np.float64).reshape(h, w, 3) / 255.0
            u = np.unique(reg.reshape(-1, 3), axis=0)
            tex = u[0] if len(u) == 1 else reg.reshape(-1, 3).mean(0)
    return mat, tex


def _build_reference(loco: str, maze: str, size: int, ob_type: str = "pixels"):
    """Construct a source maze with the requested observation and texture mode."""
    import gymnasium
    import ogbench  # noqa: F401

    env = gymnasium.make(
        f"{loco}maze-{maze}-v0", render_mode="rgb_array", ob_type=ob_type, width=int(size), height=int(size)
    )
    env.reset(options={"task_id": 1})
    return env


# One source environment per (body, maze, size, observation mode), with pristine
# values for every appearance mutation; styles share its GL resources.
_ENV_CACHE: dict = {}


def _scene_env(loco: str, maze: str, size: int, ob_type: str = "pixels"):
    key = (loco, maze, int(size), ob_type)
    hit = _ENV_CACHE.get(key)
    if hit is not None:
        return hit
    import mujoco

    env = _build_reference(loco, maze, size, ob_type)
    u = env.unwrapped
    m = u.model
    m.vis.quality.offsamples = 4  # MuJoCo's default; what OGBench datasets hold
    tex = m.tex("grid")
    a0, th, tw = int(tex.adr[0]), int(tex.height[0]), int(tex.width[0])
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    pristine = dict(
        tex=np.array(getattr(m, attr)[a0 : a0 + 3 * th * tw], copy=True),
        tex_adr=a0,
        tex_n=3 * th * tw,
        cam_distance=float(u.custom_camera.distance),
        cam_elevation=float(u.custom_camera.elevation),
        mat_rgba=np.array(m.mat_rgba, copy=True),
        geom_rgba=np.array(m.geom_rgba, copy=True),
        # Texture bytes upload at context creation, so cache a renderer per texture mode.
        # Keep contexts alive to avoid corrupting other live renderers during teardown.
        renderers={},
        agent=[
            g
            for g in range(m.ngeom)
            if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(("block_", "teleport"))
            and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) != "floor"
        ],
    )
    _ENV_CACHE[key] = (env, u, pristine)
    return _ENV_CACHE[key]


class OGBenchMaze(Env):
    """Render OGBench maze bodies over cached source backgrounds.

    Pixel-mode rows contain qpos; state-observation mode also carries the goal.
    Only PointMazeLargeNavigate adds the full-State CPU simulator adapter.
    The remaining registrations provide pose rendering without stepping."""

    family = "ogbench"

    LOCO: str = ""
    MAZE: str = ""
    default_size = 224
    # Approximate source four-sample edges by casting at double resolution.
    default_supersample = 2
    # Moving-primitive cast window; oversized projections fall back to the full frame.
    TILE = 32
    # The flat-interior metric excludes most small shaded agent pixels.
    parity_interior_min = 1.0

    parity_note = (
        "AGENT-PIXEL PARITY (supersample 2, against MuJoCo at its default offsamples=4, 25 cells x 64 "
        "states). Every disagreement is inside the agent's own silhouette: the baked plate is exact by "
        "construction, non-agent pixels are 99.88-99.99% bit-identical, and that residual is the "
        "reference's multisampling bleeding agent colour into neighbouring wall/floor pixels. On "
        "agent-interior pixels (eroded by one, so coverage is not in play) the shader reaches 48-84% "
        "within 1/255 for pointmaze (mean 1.2-2.5/255), 0-51% for antmaze and 0-3% for humanoidmaze. "
        "The legged numbers are not a shading result: at the official whole-maze camera the ant is "
        "12-70 px and the humanoid 4-21 px of 50176, and eroding leaves as few as 1 interior pixel, so "
        "their agreement is a statement about a silhouette under the reference's coverage rule. "
        "Against a NON-multisampled reference the same shader reaches 92.4% within 1/255 over 13,069 "
        "pooled agent-interior pixels (mean 0.88/255), which localises most of the remaining gap in "
        "MSAA sample positions -- GPU-specific, not recoverable by fitting -- rather than in shading. "
        "Three mechanisms were tried and are recorded so they are not retried: the fixed-function "
        "composition (FIXED - specular is additive and white and the texture modulates the whole "
        "primary colour, which took wall tops from 13/255 worst to 2/255 and made vertical wall faces "
        "bit-exact); Gouraud interpolation over MuJoCo's sphere/capsule tessellation (REFUTED - ten "
        "(slices, stacks, phase, diagonal) variants all scored 0.0-0.2% exact against 1.6% for plain "
        "per-pixel, which is also what sub-pixel triangles predict); and a mis-shaped specular lobe "
        "(REFUTED - a 24-knot piecewise-linear lobe fitted under four light/specular geometries, "
        "headlight directional or point crossed with Blinn or Phong, could not explain floor and "
        "agent with one curve)."
    )

    # Pose layout by body: point slides, or free-joint pose followed by hinges.
    # The simulator mixin supplies continuation fields separately.
    STATE_LAYOUT = {
        "point": ((("root", "position"), 2, 0),),
        "ant": ((("root", "position"), 3, 0), (("root", "orientation"), 4, 3), (("joints",), 8, 7)),
        "humanoid": ((("root", "position"), 3, 0), (("root", "orientation"), 4, 3), (("joints",), 21, 7)),
    }

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        """Derive render layout from the body and check it against the compiled model."""
        self.state_dim = len(BODY_STATE[self.LOCO])
        self.state_names = BODY_STATE[self.LOCO]
        if getattr(self, "RENDER_OB_TYPE", "pixels") == "states":
            self.state_dim += 3
            self.state_names = (*self.state_names, "goal_x", "goal_y", "goal_z")
        self.state_layout = self.STATE_LAYOUT[self.LOCO]
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        sc = self._scene()
        assert sc.model.nq == len(BODY_STATE[self.LOCO]), (
            f"{type(self).__name__}: model has nq={sc.model.nq} but expected {len(BODY_STATE[self.LOCO])}"
        )
        self._check_names(sc)
        if self.LOCO == "point":
            self._check_point_fk(sc)

    def _check_names(self, sc) -> None:
        """Verify declared joint names and order against the compiled qpos layout."""
        import mujoco

        m = sc.model
        got = []
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                got += list(_FREE)
            elif int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                got.append(nm)
            else:
                got.append(nm)
        if self.LOCO == "point":
            got = list(BODY_STATE["point"])  # slide joints are named ballx/bally in the XML
        assert tuple(got) == tuple(self.state_names[: len(got)]), (
            f"{type(self).__name__}: state_names {self.state_names} do not match the model's joint layout {tuple(got)}"
        )

    def _check_point_fk(self, sc) -> None:
        """Verify the point body’s translation shortcut against MuJoCo kinematics."""
        import mujoco

        m, d = sc.model, sc.data
        rng = np.random.default_rng(0)
        for _ in range(4):
            q = rng.uniform(-4, 20, m.nq)
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            # States mode also draws the world-body goal. It is placed from
            # State.task.goal_pos, not translated with the moving agent.
            moving = np.array([m.geom_bodyid[g] != 0 for g in sc.agent_ids])
            want = d.geom_xpos[np.asarray(sc.agent_ids)[moving]]
            got = sc.geom_local_pos[moving] + np.concatenate([q[:2], [0.0]])
            assert np.abs(want - got).max() < 1e-9, (
                f"pointmaze FK shortcut is wrong by {np.abs(want - got).max():.3e}; "
                f"the sphere is not simply translated by qpos"
            )


    def make_variation_space(self) -> Dict:
        """Expose body, floor, wall and camera appearance with source float colour defaults.

        Keep RGB in [0, 1]; uint8 quantization changes MuJoCo material shading."""
        agent_rgb = [0.7, 0.5, 0.3] if self.LOCO in ("point", "ant") else [0.7, 0.5, 0.3]
        return Dict(
            {
                "agent": Dict(
                    {
                        "color": Box(low=0.0, high=1.0, shape=(3,), init_value=agent_rgb),

                        "scale": Box(low=0.25, high=6.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "floor": Dict({"tint": Box(low=0.0, high=1.0, shape=(3,), init_value=[1.0, 1.0, 1.0])}),
                "walls": Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=[0.6, 0.6, 0.6])}),
                "camera": Dict(
                    {
                        "distance_scale": Box(low=0.6, high=1.8, shape=(1,), init_value=[1.0]),
                        "elevation": Box(low=-90.0, high=-35.0, shape=(1,), init_value=[-90.0]),
                    }
                ),
                # Pixel-mode floor encodes xy in its texture; decolor_floor replaces it with slate.
                "rendering": Dict({"decolor_floor": Discrete(2, init_value=0)}),
            }
        )

    #: Style leaves that change the STATIC scene and therefore the baked plate. Everything else is a
    #: property of the agent's primitives and is applied per environment inside the ray-cast.
    PLATE_PATHS = ("floor.tint", "walls.color", "camera.distance_scale", "camera.elevation", "rendering.decolor_floor")

    def _var(self, variation, path):
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)


    def _plate_key(self, size, variation):
        return (
            self.LOCO,
            self.MAZE,
            getattr(self, "RENDER_OB_TYPE", "pixels"),
            int(size),
            tuple(np.round(self._var(variation, "floor.tint"), 6)),
            tuple(np.round(self._var(variation, "walls.color"), 6)),
            float(np.round(self._var(variation, "camera.distance_scale")[0], 6)),
            float(np.round(self._var(variation, "camera.elevation")[0], 6)),
            int(self._var(variation, "rendering.decolor_floor")[0]),
            str(self.device),
        )

    def _scene(self, size=None, variation=None) -> _Scene:
        size = int(size or self.size)
        key = self._plate_key(size, variation)
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            return hit
        return _SCENE_CACHE.put(key, self._bake(size, variation))

    def _bake(self, size: int, variation) -> _Scene:
        """Bake a plate after applying absolute style values to the pristine model.

        Reset pose before baking. The model and GL resources are shared, so baking
        and kinematics calls must be serialized."""
        import mujoco

        env, u, pr = _scene_env(self.LOCO, self.MAZE, size, getattr(self, "RENDER_OB_TYPE", "pixels"))
        m, d = u.model, u.data
        attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"

        tint = self._var(variation, "floor.tint")
        wall = self._var(variation, "walls.color")
        m.mat_rgba[:] = pr["mat_rgba"]
        m.geom_rgba[:] = pr["geom_rgba"]
        gmat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "grid")
        wmat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "wall")
        if gmat >= 0:
            m.mat_rgba[gmat, :3] = tint
        if wmat >= 0:
            m.mat_rgba[wmat, :3] = wall
        for g in range(m.ngeom):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("block_"):
                m.geom_rgba[g, :3] = wall
        buf = getattr(m, attr)
        if int(self._var(variation, "rendering.decolor_floor")[0]):
            buf[pr["tex_adr"] : pr["tex_adr"] + pr["tex_n"]] = np.tile(
                np.array([60, 64, 72], buf.dtype), pr["tex_n"] // 3
            )
        else:
            buf[pr["tex_adr"] : pr["tex_adr"] + pr["tex_n"]] = pr["tex"]

        cam = u.custom_camera
        cam.distance = pr["cam_distance"] * float(self._var(variation, "camera.distance_scale")[0])
        cam.elevation = float(self._var(variation, "camera.elevation")[0])

        agent = pr["agent"]
        static = [g for g in range(m.ngeom) if g not in agent]

        # The shared data retains the last rendered pose. Reset it before baking: even
        # a hidden agent can affect rasterization, making cached plates depend on history.
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        keep = m.geom_rgba[agent].copy()
        m.geom_rgba[agent, 3] = 0.0  # hide the agent, bake what is left
        rkey = int(self._var(variation, "rendering.decolor_floor")[0])
        r = pr["renderers"].get(rkey)
        if r is None:
            # built AFTER the texture write for this key, so it uploads the right floor; every later
            # bake with the same key writes the same bytes, so reuse is exact
            r = pr["renderers"][rkey] = mujoco.Renderer(m, width=size, height=size)
        r.update_scene(d, camera=cam)
        plate = r.render().copy()
        m.geom_rgba[agent] = keep
        r.update_scene(d, camera=cam)
        camera = R3.Camera.from_mjv(r.scene.camera[0], r.scene.camera[1])
        nlight = int(r.scene.nlight)
        ldir = np.stack([-np.asarray(r.scene.lights[i].dir, np.float64) for i in range(nlight)])

        sc = _Scene()
        sc.size = size
        sc.model, sc.data, sc.agent_ids = m, d, agent
        sc.maze_map = np.asarray(u.maze_map)
        sc.maze_unit, sc.offset = float(u._maze_unit), (float(u._offset_x), float(u._offset_y))
        sc.cam = camera
        sc.plate = torch.tensor(plate, device=self.device, dtype=torch.uint8)
        sprims = R3.Primitives(
            kind=np.array([int(m.geom_type[g]) for g in static]),
            size=np.stack([m.geom_size[g] for g in static]),
            mat_rgb=np.stack([_mat_and_tex(m, g)[0] for g in static]),
            tex_rgb=np.stack([_mat_and_tex(m, g)[1] for g in static]),
            spec=np.full(len(static), MAT_SPECULAR),
        )
        sc.plate_t = R3.static_depth(
            camera,
            sprims,
            np.stack([d.geom_xpos[g] for g in static]),
            np.stack([d.geom_xmat[g] for g in static]),
            size,
            size,
            self.device,
        )
        sc.geom_local_pos = np.stack([m.geom_pos[g] for g in agent])
        # Keep the ant’s white leg geoms separate; they make its heading visible.
        selfmat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "self")
        sc.agent_base = np.stack([_mat_and_tex(m, g)[0] for g in agent])
        sc.agent_is_self = np.array([int(m.geom_matid[g]) == selfmat for g in agent])
        sc.prims_base = R3.Primitives(
            kind=np.array([int(m.geom_type[g]) for g in agent]),
            size=np.stack([m.geom_size[g] for g in agent]),
            mat_rgb=sc.agent_base,
            tex_rgb=np.ones((len(agent), 3)),
            spec=np.full(len(agent), MAT_SPECULAR),
        )
        # Pixel mode removes the torso light and zeros global ambient. States
        # mode retains them, so its light table must come from the actual scene.
        L = nlight
        dirs = ldir / np.linalg.norm(ldir, axis=1, keepdims=True)
        ka = SHADE_AMBIENT / max(sum(_NOMINAL["amb"]), 1e-9)
        kd = SHADE_DIFFUSE / max(sum(_NOMINAL["dif"]), 1e-9)
        ks = SHADE_SPECULAR / max(sum(_NOMINAL["spec"]) * MAT_SPECULAR, 1e-9)
        sc.lights = R3.Lights(
            dirs=dirs,
            ambient=np.array(_NOMINAL["amb"][:L]) * ka,
            diffuse=np.array(_NOMINAL["dif"][:L]) * kd,
            specular=np.array(_NOMINAL["spec"][:L]) * ks,
            exponent=SHADE_EXPONENT,
        )
        if getattr(self, "RENDER_OB_TYPE", "pixels") == "states":
            sc.lights = R3.Lights(
                dirs=dirs,
                ambient=np.array([np.mean(r.scene.lights[i].ambient) for i in range(L)]),
                diffuse=np.array([np.mean(r.scene.lights[i].diffuse) for i in range(L)]),
                specular=np.array([np.mean(r.scene.lights[i].specular) for i in range(L)]),
                exponent=float(m.mat_shininess[selfmat]) * 128,
            )
        return sc


    def _fk(self, states: torch.Tensor, sc: _Scene):
        """Return agent geom positions ``(B, P, 3)`` and rotations ``(B, P, 3, 3)``.

        Point-body translation stays on device. Legged bodies use serial CPU
        MuJoCo kinematics, gathering into preallocated arrays before one batch transfer."""
        import mujoco

        B = states.shape[0]
        P = len(sc.agent_ids)
        if self.LOCO == "point":
            off = torch.tensor(sc.geom_local_pos, device=self.device, dtype=torch.float32)
            pos = torch.zeros(B, P, 3, device=self.device)
            pos[:, :, :2] = states[:, None, :2]
            pos = pos + off.unsqueeze(0)
            if getattr(self, "RENDER_OB_TYPE", "pixels") == "states":
                target = sc.model.geom("target").id
                pos[:, sc.agent_ids.index(target), :] = states[:, 2:5]
            eye = torch.eye(3, device=self.device).view(1, 1, 3, 3).expand(B, P, 3, 3)
            return pos, eye.contiguous()
        q = states.detach().cpu().numpy().astype(np.float64)
        xp = np.empty((B, P, 3))
        xm = np.empty((B, P, 9))
        m, d = sc.model, sc.data
        ids = np.asarray(sc.agent_ids, np.intp)
        gp, gm, qpos = d.geom_xpos, d.geom_xmat, d.qpos
        take = np.take
        for i in range(B):
            qpos[:] = q[i]
            mujoco.mj_kinematics(m, d)
            take(gp, ids, axis=0, out=xp[i])
            take(gm, ids, axis=0, out=xm[i])
        return (
            torch.tensor(xp, device=self.device, dtype=torch.float32),
            torch.tensor(xm.reshape(B, P, 3, 3), device=self.device, dtype=torch.float32),
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
        """Render compact rows to uint8 frames, preserving leading batch dimensions.

        Group only by plate-affecting styles; body colour and scale remain per row."""
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

        plate_key = torch.cat([sty[p].reshape(B, -1) for p in self.PLATE_PATHS], 1).cpu().numpy()
        uniq, inv = np.unique(np.round(plate_key.astype(np.float64), 6), axis=0, return_inverse=True)
        for gi in range(len(uniq)):
            idx = np.flatnonzero(inv == gi)
            over = {p: uniq[gi][o : o + w] for p, (o, w) in zip(self.PLATE_PATHS, _offsets(self.PLATE_PATHS, sty))}
            sc = self._scene(size, over)
            sub = torch.as_tensor(idx, device=self.device, dtype=torch.long)
            mat, scale = self._agent_appearance(sc, sty, sub)
            xpos, xmat = self._fk(s[sub], sc)
            # Advanced indexing makes buf[sub] a copy. With multiple groups, render to a
            # temporary and assign back explicitly; one group can write directly into buf.
            dst = buf if len(uniq) == 1 else None
            got = R3.composite(
                sc.plate,
                sc.plate_t,
                sc.cam,
                sc.prims_base,
                xpos,
                xmat,
                sc.lights,
                size,
                size,
                supersample=ss,
                out=dst,
                tile=self.TILE,
                mat_rgb=mat,
                size_scale=scale,
            )
            if dst is None:
                buf[sub] = got
        return buf.view(*lead, size, size, 3) if lead else buf

    def _agent_appearance(self, sc: _Scene, sty, sub):
        """Return per-geom colours and per-row scale, preserving non-self materials.

        The ant’s white leg geoms distinguish its heading and must not be recoloured."""
        base = torch.as_tensor(sc.agent_base, device=self.device, dtype=torch.float32)
        is_self = torch.as_tensor(sc.agent_is_self, device=self.device)
        acol = sty["agent.color"][sub].to(torch.float32)  # (b,3)
        mat = base.unsqueeze(0).expand(len(sub), *base.shape).clone()
        mat[:, is_self] = acol.unsqueeze(1).expand(-1, int(is_self.sum()), 3)
        return mat, sty["agent.scale"][sub].to(torch.float32).reshape(-1)


    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample synthetic poses in free cells, with torso and joint noise for legged bodies.

        Varying quaternion orientation also prevents fixed-offset probes from
        normalizing straight back to the identity. These are not source rollout states."""
        rng = rng or self.rng
        sc = self._scene()
        cells = np.argwhere(sc.maze_map == 0)
        pick = cells[rng.integers(0, len(cells), batch)]
        xy = np.stack([pick[:, 1] * sc.maze_unit - sc.offset[0], pick[:, 0] * sc.maze_unit - sc.offset[1]], 1).astype(
            np.float64
        )
        xy += rng.uniform(-1.0, 1.0, xy.shape)
        q = np.tile(np.asarray(sc.model.qpos0, np.float64), (batch, 1))
        q[:, :2] = xy
        if sc.model.nq > 2:
            yaw = rng.uniform(-np.pi, np.pi, batch)
            rp = rng.uniform(-0.15, 0.15, (batch, 2))
            cw, sw = np.cos(yaw / 2), np.sin(yaw / 2)
            q[:, 3] = cw
            q[:, 4] = rp[:, 0] * 0.5
            q[:, 5] = rp[:, 1] * 0.5
            q[:, 6] = sw
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
            q[:, 7:] = q[:, 7:] + rng.uniform(-0.35, 0.35, (batch, sc.model.nq - 7))
        if getattr(self, "RENDER_OB_TYPE", "pixels") == "states":
            goals = np.tile(sc.model.geom("target").pos, (batch, 1))
            goal_cells = cells[rng.integers(0, len(cells), batch)]
            goals[:, 0] = goal_cells[:, 1] * sc.maze_unit - sc.offset[0]
            goals[:, 1] = goal_cells[:, 0] * sc.maze_unit - sc.offset[1]
            q = np.concatenate([q, goals], axis=-1)
        return q.astype(np.float32)


    @classmethod
    def reference_env(cls, size: int = 224):
        """Construct an independent source environment for history comparisons.

        Keep its GL context alive while other renderers are in use."""
        return _build_reference(cls.LOCO, cls.MAZE, size, getattr(cls, "RENDER_OB_TYPE", "pixels"))

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore qpos and, in state-observation mode, the goal before source rendering."""
        u = env.unwrapped
        nv = u.model.nv
        out = []
        for st in np.asarray(states, np.float64):
            if getattr(cls, "RENDER_OB_TYPE", "pixels") == "states":
                u.model.geom("target").pos[:] = st[2:5]
                st = st[:2]
            u.set_state(st, np.zeros(nv))
            out.append(u.render().copy())
        return np.stack(out)


class _HistoricalDefaultSemantics:
    """Compose full-State CPU simulation for the registered point/large default.

    The source uses state observations, a checker floor and a visible goal.
    CUDA selects output placement; simulation remains on CPU."""

    # Pixel observations replace the XML checker with a position-encoding
    # gradient. The registered simulator uses states, so its renderer must too.
    RENDER_OB_TYPE = "states"
    action_dim = 2
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulator = make_maze_adapter(self.B, self.device, self._seed, OGBMazeAdapter)

    def _layout(self):
        return tuple(
            (key, width, {("root", "position"): 0, ("task", "goal_pos"): 2}.get(key))
            for key, (width, _) in OGB_MAZE_FIELDS.items()
        )

    def _layout_dtypes(self):
        return {key: dtype for key, (_, dtype) in OGB_MAZE_FIELDS.items()}

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full OGBMaze State")
        return torch.cat([state["root", "position"], state["task", "goal_pos"]], dim=-1).to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, 5)
        return TensorDict(
            {("root", "position"): rows[:, :2].clone(), ("task", "goal_pos"): rows[:, 2:5].clone()},
            [len(rows)],
            device=self.device,
        )

    def reset(self, seed=None):
        self._check_open("reset")
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active)

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active, capture_state=False)

    def _step_pipeline(self) -> StepPipeline | None:
        owner = self._simulator
        return step_pipeline(owner)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()


#: ``__module__`` is pinned to the package so ``stable_envs.envs.ogbench.<Cell>`` still resolves.
#: Family identity is the explicit ``family = "ogbench"`` on :class:`OGBenchMaze`, not this path.
_MODULE = "stable_envs.envs.ogbench"


def _make_classes():
    """Register dataset-specific names, adding simulation only for point/large/navigate."""
    made = {}
    aliases = {}
    for loco, maze, dataset in CELLS:
        old = f"{loco.capitalize()}maze{maze.capitalize()}{dataset.capitalize()}"
        cname = f"{loco.capitalize()}Maze{maze.capitalize()}{dataset.capitalize()}"
        bases = (
            (_HistoricalDefaultSemantics, OGBenchMaze)
            if (loco, maze, dataset) == ("point", "large", "navigate")
            else (OGBenchMaze,)
        )
        cls = register(
            type(
                cname,
                bases,
                dict(
                    LOCO=loco,
                    MAZE=maze,
                    __module__=_MODULE,
                    domain=f"{loco}maze",
                    name=f"{loco}maze-{maze}-{dataset}",
                    reference_id=f"ogbench/{loco}maze-{maze}-v0",
                    state_dim=len(BODY_STATE[loco]),
                    state_names=BODY_STATE[loco],
                    __doc__=(
                        f"OGBench {loco}maze-{maze}-{dataset}. State is qpos "
                        f"({len(BODY_STATE[loco])} floats); the dataset suffix selects a task and does not "
                        f"reach the renderer, so it shares a baked plate with the other "
                        f"{loco}maze-{maze} cells."
                    ),
                ),
            )
        )
        made[cname] = cls
        aliases[old] = cname
    return made, aliases


#: ``{name: class}`` for every registered cell this module declares, so the package ``__init__``
#: can re-export them by NAME without transcribing the naming rule a second time.
CELL_CLASSES, CELL_ALIASES = _make_classes()
globals().update(CELL_CLASSES)
