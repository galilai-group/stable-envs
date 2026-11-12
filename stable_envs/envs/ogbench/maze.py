from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ... import tuning
from ..._core.mujoco_gl import configure_mujoco_gl, mujoco_renderer
from ..._core.plate_texels import quantise
from ..._core.stepping import _stack_states
from ..._execution.protocol import StepPipeline, step_pipeline

configure_mujoco_gl()

from ...env import Env
from ...registry import register
from ...render import _resources as _RES
from ...render import raycast3d as R3
from ...render.primitives import dot3
from ...spaces import Box, Dict, Discrete
from .simulation.execution import make_maze_adapter, maze_backend
from .simulation.maze import BODIES, maze_fields

# Shading totals fitted on point-agent interiors, not baked floor pixels.
SHADE_AMBIENT, SHADE_DIFFUSE, SHADE_SPECULAR, SHADE_EXPONENT = 0.078438, 1.416189, 0.230554, 45.0
# Split fitted totals by source headlight/global proportions when directions diverge.
_NOMINAL = dict(amb=(0.1, 0.0), dif=(0.4, 1.0), spec=(0.5, 0.1))
MAT_SPECULAR = 0.5  # every material in every OGBench maze scene
#: The XML checker texture's rgb1/rgb2, shared by the three body assets.
CHECKER_RGB = ((0.08, 0.11, 0.16), (0.15, 0.18, 0.25))

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
        "passes",
        "goal",
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


def _spec_shin(m, g):
    """Material specular and GL shininess exponent; MuJoCo's defaults without a material."""
    mid = int(m.geom_matid[g])
    if mid < 0:
        return 0.5, 64.0
    return float(m.mat_specular[mid]), float(m.mat_shininess[mid]) * 128.0


def _light_pass(n, v, terms, c, tex, ms, shin):
    """One framebuffer pass of GL fixed-function lighting, rounded to bytes.

    ``terms`` holds (unit vector to light, attenuation*spot gain, (ambient, diffuse,
    specular)) per enabled light. Lights sum and clamp before the texture modulates.
    """
    col = torch.zeros_like(c)
    for to_light, gain, (amb, dif, spec) in terms:
        ndl = dot3(n, to_light).clamp(min=0)
        h = to_light + v
        h = h / torch.sqrt(dot3(h, h)).clamp(min=1e-12).unsqueeze(-1)
        hl = torch.where(ndl > 0, dot3(n, h).clamp(min=0) ** shin, torch.zeros_like(ndl))
        lit = amb * c + dif * c * ndl.unsqueeze(-1) + spec * (ms * hl).unsqueeze(-1)
        col = col + lit * gain.unsqueeze(-1)
    return torch.round(torch.clamp(col, 0, 1) * tex * 255.0)


_SHADOW_EPS = 1e-3  # shadow-ray origin offset along the surface normal
_BOX = 6  # mjGEOM_BOX
# World-unit margin of the wall cull, far above the rounding error of a box test.
_CULL_SLACK = 0.05
# Centre of the padding box, far beyond every shadow ray.
_FAR = 1.0e6
_CULL_WALLS = True
# Composite the goal and the body in separate tiles when their screen bounds do not meet.
_SPLIT_MOVERS = True

# Rows per lit pass. On CPU, each shadow ray tests the walls that some ray of its group reaches, so small
# groups test fewer walls.
_PASS_ROWS = 1024
_CPU_PASS_ROWS = 16


class _LightPasses:
    """MuJoCo's light passes for a scene with a spotlight that tracks the agent.

    The renderer draws non-shadowing lights in one pass, then adds each shadowing light
    in its own blended pass (``mjr_render``), so every pass rounds to bytes before the
    sum saturates. The plate holds the static passes without the tracking light; per
    row this adds that light's shaded and shadowed contribution to static pixels,
    removes the agent's shadow from them, and shades the moving primitives pass by pass.
    """

    def __init__(self, scene, m, d, track, static_ids, static_hit, albedo, agent_ids, device):
        """``static_hit`` is (world point, normal, index into ``static_ids``, hit mask) per pixel."""
        import mujoco

        f = lambda x: torch.as_tensor(np.asarray(x, np.float64), device=device, dtype=torch.float32)  # noqa: E731
        self.base, self.shadowed = [], []
        self.spot = None
        for i in range(int(scene.nlight)):
            sl = scene.lights[i]
            colours = (f(sl.ambient), f(sl.diffuse), f(sl.specular))
            if int(sl.id) == track:
                D = np.asarray(sl.dir, np.float64)
                D = D / np.linalg.norm(D)
                up = np.cross(D, [-1.0, 0.0, 0.0])
                if up @ up <= 0.01:
                    up = np.cross(D, [0.0, 1.0, 0.0])
                s = np.cross(D, up / np.linalg.norm(up))
                s = s / np.linalg.norm(s)
                fov = min(2.0 * float(sl.cutoff) * float(m.vis.map.shadowscale), 160.0)
                self.spot = dict(
                    offset=f(np.asarray(sl.pos, np.float64) - np.array([d.qpos[0], d.qpos[1], 0.0])),
                    D=f(D),
                    cos_cut=float(np.cos(np.deg2rad(float(sl.cutoff)))),
                    exponent=float(sl.exponent),
                    att=[float(a) for a in sl.attenuation],
                    colours=colours,
                    castshadow=bool(sl.castshadow),
                    s=f(s),
                    u=f(np.cross(s, D)),
                    tan_half=float(np.tan(np.deg2rad(fov / 2.0))),
                    cutoff=float(sl.cutoff),
                )
                continue
            if int(sl.type) != int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL):
                raise NotImplementedError("only directional lights besides the tracking spotlight")
            L = -np.asarray(sl.dir, np.float64)
            entry = dict(to_light=f(L / np.linalg.norm(L)), colours=colours)
            if sl.castshadow:
                # The orthographic shadow map clips casters nearer than the camera's near plane.
                entry.update(pos=f(sl.pos), fwd=f(-L / np.linalg.norm(L)), near=float(scene.camera[0].frustum_near))
                self.shadowed.append(entry)
            else:
                self.base.append(entry)
        self.eye = f(np.mean([np.asarray(scene.camera[k].pos, np.float64) for k in (0, 1)], 0))

        walls = [g for g in static_ids if int(m.geom_type[g]) != 0]  # planes cast no shadow upward
        self.wall_kinds = [int(m.geom_type[g]) for g in walls]
        self.wall_size = f(np.stack([m.geom_size[g] for g in walls]))[None]
        self.wall_pos = f(np.stack([d.geom_xpos[g] for g in walls]))[None]
        self.wall_rot = f(np.stack([d.geom_xmat[g].reshape(3, 3) for g in walls]))[None]
        # World boxes of the walls, for culling walls that no shadow ray of a frame can reach.
        self.wall_boxes = None
        if walls and all(kind == _BOX for kind in self.wall_kinds):
            half = np.stack(
                [np.abs(d.geom_xmat[g].reshape(3, 3)) @ np.asarray(m.geom_size[g], np.float64) for g in walls]
            )
            centre = np.stack([d.geom_xpos[g] for g in walls])
            self.wall_boxes = (
                torch.as_tensor(centre - half, device=device, dtype=torch.float64),
                torch.as_tensor(centre + half, device=device, dtype=torch.float64),
            )

        self.hit, self.normal, which, self.valid = static_hit
        mats = f(np.stack([_mat_and_tex(m, g)[0] for g in static_ids]))
        ss = np.array([_spec_shin(m, g) for g in static_ids])
        self.mat = mats[which]
        self.tex = torch.as_tensor(albedo, device=device, dtype=torch.float32).reshape(-1, 3) / 255.0
        self.ms = f(ss[:, 0])[which]
        self.shin = f(ss[:, 1])[which]
        z = self.hit[self.valid, 2]
        self.zrange = (float(z.min()), float(z.max())) if z.numel() else (0.0, 0.0)

        pss = np.array([_spec_shin(m, g) for g in agent_ids])
        self.prim_ms, self.prim_shin = f(pss[:, 0]), f(pss[:, 1])

    def light_pos(self, xy: torch.Tensor) -> torch.Tensor:
        lp = torch.zeros(xy.shape[0], 3, device=xy.device, dtype=torch.float32)
        lp[:, :2] = xy
        return lp + self.spot["offset"]

    def _spot_terms(self, X, lpos):
        sp = self.spot
        L = lpos - X
        dist = torch.sqrt(dot3(L, L))
        L = L / dist.clamp(min=1e-12).unsqueeze(-1)
        sd = -dot3(L, sp["D"])
        gain = torch.where(sd >= sp["cos_cut"], sd.clamp(min=0) ** sp["exponent"], torch.zeros_like(sd))
        k0, k1, k2 = sp["att"]
        gain = gain / (k0 + k1 * dist + k2 * dist * dist)
        return L, dist, gain

    def _near_walls(self, o, L, lim, valid):
        """Per-row wall parameters, keeping only walls whose box meets the box of the row's shadow rays.

        A culled wall cannot hit a ray, and a far box pads the rows with fewer walls; the hits
        are the same as with every wall. Rays outside ``valid`` do not set the box.
        """
        lo_w, hi_w = self.wall_boxes
        o64, end64 = o.to(torch.float64), (o + lim.unsqueeze(-1) * L).to(torch.float64)
        keep = valid.unsqueeze(-1)
        big = torch.tensor(np.inf, dtype=torch.float64, device=o.device)
        lo = torch.where(keep, torch.minimum(o64, end64), big).amin(1) - _CULL_SLACK
        hi = torch.where(keep, torch.maximum(o64, end64), -big).amax(1) + _CULL_SLACK
        near = ((lo_w.unsqueeze(0) <= hi.unsqueeze(1)) & (hi_w.unsqueeze(0) >= lo.unsqueeze(1))).all(-1)
        count = near.sum(1)
        width = max(int(count.max()), 1)
        order = torch.argsort((~near).to(torch.int8), dim=1, stable=True)[:, :width]
        used = (torch.arange(width, device=o.device).unsqueeze(0) < count.unsqueeze(1)).unsqueeze(-1)
        size = torch.where(used, self.wall_size[0][order], 1.0)
        pos = torch.where(used, self.wall_pos[0][order], _FAR)
        rot = torch.where(used.unsqueeze(-1), self.wall_rot[0][order], torch.eye(3, device=o.device))
        return [_BOX] * width, size, pos, rot

    def _spot_shadow(self, X, n, L, dist, lpos, agent, skip, valid=None):
        """Shadowed by walls or agent, within the square shadow-map frustum; lit outside it.

        ``valid`` marks the rows' pixels whose result is used; the others may differ.
        """
        sp = self.spot
        if not sp["castshadow"]:
            return torch.zeros(X.shape[:-1], device=X.device, dtype=torch.bool)
        rel = X - lpos
        depth = dot3(rel, sp["D"])
        reach = sp["tan_half"] * depth
        inside = (depth > 0) & (dot3(rel, sp["s"]).abs() <= reach) & (dot3(rel, sp["u"]).abs() <= reach)
        o = X + _SHADOW_EPS * n
        lim = dist - _SHADOW_EPS
        # One launch row per frame keeps the per-row block count within the CUDA grid limit.
        rows = X.shape[0] if X.dim() == 3 and X.shape[1] > 1 else 1
        wk = len(self.wall_kinds)
        o, L, lim = o.reshape(rows, -1, 3), L.reshape(rows, -1, 3).contiguous(), lim.reshape(rows, -1)
        if valid is not None and self.wall_boxes is not None and X.device.type == "cpu" and _CULL_WALLS:
            kinds, size, pos, rot = self._near_walls(o, L, lim, valid.reshape(rows, -1))
        else:
            kinds = self.wall_kinds
            size = self.wall_size.expand(rows, wk, 3)
            pos = self.wall_pos.expand(rows, wk, 3)
            rot = self.wall_rot.expand(rows, wk, 3, 3)
        walls = R3.any_blocking(kinds, size, pos, rot, o, L, lim).view(X.shape[:-1])
        o, L, lim = o.view(X.shape), L.view(X.shape), lim.view(X.shape[:-1])
        kinds, size, pos, rot = agent
        moving = R3.any_blocking(kinds, size, pos, rot, o, L.contiguous(), lim, skip)
        return inside & (walls | moving)

    def _dir_shadow(self, X, n, light, agent, skip):
        o = X + _SHADOW_EPS * n
        lim = dot3(X - light["pos"], light["fwd"]) - light["near"]
        d = light["to_light"].expand_as(o).contiguous()
        kinds, size, pos, rot = agent
        return R3.any_blocking(kinds, size, pos, rot, o, d, lim, skip)

    def _window(self, cam, lpos, H, W):
        """Flat (B, T*T) pixel indices covering the spotlight's footprint on static geometry."""
        B = lpos.shape[0]
        zlo, zhi = self.zrange
        cut = self.spot["cutoff"]
        if cut >= 89.0:
            return torch.arange(H * W, device=lpos.device).expand(B, H * W)
        reach = (float(self.spot["offset"][2]) - zlo) * float(np.tan(np.deg2rad(cut))) + 1.0
        corners = []
        for sx in (-reach, reach):
            for sy in (-reach, reach):
                for z in (zlo, zhi):
                    p = lpos.clone()
                    p[:, 0] += sx
                    p[:, 1] += sy
                    p[:, 2] = z
                    corners.append(p)
        ndc = cam.project(torch.stack(corners, 1), aspect=W / H)
        px = (ndc[..., 0] + 1) * 0.5 * W
        py = (1 - ndc[..., 1]) * 0.5 * H
        x0 = px.amin(1).floor().long() - 1
        y0 = py.amin(1).floor().long() - 1
        span = torch.stack([(px.amax(1).ceil().long() + 1 - x0).max(), (py.amax(1).ceil().long() + 1 - y0).max()])
        need = int(span.max())
        if need >= min(H, W):
            return torch.arange(H * W, device=lpos.device).expand(B, H * W)
        x0 = x0.clamp(0, W - need)
        y0 = y0.clamp(0, H - need)
        ar = torch.arange(need, device=lpos.device)
        return ((y0.view(B, 1, 1) + ar.view(1, -1, 1)) * W + (x0.view(B, 1, 1) + ar.view(1, 1, -1))).reshape(B, -1)

    def plates(self, plate, cam, lpos, agent):
        """Per-row plates: static pixels gain the tracking light and lose the agent's shadow."""
        H, W = plate.shape[:2]
        B = lpos.shape[0]
        idx = self._window(cam, lpos, H, W)
        X, n = self.hit[idx], self.normal[idx]
        c, tex, ms, shin = self.mat[idx], self.tex[idx], self.ms[idx], self.shin[idx]
        v = self.eye - X
        v = v / torch.sqrt(dot3(v, v)).clamp(min=1e-12).unsqueeze(-1)
        lp = lpos.unsqueeze(1).expand(B, idx.shape[1], 3)
        L, dist, gain = self._spot_terms(X, lp)
        add = _light_pass(n, v, [(L, gain, self.spot["colours"])], c, tex, ms, shin)
        valid = self.valid[idx]
        add = torch.where(self._spot_shadow(X, n, L, dist, lp, agent, None, valid).unsqueeze(-1), 0.0, add)
        one = torch.ones_like(gain)
        for light in self.shadowed:
            lost = _light_pass(n, v, [(light["to_light"].expand_as(n), one, light["colours"])], c, tex, ms, shin)
            add = add - torch.where(self._dir_shadow(X, n, light, agent, None).unsqueeze(-1), lost, 0.0)
        add = torch.where(valid.unsqueeze(-1), add, 0.0)
        flat = plate.reshape(H * W, 3)
        patch = (flat[idx].to(torch.float32) + add).clamp(0, 255).to(torch.uint8)
        out = flat.unsqueeze(0).repeat(B, 1, 1)
        out.scatter_(1, idx.unsqueeze(-1).expand(B, idx.shape[1], 3), patch)
        return out.view(B, H, W, 3)

    def shader(self, lpos, agent):
        """Composite shader: each moving-primitive sample summed over byte-rounded passes."""
        kinds, size, pos, rot = agent

        def run(rows, X, n, v, ih, mat):
            tex = torch.ones_like(mat)
            ms, shin = self.prim_ms[ih], self.prim_shin[ih]
            one = torch.ones_like(ms)
            terms = [(b["to_light"].expand_as(n), one, b["colours"]) for b in self.base]
            total = _light_pass(n, v, terms, mat, tex, ms, shin)
            mine = (kinds, size[rows], pos[rows], rot[rows])
            Xr, nr, skip = X.unsqueeze(1), n.unsqueeze(1), ih.unsqueeze(1)
            for light in self.shadowed:
                got = _light_pass(n, v, [(light["to_light"].expand_as(n), one, light["colours"])], mat, tex, ms, shin)
                dark = self._dir_shadow(Xr, nr, light, mine, skip).squeeze(1)
                total = total + torch.where(dark.unsqueeze(-1), 0.0, got)
            lp = lpos[rows]
            L, dist, gain = self._spot_terms(X, lp)
            got = _light_pass(n, v, [(L, gain, self.spot["colours"])], mat, tex, ms, shin)
            every = torch.ones(Xr.shape[:-1], device=Xr.device, dtype=torch.bool)
            dark = self._spot_shadow(
                Xr, nr, L.unsqueeze(1), dist.unsqueeze(1), lp.unsqueeze(1), mine, skip, every
            ).squeeze(1)
            return torch.clamp(total + torch.where(dark.unsqueeze(-1), 0.0, got), max=255.0)

        return run


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
        tex_id=int(tex.id),
        checker=_checker_masks(loco),
        global_light=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "global"),
        light_diffuse=np.array(m.light_diffuse, copy=True),
        cam_distance=float(u.custom_camera.distance),
        cam_elevation=float(u.custom_camera.elevation),
        mat_rgba=np.array(m.mat_rgba, copy=True),
        geom_rgba=np.array(m.geom_rgba, copy=True),
        # Texture bytes upload at context creation, so cache a renderer per texture mode;
        # recoloured checkers share one renderer and upload their texels explicitly.
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


def _upload_xml_checker(env, loco: str) -> None:
    """Replace a pixel-scene source floor with the locomotion XML's checker texels and upload them."""
    import importlib

    import mujoco

    module = importlib.import_module(f"ogbench.locomaze.{loco}")
    xml = mujoco.MjModel.from_xml_path(getattr(module, f"{loco.capitalize()}Env").xml_file)
    u = env.unwrapped
    m = u.model
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    src, dst = xml.tex("grid"), m.tex("grid")
    shape = (int(dst.height[0]), int(dst.width[0]))
    if (int(src.height[0]), int(src.width[0])) != shape:
        raise RuntimeError(f"the {loco} XML checker and the source floor texture differ in size")
    n = 3 * shape[0] * shape[1]
    getattr(m, attr)[int(dst.adr[0]) : int(dst.adr[0]) + n] = getattr(xml, attr)[int(src.adr[0]) : int(src.adr[0]) + n]
    r = u.custom_renderer
    r._gl_context.make_current()
    mujoco.mjr_uploadTexture(m, r._mjr_context, int(dst.id))


def _checker_masks(loco: str):
    """Texel masks of the XML checker's two colours; pixel scenes overwrite it at construction."""
    import importlib

    import mujoco

    from ..._core.plate_texels import texel_slots

    module = importlib.import_module(f"ogbench.locomaze.{loco}")
    asset = getattr(module, f"{loco.capitalize()}Env").xml_file
    return texel_slots(mujoco.MjModel.from_xml_path(asset), "grid", CHECKER_RGB)["masks"]


class OGBenchMaze(Env):
    """Render OGBench maze bodies over cached source backgrounds.

    Pixel-mode rows contain qpos; state-observation mode also carries the goal.
    """

    family = "ogbench"

    LOCO: str = ""
    MAZE: str = ""
    default_size = 224
    # Approximate source four-sample edges by casting at double resolution.
    default_supersample = 2
    # Moving-primitive cast window; oversized projections fall back to the full frame.
    TILE = 32
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

        Keep RGB in [0, 1]; uint8 quantization changes MuJoCo material shading.
        """
        agent_rgb = [0.7, 0.5, 0.3]
        # Pixel observations recolour the XML's white walls; state observations keep them.
        wall_rgb = [1.0, 1.0, 1.0] if getattr(self, "RENDER_OB_TYPE", "pixels") == "states" else [0.6, 0.6, 0.6]
        return Dict(
            {
                "agent": Dict(
                    {
                        "color": Box(low=0.0, high=1.0, shape=(3,), init_value=agent_rgb),
                        "scale": Box(low=0.25, high=6.0, shape=(1,), init_value=[1.0]),
                    }
                ),
                "floor": Dict(
                    {
                        "tint": Box(low=0.0, high=1.0, shape=(3,), init_value=[1.0, 1.0, 1.0]),
                        "color": Box(low=0.0, high=1.0, shape=(2, 3), init_value=CHECKER_RGB),
                        # Pixel scenes only: 0 draws the checker, 1 the position-encoding gradient.
                        "pixel_encoding": Discrete(2, init_value=0),
                    }
                ),
                "light": Dict({"intensity": Box(low=0.0, high=1.0, shape=(1,), init_value=[1.0])}),
                "walls": Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=wall_rgb)}),
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

    #: Style leaves that change the static scene and therefore the baked plate. The rest belong to the
    #: agent's primitives and are applied per environment inside the ray-cast.
    PLATE_PATHS = (
        "floor.tint",
        "walls.color",
        "camera.distance_scale",
        "camera.elevation",
        "rendering.decolor_floor",
        "floor.color",
        "floor.pixel_encoding",
        "light.intensity",
    )
    #: Appearance leaves ``reset`` accepts by value or by sampling, as swm's OGBMaze does.
    reset_value_paths = frozenset(
        {"agent.color", "walls.color", "floor.color", "floor.pixel_encoding", "light.intensity"}
    )
    reset_sample_paths = reset_value_paths

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
            self._floor_key(variation),
            float(np.round(self._var(variation, "light.intensity")[0], 6)),
            str(self.device),
        )

    def _checker_shown(self, variation) -> bool:
        if int(self._var(variation, "rendering.decolor_floor")[0]):
            return False
        return getattr(self, "RENDER_OB_TYPE", "pixels") == "states" or not int(
            self._var(variation, "floor.pixel_encoding")[0]
        )

    def _floor_key(self, variation):
        """Checker bytes when the checker is drawn, so equal textures share a plate."""
        if not self._checker_shown(variation):
            return None
        return tuple(int(x) for x in quantise(self._var(variation, "floor.color")))

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
        and kinematics calls must be serialized.
        """
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
        floor = buf[pr["tex_adr"] : pr["tex_adr"] + pr["tex_n"]]
        if int(self._var(variation, "rendering.decolor_floor")[0]):
            floor[:] = np.tile(np.array([60, 64, 72], buf.dtype), pr["tex_n"] // 3)
        elif self._checker_shown(variation):
            texels = floor.reshape(-1, 3)
            for mask, colour in zip(pr["checker"], self._var(variation, "floor.color").reshape(2, 3), strict=True):
                texels[mask] = quantise(colour)
        else:
            floor[:] = pr["tex"]
        m.light_diffuse[:] = pr["light_diffuse"]
        m.light_diffuse[pr["global_light"]] = float(self._var(variation, "light.intensity")[0])

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
        custom = not rkey and not np.array_equal(floor, pr["tex"])
        rkey = "custom" if custom else rkey
        r = pr["renderers"].get(rkey)
        if r is None:
            # Built after the texture write for this key, so it uploads the right floor; every later
            # bake with the same key writes the same bytes, so reuse is exact.
            r = pr["renderers"][rkey] = mujoco_renderer(m, width=size, height=size)
        if custom:
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r._mjr_context, pr["tex_id"])
        # A light that tracks the agent cannot be baked; it is added per row instead.
        trackcom = int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM)
        track = [i for i in range(m.nlight) if m.light_active[i] and int(m.light_mode[i]) == trackcom]
        m.light_active[track] = 0
        r.update_scene(d, camera=cam)
        plate = r.render().copy()
        albedo = self._bake_albedo(m, r, d, cam, static) if track else None
        m.light_active[track] = 1
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
        # The fitted split is calibrated at unit intensity; the global light's diffuse share scales.
        dimmed = np.array(
            [
                m.light_diffuse[pr["global_light"], 0] if int(r.scene.lights[i].id) == pr["global_light"] else 1.0
                for i in range(len(_NOMINAL["dif"][:L]))
            ]
        )
        ka = SHADE_AMBIENT / max(sum(_NOMINAL["amb"]), 1e-9)
        kd = SHADE_DIFFUSE / max(sum(_NOMINAL["dif"]), 1e-9)
        ks = SHADE_SPECULAR / max(sum(_NOMINAL["spec"]) * MAT_SPECULAR, 1e-9)
        sc.lights = R3.Lights(
            dirs=dirs,
            ambient=np.array(_NOMINAL["amb"][:L]) * ka,
            diffuse=np.array(_NOMINAL["dif"][:L]) * kd * dimmed,
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
        sc.goal = None
        if getattr(self, "RENDER_OB_TYPE", "pixels") == "states" and len(agent) > 1:
            target = agent.index(m.geom("target").id)
            sc.goal = ([target], [i for i in range(len(agent)) if i != target])
        sc.passes = None
        if track:
            assert len(track) == 1 and self.LOCO == "point", "a tracking light is modelled for the point body only"
            f = lambda x: torch.as_tensor(np.asarray(x), device=self.device, dtype=torch.float32)[None]  # noqa: E731
            o, ray = R3.cached_rays(camera, size, size, self.device)
            t, nrm, which = R3.cast_nearest(
                sprims.kind,
                f(sprims.size),
                f(np.stack([d.geom_xpos[g] for g in static])),
                f(np.stack([d.geom_xmat[g].reshape(3, 3) for g in static])),
                o.expand(1, size * size, 3),
                ray[None],
            )
            t, hit = t[0], o + t[0].clamp(max=1e6).unsqueeze(-1) * ray
            sc.passes = _LightPasses(
                r.scene, m, d, track[0], static, (hit, nrm[0], which[0], t < R3.INF / 2), albedo, agent, self.device
            )
        return sc

    @staticmethod
    def _bake_albedo(m, r, d, cam, static):
        """Static texels as bytes: white materials under a unit ambient headlight, agent hidden."""
        hl = m.vis.headlight
        saved = (hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy(), m.light_active.copy())
        mat, geom = m.mat_rgba.copy(), m.geom_rgba.copy()
        hl.ambient[:], hl.diffuse[:], hl.specular[:] = 1.0, 0.0, 0.0
        m.light_active[:] = 0
        m.mat_rgba[:, :3] = 1.0
        m.geom_rgba[static, :3] = 1.0
        try:
            r.update_scene(d, camera=cam)
            return r.render().copy()
        finally:
            hl.ambient[:], hl.diffuse[:], hl.specular[:], m.light_active[:] = saved
            m.mat_rgba[:], m.geom_rgba[:] = mat, geom

    def _fk(self, states: torch.Tensor, sc: _Scene):
        """Return agent geom positions ``(B, P, 3)`` and rotations ``(B, P, 3, 3)``.

        Point-body translation stays on device. Legged bodies use serial CPU
        MuJoCo kinematics, gathering into preallocated arrays before one batch transfer.
        """
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

        Group only by plate-affecting styles; body colour and scale remain per row.
        """
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = s.shape[0]
        size = int(size or self.size)
        tuning.on_render("ogb-maze", size, B, self)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        plate_key = torch.cat([sty[p].reshape(B, -1) for p in self.PLATE_PATHS], 1).cpu().numpy()
        plate_key = np.round(plate_key.astype(np.float64), 6)
        if (plate_key == plate_key[:1]).all():  # one shared scene: skip the row sort
            uniq, inv = plate_key[:1], np.zeros(B, np.intp)
        else:
            uniq, inv = np.unique(plate_key, axis=0, return_inverse=True)
        for gi in range(len(uniq)):
            idx = np.flatnonzero(inv == gi)
            over = {p: uniq[gi][o : o + w] for p, (o, w) in zip(self.PLATE_PATHS, _offsets(self.PLATE_PATHS, sty))}
            sc = self._scene(size, over)
            # Per-row lighting holds several float images per row; bound the rows in flight.
            rows = _CPU_PASS_ROWS if self.device.type == "cpu" else _PASS_ROWS
            step = rows if sc.passes is not None else len(idx)
            for lo in range(0, len(idx), step):
                part = idx[lo : lo + step]
                sub = torch.as_tensor(part, device=self.device, dtype=torch.long)
                mat, scale = self._agent_appearance(sc, sty, sub)
                xpos, xmat = self._fk(s[sub], sc)
                plate, shader, agent, lpos = sc.plate, None, None, None
                if sc.passes is not None:
                    size_b = torch.as_tensor(np.asarray(sc.prims_base.size), device=self.device, dtype=torch.float32)
                    agent = (list(sc.prims_base.kind), size_b * scale.view(-1, 1, 1), xpos, xmat)
                    lpos = sc.passes.light_pos(s[sub][:, :2])
                    plate = sc.passes.plates(sc.plate, sc.cam, lpos, agent)
                    shader = sc.passes.shader(lpos, agent)
                # Advanced indexing makes buf[sub] a copy. With multiple groups, render to a
                # temporary and assign back explicitly; one group can write directly into buf.
                dst = buf if len(uniq) == 1 and len(part) == B else None
                apart = self._movers_apart(sc, xpos, scale, size)
                if apart is None:
                    got = R3.composite(
                        plate,
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
                        shader=shader,
                    )
                else:
                    got = self._composite_apart(sc, apart, plate, agent, lpos, xpos, xmat, mat, scale, size, ss, dst)
                if dst is None:
                    buf[sub] = got
        return buf.view(*lead, size, size, 3) if lead else buf

    def _movers_apart(self, sc: _Scene, xpos, scale, size):
        """Return a host mask of rows whose goal and body screen bounds do not meet, or None.

        None means that every row is drawn in one composite.
        """
        if sc.goal is None or not _SPLIT_MOVERS:
            return None
        bs = float(scale.max())
        bounds = []
        for ids in sc.goal:
            got = R3.screen_bounds(sc.cam, sc.prims_base.subset(ids), xpos[:, ids], size, size, self.device, bs)
            if got is None:
                return None
            bounds.append(got)
        (gx0, gx1, gy0, gy1), gndc = bounds[0]
        (ax0, ax1, ay0, ay1), andc = bounds[1]
        apart = (ax1 < gx0) | (gx1 < ax0) | (ay1 < gy0) | (gy1 < ay0)
        finite = torch.isfinite(gndc).flatten(1).all(1) & torch.isfinite(andc).flatten(1).all(1)
        apart = (apart & finite).cpu().numpy()
        return apart if apart.any() else None

    def _composite_apart(self, sc, apart, plate, agent, lpos, xpos, xmat, mat, scale, size, ss, out):
        """Composite rows whose goal and body do not meet in two small tiles, the others in one pass.

        Neither mover can cover a sample of the other's tile, so each tile gets the bytes of one pass.
        """
        if out is None:
            out = torch.empty((len(xpos), size, size, 3), dtype=torch.uint8, device=self.device)
        for rows, split in ((np.flatnonzero(apart), True), (np.flatnonzero(~apart), False)):
            if not len(rows):
                continue
            r = torch.as_tensor(rows, device=self.device, dtype=torch.long)
            shader = None
            if sc.passes is not None:
                kinds, sizes, _, _ = agent
                shader = sc.passes.shader(lpos[r], (kinds, sizes[r], xpos[r], xmat[r]))
            base = plate[r] if plate.dim() == 4 else plate
            args = (sc.plate_t, sc.cam, sc.prims_base, xpos[r], xmat[r], sc.lights, size, size)
            kw = dict(supersample=ss, tile=self.TILE, mat_rgb=mat[r], size_scale=scale[r], shader=shader)
            if split:
                goal_ids, body_ids = sc.goal
                base = R3.composite(base, *args, cast=goal_ids, **kw)
                got = R3.composite(base, *args, cast=body_ids, **kw)
            else:
                got = R3.composite(base, *args, **kw)
            out[r] = got
        return out

    def _agent_appearance(self, sc: _Scene, sty, sub):
        """Return per-geom colours and per-row scale, preserving non-self materials.

        The ant’s white leg geoms distinguish its heading and must not be recoloured.
        """
        base = torch.as_tensor(sc.agent_base, device=self.device, dtype=torch.float32)
        is_self = torch.as_tensor(sc.agent_is_self, device=self.device)
        acol = sty["agent.color"][sub].to(torch.float32)  # (b,3)
        mat = base.unsqueeze(0).expand(len(sub), *base.shape).clone()
        mat[:, is_self] = acol.unsqueeze(1).expand(-1, int(is_self.sum()), 3)
        return mat, sty["agent.scale"][sub].to(torch.float32).reshape(-1)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample synthetic poses in free cells, with torso and joint noise for legged bodies.

        Varying quaternion orientation also prevents fixed-offset probes from
        normalizing straight back to the identity. These are not source rollout states.
        """
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
        """Construct an independent source environment for reference comparisons.

        A pixel scene shows the XML checker floor, which is the default ``floor.pixel_encoding=0``
        and what swm's first reset draws. Keep its GL context alive while other renderers are in use.
        """
        ob_type = getattr(cls, "RENDER_OB_TYPE", "pixels")
        env = _build_reference(cls.LOCO, cls.MAZE, size, ob_type)
        if ob_type == "pixels":
            _upload_xml_checker(env, cls.LOCO)
        return env

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


class MazeSemantics:
    """Compose full-State batched simulation of the source maze with state observations.

    The renderer projection is qpos, followed by the goal when the scene draws it.
    """

    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"
    # Each step returns outputs in new storage.
    _step_outputs_owned = True

    def __init__(self, *args, simulation_backend="default", mjwarp_min_envs=None, **kwargs):
        """Select physics at construction; outputs are on ``device``.

        ``"default"`` steps CUDA batches on MuJoCo Warp (float32) and CPU batches on ``mujoco.batch``, which
        equals the source bit for bit. ``"auto"`` steps CUDA batches smaller than ``mjwarp_min_envs`` (by
        default a measured size per body) on ``mujoco.batch``. ``"native"`` and ``"mjwarp"`` select explicitly.
        """
        super().__init__(*args, **kwargs)
        self.simulation_backend = maze_backend(self.B, self.device, self.LOCO, simulation_backend, mjwarp_min_envs)
        self._simulator = make_maze_adapter(
            self.B, self.device, self._seed, loco=self.LOCO, maze=self.MAZE, simulation_backend=self.simulation_backend
        )
        self.execution_mode = getattr(self._simulator, "execution_mode", "DELEGATED_CPU")

    @property
    def _qpos_fields(self):
        return BODIES[self.LOCO][0]

    def _draws_goal(self):
        return getattr(self, "RENDER_OB_TYPE", "pixels") == "states"

    def _layout(self):
        offsets, start = {}, 0
        for key, width in self._qpos_fields:
            offsets[key] = start
            start += width
        if self._draws_goal():
            offsets[("task", "goal_pos")] = start
        return tuple((key, width, offsets.get(key)) for key, (width, _) in maze_fields(self.LOCO).items())

    def _layout_dtypes(self):
        return {key: dtype for key, (_, dtype) in maze_fields(self.LOCO).items()}

    def _render_fields(self):
        return self._qpos_fields + (((("task", "goal_pos"), 3),) if self._draws_goal() else ())

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full OGBMaze State")
        return torch.cat([state[key] for key, _ in self._render_fields()], dim=-1).to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, self.state_dim)
        out, start = {}, 0
        for key, width in self._render_fields():
            out[key] = rows[:, start : start + width].clone()
            start += width
        return TensorDict(out, [len(rows)], device=self.device)

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        self.draw_reset_values(seed, options)
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

    def _stack_states(self, values):
        """Stack the States of a step sequence; the CUDA simulator stacks its own States with one copy."""
        stack = getattr(self._simulator, "stack_states", None)
        stacked = None if stack is None else stack(values)
        return _stack_states(values) if stacked is None else stacked

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()


class _StatesScene:
    """Render the state-observation scene: checker floor, white walls and the goal disc."""

    RENDER_OB_TYPE = "states"


#: ``__module__`` is pinned to the package so ``stable_envs.envs.ogbench.<Cell>`` resolves.
_MODULE = "stable_envs.envs.ogbench"


def _make_classes():
    """Register dataset-specific names; point/large/navigate renders the state-observation scene."""
    made = {}
    for loco, maze, dataset in CELLS:
        cname = f"{loco.capitalize()}Maze{maze.capitalize()}{dataset.capitalize()}"
        bases = (MazeSemantics, OGBenchMaze)
        if (loco, maze, dataset) == ("point", "large", "navigate"):
            bases = (_StatesScene, *bases)
        cls = register(
            type(
                cname,
                bases,
                dict(
                    LOCO=loco,
                    MAZE=maze,
                    action_dim=BODIES[loco][2],
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
    return made


#: ``{name: class}`` for every registered cell this module declares; the package ``__init__`` re-exports them.
CELL_CLASSES = _make_classes()
globals().update(CELL_CLASSES)
