from __future__ import annotations

import warnings

import numpy as np
import torch

from ..._core.plate_texels import quantise as _quantise
from ..._core.plate_texels import texel_slots as _texel_slots
from ...render import _resources as _RES
from ...render import cpu3d as C3
from ...render import mjwarp as MW
from ...spaces import Box, Dict
from ._mass_rows import ARRAYS as _MASS_ARRAYS
from ._mass_rows import MassRows
from ._model import (
    _TC_CACHE,
    MJW_CHUNK,
    VISIBLE_GROUPS,
    _effective_alpha,
    _grid_defaults,
    _grid_masks,
    _row_width,
    _slot_element,
)
from .base import _DMCModel
from .simulation import physics as _sim_physics


class _TCtx:
    """Hold a captured MJWarp render graph and the per-world arrays it reads and writes."""

    __slots__ = (
        "wm",
        "wd",
        "rc",
        "bgra",
        "graph",
        "nworld",
        "res",
        "wp",
        "mjw",
        "textures",
        "reflection",
        "masses_varied",
        "floor_tilted",
    )


def _sky_plate(cls, size: int) -> np.ndarray:
    import mujoco
    from dm_control.mujoco import wrapper

    source, model, *_ = cls._bundle(cls.default_size)
    if model.cam_mode[cls.CAMERA] != mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM:
        raise ValueError("Cached TrackCom sky requires a camera with fixed orientation")
    option = wrapper.MjvOption()
    option.geomgroup[:] = 0
    option.sitegroup[:] = 0
    return source.physics.render(height=size, width=size, camera_id=cls.CAMERA, scene_option=option).copy()


def _tc_context(cls, res: int, nworld: int) -> _TCtx:
    """Cache a captured MJWarp context by domain, task, size, world count and flags.

    Run com_pos between kinematics and camlight: tracking cameras read subtree_com,
    which kinematics does not update. Capture the complete frame sequence to avoid
    repeated host launches."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    key = (cls.DOMAIN, cls.TASK, int(res), int(nworld), tuple(sorted(cls.RENDER_FLAGS.items())))
    hit = _TC_CACHE.get(key)
    if hit is not None:
        return hit
    _, m, _, pr, _, slots = cls._bundle(cls.default_size)
    # Canonical model: the context depends on the domain alone.
    for arr in {a for a, _ in slots}:
        getattr(m, arr)[:] = pr[arr]
    m.mat_rgba[:] = pr["mat_rgba"]
    m.light_diffuse[:] = pr["light_diffuse"]
    if not MW.needs_com_pos(m):
        raise RuntimeError(
            f"{cls.__name__}: no camera or light on this model tracks subtree_com, so it does not "
            f"belong in the trackcom family; the plate design in DMControlSuite is strictly better "
            f"for it"
        )
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(res))
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(res))

    # Hide zero-alpha elements before context creation; the MJWarp shader does not
    # read alpha and caches the selected geometry for the context lifetime.
    MW.prepare_model(m, VISIBLE_GROUPS)

    c = _TCtx()
    c.wp, c.mjw, c.nworld, c.res = wp, mjw, int(nworld), int(res)
    d0 = mujoco.MjData(m)
    d0.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d0)
    batched = sorted(
        {"mat_rgba", "light_diffuse"} | {a for a, _ in slots} | (set(_MASS_ARRAYS) if cls._mass_rows() else set())
    )
    c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in batched})
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    c.textures = c.reflection = None
    c.masses_varied = c.floor_tilted = False
    # Keep GL-style rounding active during source generation and first compilation,
    # including the warm-up render.
    with MW.gl_rounding():
        c.rc = MW.create_render_context(
            m,
            nworld=int(nworld),
            cam_res=(int(res), int(res)),
            render_rgb=True,
            render_seg=True,
            cam_active=[i == int(cls.CAMERA) for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS),
            use_textures=True,
            use_shadows=True,
            render_skybox=False,  # Misses receive the cached source sky after this pass.
            **dict(cls.RENDER_FLAGS),
        )
        from .trackcom_reflection import Reflection
        from .trackcom_texture import Textures

        c.textures = Textures(m, c.rc, _sky_plate(cls, int(res)))
        supported = _reflection_supported(c.rc)
        c.reflection = Reflection(c.wm, c.wd, c.rc, m, cls._floor_slots()) if supported else None
        _tc_frame(c)
        wp.synchronize()
    c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
    with wp.ScopedCapture() as cap:
        _tc_frame(c)
    c.graph = cap.graph
    _TC_CACHE.put(key, c)
    return c


def _reflection_supported(rc) -> bool:
    """The floor reflection reuses MJWarp 3.12 render internals; older versions render without it."""
    try:
        from mujoco_warp._src import render as upstream
    except ImportError:
        return False
    if hasattr(rc, "splat_count") and hasattr(upstream, "_make_cast_ray"):
        return True
    warnings.warn(
        "TrackCom floor reflections need mujoco-warp 3.12; rendering without them, so frames differ from "
        "the measured appearance. Install stable-envs[dmcontrol-gpu] for the full renderer.",
        RuntimeWarning,
        stacklevel=3,
    )
    return False


def _tc_frame(c: _TCtx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.com_pos(c.wm, c.wd)  # see _tc_context: the TRACKCOM camera reads subtree_com
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)
    if c.textures is not None:
        c.textures.render()
    if c.reflection is not None:
        c.reflection.render()


class DMControlTrackCom(_DMCModel):
    """Render body-tracking cameras with MJWarp on CUDA or cpu3d on CPU.

    The moving view requires a full scene render. Texture filtering and shading
    differ from source OpenGL; an explicit opengl backend uses its separate context.
    This path supports one sample per pixel and refuses uncertified CPU cells."""

    fidelity_tier = "visual"
    parity_interior_min = None
    default_supersample = 1
    # Field of view cannot vary per world in this MJWarp model upload.
    CAMERA_KNOB = False
    # Visibility is fixed during context creation; a per-row toggle would need
    # an alpha-aware shader or a separate context for each configuration.
    VISIBILITY_KNOB = False

    # Render-context flags participate in the cache key; defaults may differ by domain.
    RENDER_FLAGS: dict = {}

    CPU_RENDER_CHUNK = C3.CPU_CHUNK
    # Measured CPU choices at source-tested resolutions; larger images retain the memory bound.
    _CPU_RENDER_CHUNKS = {
        ("hopper", 64): 64,
        ("cheetah", 64): 64,
        ("cheetah", 84): 32,
        ("humanoid", 64): 16,
        ("humanoid", 84): 16,
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

        if self.backend == "opengl":
            return  # The explicit fidelity backend owns a separate model/context.
        if self.device.type == "cuda":
            self._context(self.size)  # build and capture up front, not mid-render
        elif self.device.type == "cpu":
            reason = type(self).CPU_UNCERTIFIED_REASON or None
            if reason:
                raise ValueError(
                    f"{type(self).__name__} ({self.DOMAIN}-{self.TASK}) is not certified on the CPU "
                    f"backend and is refused on device='cpu' by CPU_UNCERTIFIED ({reason}); "
                    f"device='cuda' draws it through MJWarp"
                )
            self._cpu_scene(self.size)
        else:
            raise ValueError(
                f"{type(self).__name__}: no 3-D backend for device {self.device}; use "
                f"device='cuda' (MJWarp) or device='cpu' (stable_envs.render.cpu3d)"
            )

    def _context(self, size: int, nworld: int = MJW_CHUNK) -> _TCtx:
        return _tc_context(type(self), int(size), int(nworld))

    @classmethod
    def _mass_rows(cls) -> MassRows | None:
        """Mass leaves move the tracking camera through subtree centres of mass; None without them."""
        if "_mass_rows_cache" not in cls.__dict__:
            key = cls._simulator_key() if hasattr(cls, "_simulator_key") else None
            rows = None if key is None else MassRows(key, cls._model())
            cls._mass_rows_cache = rows if rows is not None and rows.paths else None
        return cls._mass_rows_cache

    @property
    def FRAME_PHYSICS(self) -> frozenset[str]:  # noqa: N802
        rows = self._mass_rows()
        tilt = self._floor_tilt()
        return frozenset((*(rows.paths if rows else ()), *((tilt[0],) if tilt else ())))

    @classmethod
    def _floor_slots(cls) -> dict | None:
        """Texel slots of the floor's checker colours, or None without the suite's grid texture."""
        if "_floor_slots_cache" not in cls.__dict__:
            import mujoco

            m = cls._model()
            t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
            slots = None if t < 0 else _texel_slots(m, t, _grid_defaults()[:2], masks=_grid_masks(m, t))
            cls._floor_slots_cache = slots
        return cls._floor_slots_cache

    @classmethod
    def _floor_tilt(cls) -> tuple[str, int] | None:
        """The simulator's floor-tilt leaf and its plane geom, or None when the floor cannot tilt."""
        if "_floor_tilt_cache" not in cls.__dict__:
            import mujoco

            key = cls._simulator_key() if hasattr(cls, "_simulator_key") else None
            tilts = [k for k in (_sim_physics.knobs(key) if key else ()) if isinstance(k, _sim_physics.PlaneTilt)]
            cls._floor_tilt_cache = None
            if tilts:
                geom = mujoco.mj_name2id(cls._model(), mujoco.mjtObj.mjOBJ_GEOM, tilts[0].geom)
                cls._floor_tilt_cache = (tilts[0].leaf, geom)
        return cls._floor_tilt_cache

    def make_variation_space(self) -> Dict:
        """Add the floor checker's two RGB colours, defaulting to the suite's material file."""
        space = super().make_variation_space()
        slots = self._floor_slots()
        if slots is None:
            return space
        color = Box(low=0.0, high=1.0, shape=(2, 3), init_value=slots["default"].tolist())
        return Dict({**{k: space[k] for k in space}, "floor": Dict({"color": color})})

    def _row_floor(self, sty, lo: int, hi: int):
        """Per-row floor orientations (n,3,3), or None when every row keeps the compiled level floor."""
        tilt = self._floor_tilt()
        if tilt is None:
            return None
        degrees = sty[tilt[0]][lo:hi].reshape(-1)
        if not bool(degrees.any()):
            return None
        angle = torch.deg2rad(degrees.to(torch.float64))
        s, c = torch.sin(angle).float(), torch.cos(angle).float()
        z, o = torch.zeros_like(s), torch.ones_like(s)
        return torch.stack((c, z, s, z, o, z, -s, z, c), -1).reshape(-1, 3, 3)

    def _row_palette(self, sty, lo: int, hi: int):
        """Per-row checker colours as the compiled texture stores them, or None when every row has the default."""
        slots = self._floor_slots()
        if slots is None:
            return None
        colours = sty["floor.color"][lo:hi].reshape(-1, 2, 3)
        default = torch.as_tensor(slots["default"], dtype=colours.dtype, device=colours.device)
        if bool((colours == default).all()):
            return None
        rows = torch.floor(colours.to(torch.float64) * 255).clamp(0, 255)
        if torch.equal(
            rows, torch.as_tensor(_quantise(slots["default"]), dtype=rows.dtype, device=rows.device).expand_as(rows)
        ):
            return None
        return (rows / 255).float()

    def _row_masses(self, sty, lo: int, hi: int):
        """Per-row mass arrays, or None when every row draws the compiled model's masses."""
        rows = self._mass_rows()
        if rows is None:
            return None
        values = torch.cat([sty[p][lo:hi].reshape(hi - lo, 1) for p in rows.paths], 1)
        return None if rows.registered(values) else rows(values)

    @classmethod
    def _used_materials(cls) -> list[str]:
        """Expose materials on visible geoms only; these backends do not render sites."""
        import mujoco

        m, pr = cls._model(), cls._pristine()
        alpha = _effective_alpha(m, "geom", rgba=pr["geom_rgba"], mat_rgba=pr["mat_rgba"])
        used: list[int] = []
        for i in range(m.ngeom):
            if int(pr["geom_group"][i]) not in VISIBLE_GROUPS or alpha[i] <= 0:
                continue
            mid = int(m.geom_matid[i])
            if mid >= 0 and mid not in used:
                used.append(mid)
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}" for i in used]

    def _cpu_scene(self, size: int) -> C3.Scene:
        """Cache a CPU scene per resolution from canonical geometry and render flags."""
        cache = self.__dict__.setdefault("_cpu_scenes", _RES.BoundedCache(4, name="dmcontrol.trackcom.cpu"))
        hit = cache.get(int(size))
        if hit is not None:
            return hit
        m, pr = self._model(), self._pristine()
        for arr in {a for a, _ in self._slots()}:  # canonical model, as in _tc_context
            getattr(m, arr)[:] = pr[arr]
        m.mat_rgba[:] = pr["mat_rgba"]
        m.light_diffuse[:] = pr["light_diffuse"]
        m.geom_group[:] = pr["geom_group"]
        # Match MJWarp’s visible geom set, excluding sites. Static non-plane geoms
        # also receive per-row poses; their transforms simply repeat.
        alpha = _effective_alpha(m, "geom", rgba=pr["geom_rgba"], mat_rgba=pr["mat_rgba"])
        movers = [
            g
            for g in range(m.ngeom)
            if int(pr["geom_group"][g]) in VISIBLE_GROUPS and alpha[g] > 0 and int(m.geom_type[g]) != C3.PLANE
        ]
        flags = dict(self.RENDER_FLAGS)
        try:
            sc = C3.Scene(
                m,
                self.CAMERA,
                int(size),
                movers,
                groups=VISIBLE_GROUPS,
                enable_specular=bool(flags.get("enable_specular", True)),
                use_shadows=True,
                device=self.device,
                plane_palette=None if self._floor_slots() is None else self._floor_slots()["masks"],
            )
        except ValueError as e:
            # Report unsupported CPU geometry or lighting at construction.
            raise ValueError(
                f"{type(self).__name__} ({self.DOMAIN}-{self.TASK}) has no CPU 3-D backend: {e}. "
                f"device='cuda' draws it through MJWarp; device='cpu' is refused rather than "
                f"silently served by the GPU"
            ) from e
        sc.sky = (
            torch.as_tensor(_sky_plate(type(self), int(size)), device=self.device, dtype=torch.float32).reshape(-1, 3)
            / 255
        )
        return cache.put(int(size), sc)

    def _cpu_poses(self, s: torch.Tensor, sc: C3.Scene, masses=None):
        """Compute camera, mover and light poses on CPU from compact render rows.

        Preserve kinematics → comPos → camlight order so tracking uses current
        subtree centres of mass. Restore episode model entries after the batch."""
        import mujoco

        m, d = self._model(), self._bundle(self.default_size)[2]
        pr = self._pristine()
        st = s.detach().cpu().numpy().astype(np.float64)
        q, ep = st[:, : self.nq], st[:, self.nq :]
        B, K = st.shape[0], len(sc.movers)
        cam = np.empty((B, 3))
        xp = np.empty((B, K, 3))
        xm = np.empty((B, K, 3, 3))
        lp = np.empty((B, m.nlight, 3))
        ld = np.empty((B, m.nlight, 3))
        slots = self._slots()
        if masses is not None:
            masses = {name: value.detach().cpu().numpy() for name, value in masses.items()}
        try:
            for i in range(B):
                for (arr, k), v in zip(slots, ep[i]):
                    getattr(m, arr).reshape(-1)[k] = v
                for name, value in (masses or {}).items():
                    getattr(m, name)[:] = value[i]
                d.qpos[:] = q[i]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                mujoco.mj_camlight(m, d)
                cam[i] = d.cam_xpos[self.CAMERA]
                xp[i] = d.geom_xpos[sc.movers]
                xm[i] = d.geom_xmat[sc.movers].reshape(K, 3, 3)
                lp[i] = d.light_xpos
                ld[i] = d.light_xdir
        finally:
            for arr in {a for a, _ in slots}:
                getattr(m, arr)[:] = pr[arr]
            if masses is not None:
                self._mass_rows().restore(m)
        t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)  # noqa: E731
        return t(cam), t(xp), t(xm), t(lp), t(ld)

    def _render_cpu(self, s: torch.Tensor, sty, size: int, buf: torch.Tensor) -> None:
        import mujoco

        sc = self._cpu_scene(size)
        m, pr = self._model(), self._pristine()
        B = int(s.shape[0])
        cam, xp, xm, lp, ld = self._cpu_poses(s, sc, self._row_masses(sty, 0, B))
        mat = (
            torch.as_tensor(pr["mat_rgba"][:, :3], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(B, 1, 1)
        )
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                mat[:, i] = sty[f"material.{name}"][:B]
        base = torch.as_tensor(pr["light_diffuse"], dtype=torch.float32, device=self.device)
        ldif = torch.clamp(base.unsqueeze(0) * sty["light.diffuse_scale"][:B].reshape(B, 1, 1), 0, 1)
        chunk = self._CPU_RENDER_CHUNKS.get((self.DOMAIN, size), self.CPU_RENDER_CHUNK)
        # An instance-level CPU_RENDER_CHUNK overrides the measured table.
        chunk = vars(self).get("CPU_RENDER_CHUNK", chunk)
        C3.render(
            sc,
            cam,
            xp,
            xm,
            lp,
            ld,
            mat,
            ldif,
            out=buf,
            chunk_size=chunk,
            plane_mat=self._row_floor(sty, 0, B),
            plane_palette=self._row_palette(sty, 0, B),
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
        backend=None,
    ) -> torch.Tensor:
        """Render compact rows to uint8 frames using the selected backend and per-row styles."""
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = s.shape[0]
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        if ss != 1:
            raise ValueError(f"{type(self).__name__}: only supersample=1 is supported.")
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        if (self.backend if backend is None else backend) == "opengl":
            from . import trackcom_opengl

            trackcom_opengl.render(self, s, sty, size, buf)
            return buf.view(*lead, size, size, 3) if lead else buf
        if self.device.type == "cpu":
            self._render_cpu(s, sty, size, buf)
            return buf.view(*lead, size, size, 3) if lead else buf
        # Small images amortize uploads across more worlds; large ones keep the memory bound.
        capacity = 256 if size <= 84 else MJW_CHUNK
        worlds = min(capacity, 1 << max(0, B - 1).bit_length())
        c = self._context(size, worlds)
        masses = self._row_masses(sty, 0, B)
        floor, palette = self._row_floor(sty, 0, B), self._row_palette(sty, 0, B)
        if palette is not None and (c.reflection is None or c.reflection.palette is None):
            raise RuntimeError(
                f"{type(self).__name__}: floor.color is drawn by the floor reflection pass, which needs "
                f"mujoco-warp 3.12; install stable-envs[dmcontrol-gpu]"
            )
        for lo in range(0, B, worlds):
            hi = min(B, lo + worlds)
            self._upload(c, s[lo:hi], sty, lo, hi, masses, floor, palette)
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            # MJWarp’s packed byte view is BGRA; select RGB channels explicitly.
            buf[lo:hi] = c.bgra[: hi - lo][..., [2, 1, 0]]
        return buf.view(*lead, size, size, 3) if lead else buf

    @classmethod
    def _static_slot_rows(cls) -> dict:
        """Map world-body position slots to derived data rows requiring direct uploads.

        MJWarp kinematics skips world-body geoms/sites. Updating only their model
        positions leaves stale world poses; the identity parent frame permits direct copies."""
        hit = cls.__dict__.get("_slot_rows_cache")
        if hit is not None:
            return hit
        m, pr = cls._model(), cls._pristine()
        rows: dict = {}
        for j, (arr, k) in enumerate(cls._slots()):
            kind, eid = _slot_element(m, arr, k)
            if kind not in ("geom", "site") or not arr.endswith("_pos"):
                continue
            bodyid = m.geom_bodyid if kind == "geom" else m.site_bodyid
            if int(bodyid[eid]) != 0:
                continue  # kinematics DOES recompute this one
            rows.setdefault((kind, eid), [np.asarray(pr[arr], np.float64)[eid].copy(), []])[1].append(
                (j, int(k % _row_width(m, arr)))
            )
        cls._slot_rows_cache = rows
        return rows

    def _upload(self, c: _TCtx, s: torch.Tensor, sty, lo: int, hi: int, masses=None, floor=None, palette=None) -> None:
        """Write compact rows and styles into the context’s per-world buffers."""
        import mujoco

        n = hi - lo
        wp = c.wp
        wp.to_torch(c.wd.qpos)[:n] = s[:, : self.nq]
        for j, (arr, k) in enumerate(self._slots()):
            wp.to_torch(getattr(c.wm, arr))[:n].reshape(n, -1)[:, k] = s[:, self.nq + j]
        for (kind, eid), (base, comps) in self._static_slot_rows().items():
            dst = wp.to_torch(c.wd.geom_xpos if kind == "geom" else c.wd.site_xpos)
            row = torch.as_tensor(base, device=self.device, dtype=torch.float32).repeat(n, 1)
            for j, comp in comps:
                row[:, comp] = s[:, self.nq + j]
            dst[:n, eid] = row
        if masses is not None:
            for name, value in masses.items():
                wp.to_torch(getattr(c.wm, name))[:n] = value[lo:hi].to(torch.float32)
            c.masses_varied = True
        elif c.masses_varied:
            for name, value in self._mass_rows()._constants(self.device).items():
                if name in _MASS_ARRAYS:
                    wp.to_torch(getattr(c.wm, name))[:] = value.to(torch.float32)
            c.masses_varied = False
        tilt = self._floor_tilt()
        if floor is not None:
            wp.to_torch(c.wd.geom_xmat)[:n, tilt[1]] = floor[lo:hi]
            c.floor_tilted = True
        elif c.floor_tilted:
            xmat = wp.to_torch(c.wd.geom_xmat)
            xmat[:, tilt[1]] = torch.eye(3, device=xmat.device, dtype=xmat.dtype)
            c.floor_tilted = False
        if c.reflection is not None and c.reflection.palette is not None:
            c.reflection.recolour(None if palette is None else palette[lo:hi])
        m, pr = self._model(), self._pristine()
        mr = wp.to_torch(c.wm.mat_rgba)
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                mr[:n, i, :3] = sty[f"material.{name}"][lo:hi]
        base = torch.as_tensor(pr["light_diffuse"], device=self.device, dtype=torch.float32)
        wp.to_torch(c.wm.light_diffuse)[:n] = torch.clamp(
            base.unsqueeze(0) * sty["light.diffuse_scale"][lo:hi].reshape(n, 1, 1), 0, 1
        )
