from __future__ import annotations

import numpy as np
import torch

from ...render import _resources as _RES
from ...render import cpu3d as C3
from ...render import mjwarp as MW
from ._model import _TC_CACHE, MJW_CHUNK, VISIBLE_GROUPS, _effective_alpha, _row_width, _slot_element
from .base import _DMCModel


class _TCtx:
    """Hold a captured MJWarp render graph and the per-world arrays it reads and writes."""

    __slots__ = ("wm", "wd", "rc", "bgra", "graph", "nworld", "res", "wp", "mjw", "textures", "reflection")


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
    for arr in {a for a, _ in slots}:  # a canonical model, so the context is a
        getattr(m, arr)[:] = pr[arr]  # function of the domain and nothing else
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
    batched = sorted({"mat_rgba", "light_diffuse"} | {a for a, _ in slots})
    c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in batched})
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    c.textures = c.reflection = None
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
        c.reflection = Reflection(c.wm, c.wd, c.rc, m)
        _tc_frame(c)
        wp.synchronize()
    import torch

    c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(int(nworld), int(res), int(res), 4)
    with wp.ScopedCapture() as cap:
        _tc_frame(c)
    c.graph = cap.graph
    _TC_CACHE.put(key, c)
    return c


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
        for arr in {a for a, _ in self._slots()}:  # the canonical model, as _tc_context
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

    def _cpu_poses(self, s: torch.Tensor, sc: C3.Scene):
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
        try:
            for i in range(B):
                for (arr, k), v in zip(slots, ep[i]):
                    getattr(m, arr).reshape(-1)[k] = v
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
        t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)  # noqa: E731
        return t(cam), t(xp), t(xm), t(lp), t(ld)

    def _render_cpu(self, s: torch.Tensor, sty, size: int, buf: torch.Tensor) -> None:
        import mujoco

        sc = self._cpu_scene(size)
        m, pr = self._model(), self._pristine()
        B = int(s.shape[0])
        cam, xp, xm, lp, ld = self._cpu_poses(s, sc)
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
        # An explicit instance override is used by controlled chunk sweeps.
        chunk = vars(self).get("CPU_RENDER_CHUNK", chunk)
        C3.render(sc, cam, xp, xm, lp, ld, mat, ldif, out=buf, chunk_size=chunk)

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
        # Small images can amortize uploads across more worlds; keep the existing large-image bound.
        capacity = 256 if size <= 84 else MJW_CHUNK
        worlds = min(capacity, 1 << max(0, B - 1).bit_length())
        c = self._context(size, worlds)
        for lo in range(0, B, worlds):
            hi = min(B, lo + worlds)
            self._upload(c, s[lo:hi], sty, lo, hi)
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

    def _upload(self, c: _TCtx, s: torch.Tensor, sty, lo: int, hi: int) -> None:
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
