"""The VISUAL tier: a ``mode="trackcom"`` camera, so the whole scene goes to a 3-D backend every frame.

With the camera riding the body there is no static half to bake -- hide every mover and the remaining
frame still moves by 51-220/255 -- so the plate-and-cast design of ``suite`` has nothing to stand on
and the frame is MJWarp's raytrace (CUDA) or ``cpu3d``'s (CPU) of dm_control's own scene. These are
NOT the official pixels; every cell here declares what the difference measures
(``visual_fidelity``) and names the two mechanisms that carry it (``VISUAL_NOTE``).

One module because it is ONE renderer, exactly as ``suite`` is one module. The six cells differ only
in their declarations, which live in their own domain modules beside this one.
"""

from __future__ import annotations

import numpy as np
import torch

from ...render import _resources as _RES
from ...render import cpu3d as C3
from ...render import mjwarp as MW
from ._model import _TC_CACHE, MJW_CHUNK, VISIBLE_GROUPS, _effective_alpha, _row_width, _slot_element
from .base import _DMCModel


class _TCtx:
    """One CUDA-graph-captured MJWarp render context, plus the device views a render writes."""

    __slots__ = ("wm", "wd", "rc", "bgra", "graph", "nworld", "res", "wp", "mjw")


def _tc_context(cls, res: int, nworld: int) -> _TCtx:
    """One captured MJWarp context per (domain, task, resolution, worlds).

    Two things here are not optional and both are silent when omitted.

    **``com_pos`` between ``kinematics`` and ``camlight``.** These domains' camera is
    ``mjCAMLIGHT_TRACKCOM`` — it rides the body's centre of mass, which is the whole reason the plate
    design cannot serve them — and ``mujoco_warp.kinematics`` writes ``xpos``/``xquat``/``xmat``/
    ``xipos`` and NOT ``subtree_com``. Without the extra launch the camera never leaves the pose
    ``put_data`` uploaded and every state is drawn from it, with no error and a plausible-looking
    picture. ``MW.needs_com_pos`` answers the question from the model and is asserted below rather
    than assumed, because the failure has no other symptom.
    ``docs/research/mujoco_locomotion.md`` S6.2 measures the two frame loops: camera spread 0.0
    (frozen) against MuJoCo's own, and 0.0 distance FROM MuJoCo's once ``com_pos`` runs.

    **Graph capture.** Un-captured these five kernels run at roughly 1.4x the official renderer rather
    than 30x, because at this scene size the frame is launch-bound. A throughput number near 1.4x
    means the capture did not happen, not that the GPU is slow.
    """
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

    # The zero-alpha pre-pass. MJWarp's shader never reads alpha, so ``swimmer``'s six ``inertial``
    # geoms and six sites and ``fish``'s target and torso site would be painted OPAQUE over geometry
    # the reference does not draw at all. Left applied for the life of the context: MJWarp reads the
    # geometry once, at create_render_context, so restoring would not un-draw anything.
    MW.prepare_model(m, VISIBLE_GROUPS)

    c = _TCtx()
    c.wp, c.mjw, c.nworld, c.res = wp, mjw, int(nworld), int(res)
    d0 = mujoco.MjData(m)
    d0.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d0)
    batched = sorted({"mat_rgba", "light_diffuse"} | {a for a, _ in slots})
    c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in batched})
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # The rounding patch has to be visible while the megakernel's source is GENERATED (context
    # creation) and while it is COMPILED (first launch), which is why the warm-up frame is inside the
    # block. MJWarp packs float->uint8 with C's ``int()``, which truncates, and a GL pipeline rounds;
    # on Fetch that one rule was worth interior parity 0.791 -> 0.9988.
    with MW.gl_rounding():
        c.rc = MW.create_render_context(
            m,
            nworld=int(nworld),
            cam_res=(int(res), int(res)),
            render_rgb=True,
            cam_active=[i == int(cls.CAMERA) for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS),
            use_textures=True,
            use_shadows=True,
            render_skybox=True,
            **dict(cls.RENDER_FLAGS),
        )
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


class DMControlTrackCom(_DMCModel):
    """DM Control's body-tracking domains, drawn by MJWarp. VISUAL TIER — not the official pixels.

    These six domains put camera 0 in ``mode="trackcom"``: it rides the body's centre of mass, so the
    "static" half of the scene moves with the state and there is no plate to bake — hide every mover
    and the remaining frame still changes by 51-220/255 between states. ``DMControlSuite``'s design
    is therefore structurally unavailable here, and the alternative is a backend that draws the whole
    scene every frame. MJWarp is that backend, and its ``mjCAMLIGHT_TRACKCOM`` camera lands on
    MuJoCo's ``cam_xpos`` to 1e-8 metres, so the camera is not what the residual is.

    WHAT THE RESIDUAL IS
    --------------------

    Two named mechanisms, measured per surface off MuJoCo's own segmentation buffer (84 px, 32 states
    per domain, ``scratch/dmcontrol/probe_trackcom.py``):

      * **the speckled gradient skybox**, which is 48-89% of the frame on cheetah, hopper and walker.
        DM Control's ``skybox`` is an 800x4800 ``builtin="gradient"`` texture with ``mark="random"``,
        i.e. white specks scattered by MuJoCo's own RNG, and MJWarp samples it with a single
        unfiltered tap. Mean 4.1-4.7/255 with 89-92% of those pixels within 8, and it is what holds
        SSIM down — a speck one pixel out of place costs structure twice.
      * **the minified checker floor**, the same mechanism that decides
        ``docs/research/mujoco_locomotion.md``: in the far band MuJoCo's mipmap has averaged the
        checker into one grey (luminance spread 0.1 on humanoid) and MJWarp point-samples it at full
        contrast (9.7). Floor mean 4.3-8.2/255.

    Neither is closable here. The floor needs level-of-detail selection in ``mujoco_warp``'s render
    kernel, which needs ray differentials the kernel does not compute; nvdiffrast's mipmapped sampler
    closes it to 0.66-0.81 of far-band pixels (``docs/research/nvdiffrast.md``) and is not exactness
    either, is 3-5x slower, and cannot carry these scenes' MOVING lights, which are ``trackcom`` too.

    Supersampling is deliberately not offered. It was measured on the same mechanism for Gymnasium
    locomotion (``mujoco_locomotion.md`` S6.5): 16 samples per pixel take the far field's mean error
    from 100 to 8-14/255 and reach 0.6-1.4% exact, while whole-frame exactness FALLS on every domain,
    because a box average over 16 samples is not GL's mipmap pyramid and everything the renderer had
    right gets blurred.
    """

    fidelity_tier = "visual"
    parity_interior_min = None
    default_supersample = 1
    #: No camera knob. ``cam_fovy`` is not among the model arrays ``mujoco_warp.put_model`` will carry
    #: per world, so a per-environment field of view would need one render context per distinct value
    #: — and a variation that silently costs a context rebuild is worse than one that is not offered.
    CAMERA_KNOB = False
    #: No visibility knob either, and for the same shape of reason — a fact about MJWarp, not about
    #: these domains. ``mujoco_warp``'s shader never reads alpha: ``_tc_context`` hides zero-alpha
    #: elements through ``MW.prepare_model``, which moves them out of the ENABLED GEOM GROUPS, and
    #: ``create_render_context`` reads the geometry once per (domain, resolution, worlds) and then the
    #: five kernels are CUDA-graph captured. So visibility is fixed for the life of a context and a
    #: per-world value would need either one captured context per distinct combination — the same
    #: trade ``CAMERA_KNOB`` refuses — or an alpha-aware shader upstream in ``mujoco_warp``. Only
    #: ``DMCSwimmerSwimmer6`` draws a hideable element per episode in this tier, and it is a VISUAL
    #: cell; the parity cells that need the knob are all in ``DMControlSuite``. Recorded here rather
    #: than fixed because the fix is in a renderer this file does not own.
    VISIBILITY_KNOB = False

    #: Per-domain overrides for ``create_render_context``, folded into the cache key so two domains
    #: that disagree cannot share a context. Empty for every domain but swimmer; see
    #: ``SPECULAR_NOTE`` for what swimmer sets and why it is not set family-wide.
    RENDER_FLAGS: dict = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # DEVICE SELECTION IS HONEST. ``cuda`` gets MJWarp; ``cpu`` gets the CPU-native backend in
        # ``stable_envs/render/cpu3d.py`` -- the same scene, camera, materials and lighting law drawn
        # by an analytic caster over a baked skybox plate; anything else raises here rather than
        # rendering somewhere the caller did not ask for. Step 3's "cpu" rows for these cells were
        # MJWarp on the GPU behind a ``device="cpu"`` request (``DEVICE_NOT_HONOURED``); that path no
        # longer exists.
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

    def _context(self, size: int) -> _TCtx:
        return _tc_context(type(self), int(size), MJW_CHUNK)

    @classmethod
    def _used_materials(cls) -> list[str]:
        """``_DMCModel._used_materials`` restricted to GEOM wearers: neither MJWarp nor cpu3d draws
        a site, so a material worn only by sites cannot move a pixel on this tier and must not be a
        declared knob (``check_variations_do_something`` moves every leaf and fails when the frame
        does not follow). Reachable on exactly one cell: quadruped's ``eye`` material is worn by the
        two pupil sites alone. The other cells' sites are hidden (group 3/4 or alpha 0) and their
        lists are unchanged (asserted in tests/unit/test_cpu3d.py)."""
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

    # ---- the CPU backend ---------------------------------------------------

    def _cpu_scene(self, size: int) -> C3.Scene:
        """One ``cpu3d.Scene`` per resolution: rays, the baked skybox, the plane, the movers."""
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
        # the geoms MJWarp draws: visible groups, resolved alpha > 0 (its zero-alpha pre-pass), no
        # sites (it draws none); every non-plane geom is a "mover" whose world pose FK supplies,
        # static ones included -- their pose is simply the same every frame
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
            # A cell the CPU backend cannot draw refuses HERE, at construction, naming itself. It
            # used to render on the GPU behind the caller's back; the policy is an honest raise.
            # Since Step 7 no shipped trackcom cell reaches this branch by geom type (raycast3d
            # covers the ellipsoid too); swimmer is refused earlier, by name, through
            # CPU_UNCERTIFIED (no CPU certification). This stays for the next unsupported kind,
            # light type or camera mode Scene.__init__ refuses.
            raise ValueError(
                f"{type(self).__name__} ({self.DOMAIN}-{self.TASK}) has no CPU 3-D backend: {e}. "
                f"device='cuda' draws it through MJWarp; device='cpu' is refused rather than "
                f"silently served by the GPU"
            ) from e
        return cache.put(int(size), sc)

    def _cpu_poses(self, s: torch.Tensor, sc: C3.Scene):
        """``(B, state_dim)`` -> camera, mover, light poses through MuJoCo's own FK.

        ``mj_kinematics`` then ``mj_comPos`` then ``mj_camlight``: the trackcom camera and light read
        ``subtree_com``, which kinematics alone never writes -- the same trap ``_tc_context`` records
        for MJWarp's ``com_pos``. 3-10 us per state on the certification core.
        """
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
        C3.render(sc, cam, xp, xm, lp, ld, mat, ldif, out=buf)

    # ---- render -----------------------------------------------------------

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
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8. Style may differ per environment."""
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = s.shape[0]
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        if ss != 1:
            raise ValueError(
                f"{type(self).__name__}: supersample={ss} is not offered. MJWarp casts one ray per "
                f"pixel and a box average over sub-samples is not GL's mipmap pyramid: measured on "
                f"the identical mechanism it lowers the far floor's mean error and lowers whole-frame "
                f"exactness at the same time (docs/research/mujoco_locomotion.md S6.5)."
            )
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        if self.device.type == "cpu":
            self._render_cpu(s, sty, size, buf)
            return buf.view(*lead, size, size, 3) if lead else buf
        c = self._context(size)
        for lo in range(0, B, MJW_CHUNK):
            hi = min(B, lo + MJW_CHUNK)
            self._upload(c, s[lo:hi], sty, lo, hi)
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            # the BGRA -> RGB swap, ~3% of frame time and never optional: a byte view of MJWarp's
            # rgb_data is B,G,R,A, and reading it as RGB scores 0.0% exact on every domain
            buf[lo:hi] = c.bgra[: hi - lo][..., [2, 1, 0]]
        return buf.view(*lead, size, size, 3) if lead else buf

    @classmethod
    def _static_slot_rows(cls) -> dict:
        """``(kind, element id) -> (base row, [(state index, component), ...])`` for slot elements
        whose world transform ``mujoco_warp.kinematics`` never recomputes.

        This is the one place the per-world model upload is not enough, and it is silent when missed.
        ``mujoco_warp``'s ``_geom_local_to_global`` EARLY-OUTS on geoms and sites that hang off the
        world body -- their world pose cannot change under kinematics, so it is not recomputed -- and
        ``swimmer``'s per-episode ``target`` geom is exactly such an element. Writing the drawn
        ``geom_pos`` per world therefore reached nothing: measured, moving ``target_pos_x`` by 1.5 m
        changed the official frame by 216/255 and ours by **0**, while the per-episode LIGHT in the
        same state vector moved ours by 102. So the derived ``data`` row has to be written as well,
        which is free here because the world body's frame is the identity.
        """
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
        """Write this chunk's state and appearance into the context's PER-WORLD arrays."""
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
