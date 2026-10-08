from __future__ import annotations

import numpy as np
import torch

from ..._core.plate_texels import substitute as _substitute_texels
from ..._core.plate_texels import texel_key as _texel_key
from ..._core.plate_texels import texel_slots as _texel_slots
from ...render import raycast3d as R3
from ...spaces import Box, Dict
from ._model import (
    _GRID_SLOTS,
    _SCENE_CACHE,
    HIDDEN_GROUP,
    LIGHT_DIRECTIONAL,
    VISIBLE_GROUPS,
    _effective_alpha,
    _grid_defaults,
    _grid_masks,
    _mat_rgb,
    _mat_spec_shin,
    _slot_element,
    _tex_rgb,
)
from ._shading import _dot, _Lights, _norm, _Scene, _shade, _upsample_rgb
from ._tendon import _chain_capsules_batch, _find_tendon, _tendon_polyline
from .base import _DMCModel


class DMControlSuite(_DMCModel):
    """Render fixed-camera domains using static plates and analytic moving geometry.

    Plates store the lit background, its shadowed colour, and its unreflected
    surface. Subclasses declare the domain, task and camera. Shared model mutation
    during baking and kinematics requires serialized access."""

    # One ray per pixel by default; higher supersampling changes silhouette coverage.
    default_supersample = 1
    # Bound rays per chunk, including supersamples and intermediate shading buffers.
    RAY_BUDGET = 1 << 24

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        sc = self._scene()
        assert sc.model.nq == self.nq, f"{type(self).__name__}: nq changed under us"

    @classmethod
    def _grid_slots(cls):
        """Cache checker texel positions from canonical masks, independent of recolouring."""
        key = (cls.DOMAIN, cls.TASK, int(cls.default_size))
        if key in _GRID_SLOTS:
            return _GRID_SLOTS[key]
        import mujoco

        m = cls._model()
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
        if t < 0:
            _GRID_SLOTS[key] = None
            return None
        _GRID_SLOTS[key] = _texel_slots(m, t, _grid_defaults()[:2], masks=_grid_masks(m, t))
        return _GRID_SLOTS[key]

    def make_variation_space(self) -> Dict:
        """Add the checker’s two RGB colours to the model-derived appearance space.

        The source material file supplies their defaults. Texture substitution uploads
        into the existing GL context rather than creating another renderer."""
        space = super().make_variation_space()
        slots = self._grid_slots()
        if slots is None:
            return space
        d = slots["default"]
        return Dict(
            {
                **{k: space[k] for k in space},
                "floor": Dict({"color": Box(low=0.0, high=1.0, shape=(2, 3), init_value=d.tolist())}),
            }
        )

    @property
    def PLATE_PATHS(self) -> tuple[str, ...]:  # noqa: N802
        """Return every appearance leaf; any may affect static surfaces through shared materials."""
        return tuple(p for p, _, _ in self.variation_space.flat_layout())


    def _plate_key(self, size, variation):
        """Key texture colours by their stored uint8 bytes and other leaves to six decimals."""
        out = [(self.DOMAIN, self.TASK, int(self.CAMERA), int(size), str(self.device))]
        for p in self.PLATE_PATHS:
            if p == "floor.color":
                out.append(_texel_key(self._grid_slots(), self._var(variation, p).reshape(2, 3)))
            else:
                out.append(tuple(np.round(self._var(variation, p), 6)))
        return tuple(out)

    def _scene(self, size=None, variation=None) -> _Scene:
        size = int(size or self.size)
        key = self._plate_key(size, variation)
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            return hit
        return _SCENE_CACHE.put(key, self._bake(size, variation))

    def _apply_variation(self, m, pr, variation):
        """Restore canonical model values, then apply absolute appearance changes.

        Shape edits leave compiled collision bounds stale. They are restricted to
        noncolliding render targets; extending them to physics requires recomputing
        model constants or recompilation. Visibility is applied after shape changes."""
        import mujoco

        m.mat_rgba[:] = pr["mat_rgba"]
        m.geom_rgba[:] = pr["geom_rgba"]
        m.site_rgba[:] = pr["site_rgba"]
        m.mat_reflectance[:] = pr["mat_reflectance"]
        m.geom_type[:] = pr["geom_type"]
        m.geom_size[:] = pr["geom_size"]
        # groups too, so a bake that raises part-way cannot leave the movers hidden for the next one
        m.geom_group[:] = pr["geom_group"]
        m.site_group[:] = pr["site_group"]
        # Restore episode entries too: the previous kinematics call may have changed them.
        for arr in {a for a, _ in self._slots()}:
            getattr(m, arr)[:] = pr[arr]
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                m.mat_rgba[i, :3] = self._var(variation, f"material.{name}")
        m.cam_fovy[:] = pr["cam_fovy"] * float(self._var(variation, "camera.fovy_scale")[0])
        m.light_diffuse[:] = np.clip(pr["light_diffuse"] * float(self._var(variation, "light.diffuse_scale")[0]), 0, 1)
        m.light_specular[:] = pr["light_specular"]
        # Render-only shape edits: see the collision-bound restriction in this method’s contract.
        for gname, alts in self.SHAPE_KNOBS.items():
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            k = int(round(float(self._var(variation, f"{gname}.shape")[0])))
            if not 0 <= k < len(alts):
                raise ValueError(f"{type(self).__name__}: {gname}.shape = {k} is outside [0, {len(alts)})")
            t, s = alts[k]
            m.geom_type[gid] = int(t)
            m.geom_size[gid] = np.asarray(s, np.float64)
        # Hide the element rather than its material, which may also be worn by other elements.
        for kind, name, eids in self._hideable():
            # Styles store discrete visibility as float32; threshold manually supplied rows too.
            if float(self._var(variation, f"visibility.{name}")[0]) < 0.5:
                for eid in eids:  # one id for an element leaf, N for a body leaf
                    getattr(m, f"{kind}_rgba")[eid, 3] = 0.0

    @staticmethod
    def _visible(m, kind: str) -> list[int]:
        """Select elements with both a visible group and positive resolved alpha."""
        grp = m.geom_group if kind == "geom" else m.site_group
        n = m.ngeom if kind == "geom" else m.nsite
        alpha = _effective_alpha(m, kind)
        return [i for i in range(n) if int(grp[i]) in VISIBLE_GROUPS and alpha[i] > 0]

    def _movers(self, m) -> tuple[list[int], list[int]]:
        """Select visible geoms/sites moved by joints or episode-specific model entries.

        World-body landmarks can move between episodes and must stay out of plates.
        Body slots already enter through attached geoms; light IDs are not geometry IDs."""
        ep_g, ep_s = set(), set()
        for arr, i in self._slots():
            kind, eid = _slot_element(m, arr, i)
            if kind == "geom":
                ep_g.add(eid)
            elif kind == "site":
                ep_s.add(eid)
            # Body and light IDs belong to different namespaces; never treat them as site IDs.
        g = [i for i in self._visible(m, "geom") if int(m.geom_bodyid[i]) != 0 or i in ep_g]
        s = [i for i in self._visible(m, "site") if int(m.site_bodyid[i]) != 0 or i in ep_s]
        return g, s

    def _apply_floor(self, m, r, variation) -> None:
        """Assign floor texels absolutely and upload with the renderer’s GL context current.

        Both model and context retain the previous bake’s colours, so every bake
        must assign its own texture, including the default."""
        slots = self._grid_slots()
        if slots is None:
            return
        _substitute_texels(m, r, slots, self._var(variation, "floor.color").reshape(2, 3))

    def _bake(self, size: int, variation) -> _Scene:
        import mujoco

        env, m, d, pr, r, _ = self._bundle(size)
        self._apply_variation(m, pr, variation)
        self._apply_floor(m, r, variation)

        gmov, smov = self._movers(m)
        gstat = [i for i in self._visible(m, "geom") if i not in gmov]
        sstat = [i for i in self._visible(m, "site") if i not in smov]

        tendon = _find_tendon(m)

        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        opt = mujoco.MjvOption()  # the default: exactly what dm_control renders at
        if tendon is not None:
            # Tendons move but are not geoms; geom-group hiding alone leaves them in the plate.
            opt.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = 0

        def shot():
            r.update_scene(d, camera=self.CAMERA, scene_option=opt)
            return r.render().copy()

        m.geom_group[gmov] = HIDDEN_GROUP
        m.site_group[smov] = HIDDEN_GROUP
        plate = shot()
        r.update_scene(d, camera=self.CAMERA, scene_option=opt)
        cam = R3.Camera.from_mjv(r.scene.camera[0], r.scene.camera[1])
        lights = _Lights(r.scene)
        sl = lights.shadow_light()

        # plate 2: the shadow-casting light contributing only its ambient. This IS the colour of a
        # static surface the light does not reach, so nothing about MuJoCo's shadow MAP is modelled.
        if sl >= 0:
            head = 1 if int(m.vis.headlight.active) else 0
            mi = sl - head
            keep_d = np.array(m.light_diffuse, copy=True)
            keep_s = np.array(m.light_specular, copy=True)
            if 0 <= mi < m.nlight:
                m.light_diffuse[mi] = 0.0
                m.light_specular[mi] = 0.0
            plate_shadow = shot()
            m.light_diffuse[:] = keep_d
            m.light_specular[:] = keep_s
        else:
            plate_shadow = plate

        # plate 3: reflection removed, so MuJoCo's own compositing rule can be inverted
        keep_r = np.array(m.mat_reflectance, copy=True)
        m.mat_reflectance[:] = 0.0
        plate_norefl = shot()
        m.mat_reflectance[:] = keep_r

        m.geom_group[:] = pr["geom_group"]
        m.site_group[:] = pr["site_group"]

        sc = _Scene()
        sc.size = size
        sc.model, sc.data = m, d
        sc.cam, sc.lights, sc.shadow_light = cam, lights, sl
        sc.movers = (gmov, smov)
        # Capture visibility while this appearance is applied. A later bake mutates the
        # shared model, so lazy depth creation must not rediscover static membership.
        sc.statics = (gstat, sstat)
        sc.tendon = tendon
        sc.plate = torch.tensor(plate, device=self.device, dtype=torch.uint8)
        sc.plate_shadow = torch.tensor(plate_shadow, device=self.device, dtype=torch.uint8)
        sc.plate_norefl = torch.tensor(plate_norefl, device=self.device, dtype=torch.uint8)

        # Depth is evaluated lazily on the requested supersampling grid.
        sprims, spos, smat = self._prims(m, d, gstat, sstat)
        sc.static_t = {}
        sc.shadow_candidates = {}
        sc.rays = {}
        sc.const = {}
        # the reflective plane, if any: MuJoCo only mirrors PLANES
        sc.reflectance, sc.floor = 0.0, None
        for i in gstat:
            mid = int(m.geom_matid[i])
            if int(m.geom_type[i]) == R3.PLANE and mid >= 0 and float(m.mat_reflectance[mid]) > 0:
                sc.reflectance = float(m.mat_reflectance[mid])
                sc.floor = (
                    np.asarray(m.geom_size[i], np.float64),
                    np.asarray(d.geom_xpos[i], np.float64),
                    np.asarray(d.geom_xmat[i], np.float64).reshape(3, 3),
                )
                break
        # a length scale for ray offsets, taken from the scene rather than assumed: cartpole is 8 m
        # across and point_mass 0.6, so one absolute epsilon cannot serve both
        sc.eps = float(max(1e-6, 1e-5 * np.abs(spos).max())) if len(spos) else 1e-6

        sc.prims, _, _ = self._prims(m, d, gmov, smov, tendon)
        return sc

    def _prims(self, m, d, geoms, sites, tendon: _Tendon | None = None):
        """Build primitive attributes and poses from geoms, sites and tendon segments.

        Tendon count is fixed at bake time; kinematics replaces its placeholder poses,
        half-lengths and colours for each row."""
        kind, size, rgb, tex, spec, shin, pos, mat = [], [], [], [], [], [], [], []
        for i in geoms:
            mid = int(m.geom_matid[i])
            s, sh = _mat_spec_shin(m, mid)
            kind.append(int(m.geom_type[i]))
            size.append(np.asarray(m.geom_size[i], np.float64))
            rgb.append(_mat_rgb(m, mid, m.geom_rgba[i]))
            tex.append(_tex_rgb(m, mid))
            spec.append(s)
            shin.append(sh)
            pos.append(np.asarray(d.geom_xpos[i], np.float64))
            mat.append(np.asarray(d.geom_xmat[i], np.float64).reshape(3, 3))
        for i in sites:
            mid = int(m.site_matid[i])
            s, sh = _mat_spec_shin(m, mid)
            kind.append(int(m.site_type[i]))
            size.append(np.asarray(m.site_size[i], np.float64))
            rgb.append(_mat_rgb(m, mid, m.site_rgba[i]))
            tex.append(_tex_rgb(m, mid))
            spec.append(s)
            shin.append(sh)
            pos.append(np.asarray(d.site_xpos[i], np.float64))
            mat.append(np.asarray(d.site_xmat[i], np.float64).reshape(3, 3))
        if tendon is not None:
            for _ in range(tendon.nseg):
                kind.append(R3.CAPSULE)
                size.append(np.array([tendon.radius, 0.0, 0.0]))
                rgb.append(tendon.grey.copy())
                tex.append(np.ones(3))
                # matid -1 on this tendon, so MuJoCo's material defaults apply, the same ones
                # ``_mat_spec_shin`` returns for an unmaterialled geom
                spec.append(0.5)
                shin.append(0.5)
                pos.append(np.zeros(3))
                mat.append(np.eye(3))
        if not kind:
            prims = R3.Primitives(
                kind=np.zeros(0, int),
                size=np.zeros((0, 3)),
                mat_rgb=np.zeros((0, 3)),
                tex_rgb=np.zeros((0, 3)),
                spec=np.zeros(0),
            )
            prims.shininess = np.zeros(0)
            return prims, np.zeros((0, 3)), np.zeros((0, 3, 3))
        prims = R3.Primitives(
            kind=np.array(kind), size=np.stack(size), mat_rgb=np.stack(rgb), tex_rgb=np.stack(tex), spec=np.array(spec)
        )
        prims.shininess = np.array(shin)
        return prims, np.stack(pos), np.stack(mat)

    def _shadow_candidates(self, sc: _Scene, ss: int):
        """Cache static samples whose lit and shadowed plates differ.

        Skipping other samples is exact: their occlusion result cannot change a pixel."""
        key = (int(ss), bool(self.SHADOW_COMPACTION))
        hit = sc.shadow_candidates.get(key)
        if hit is not None:
            return hit
        a = _upsample_rgb(sc.plate, ss).reshape(-1, 3).to(torch.int16)
        b = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.int16)
        if self.SHADOW_COMPACTION:
            cand = (a != b).any(-1) & (self._static_t(sc, ss).view(-1) < R3.INF)
            idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        else:
            idx = torch.arange(a.shape[0], device=a.device)  # the pass this compaction replaced
        rows = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.float32)[idx]
        sc.shadow_candidates[key] = hit = (idx, rows.unsqueeze(0))
        return hit

    def _static_shadow_rays(self, sc: _Scene, ss: int):
        """Cache shadow origins, directions and limits for fixed surfaces and lights.

        Only occlusion by moving geometry depends on the render row."""
        key = ("shadow", int(ss), bool(self.SHADOW_COMPACTION))
        hit = sc.const.get(key)
        if hit is not None:
            return hit
        cand, dark = self._shadow_candidates(sc, ss)
        o, d = self._per_chunk_constants(sc, ss)[:2]
        N = (sc.size * int(ss)) ** 2
        st = self._static_t(sc, ss).view(1, N)
        p = o.expand(1, N, 3)[:, cand] + st[:, cand].clamp(max=1e6).unsqueeze(-1) * d.view(1, N, 3)[:, cand]
        L = sc.lights.to(self.device)
        sl = sc.shadow_light
        shape = (1,) * (p.dim() - 1) + (3,)
        if L["kind"][sl] == LIGHT_DIRECTIONAL:
            rd = (-_norm(L["dir"][sl].view(shape))).expand_as(p)
            lim = torch.full(p.shape[:-1], R3.INF, device=p.device)
        else:
            v = L["pos"][sl].view(shape) - p
            lim = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
            rd = v / lim.unsqueeze(-1)
        hit = sc.const[key] = (cand, dark, p + rd * sc.eps, rd, lim, st[:, cand] < R3.INF)
        return hit

    def _floor_rays(self, sc: _Scene, ss: int):
        """Cache reflected rays and unreflected colours at samples that hit the mirror."""
        key = ("floor", int(ss))
        hit = sc.const.get(key)
        if hit is not None:
            return hit
        o, d = self._per_chunk_constants(sc, ss)[:2]
        N = (sc.size * int(ss)) ** 2
        st = self._static_t(sc, ss).view(1, N)
        oo, dd = o.expand(1, N, 3), d.view(1, N, 3)
        fsize, fpos, fmat = self._floor_tensors(sc)
        # Compare ray parameters to identify the mirror; absolute hit-position tolerances
        # can miss the floor or include nearby geometry at different scene scales.
        t_f, n = R3.intersect(R3.PLANE, fsize.view(1, 1, 3), fpos.view(1, 1, 3), fmat.view(1, 1, 3, 3), oo, dd)
        on_plane = (t_f < R3.INF) & (st < R3.INF) & (torch.abs(t_f - st) <= 1e-4 * (1.0 + st))
        idx = torch.nonzero(on_plane.reshape(-1), as_tuple=False).reshape(-1)
        rd = dd - 2 * _dot(dd, n.expand_as(dd)).unsqueeze(-1) * n
        ro = oo + st.clamp(max=1e6).unsqueeze(-1) * dd + rd * sc.eps
        surf = sc.const.get(("norefl", int(ss)))
        if surf is None:
            surf = sc.const[("norefl", int(ss))] = (
                _upsample_rgb(sc.plate_norefl, ss).reshape(1, N, 3).to(torch.float32)
            )
        hit = sc.const[key] = (idx, ro[:, idx].contiguous(), rd[:, idx].contiguous(), surf[:, idx].contiguous())
        return hit

    def _static_t(self, sc: _Scene, ss: int) -> torch.Tensor:
        """Cast and cache static depth at the requested supersampling grid.

        Upsampling output-resolution depth would place edge samples on wrong surfaces."""
        hit = sc.static_t.get(int(ss))
        if hit is not None:
            return hit
        m, d = sc.model, sc.data
        # Use static membership captured at bake time, not the shared model’s current visibility.
        gstat, sstat = sc.statics
        import mujoco

        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        sprims, spos, smat = self._prims(m, d, gstat, sstat)
        R = sc.size * int(ss)
        t = R3.static_depth(sc.cam, sprims, spos, smat, R, R, self.device)
        sc.static_t[int(ss)] = t
        return t


    # Candidates only: conformance must validate each against render_scene_signature.
    RENDER_INVARIANT_DIMS: tuple[str, ...] = ()

    def render_invariant_dims(self) -> tuple[str, ...]:
        return tuple(self.RENDER_INVARIANT_DIMS)

    def render_scene_signature(self, states, variation=None) -> dict:
        """Return model attributes for the first compact render row and requested appearance.

        Only drawn geoms/sites are included. Conformance checks use these attributes
        to validate proposed invariant dimensions; declarations alone grant no exclusion.
        Live geometry types ensure a sphere-rotation exemption fails for a box style."""
        import mujoco

        m, d = self._bundle(self.size)[1], self._bundle(self.size)[2]
        slots = self._slots()
        pr = self._pristine()
        st = np.asarray(states, np.float64).reshape(-1, self.state_dim)[0]
        try:
            # A scene-cache hit does not restore the shared model; apply this appearance explicitly.
            self._apply_variation(m, pr, variation)
            for (arr, k), v in zip(slots, st[self.nq :]):
                getattr(m, arr).reshape(-1)[k] = v
            d.qpos[:] = st[: self.nq]
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            g = self._visible(m, "geom")
            si = self._visible(m, "site")
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}" for i in g] + [
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or f"site{i}" for i in si
            ]

            def stack(rows, dtype=np.float64):
                return np.asarray(rows, dtype) if len(rows) else np.zeros((0,), dtype)

            xpos = stack(
                [np.asarray(d.geom_xpos[i], np.float64) for i in g]
                + [np.asarray(d.site_xpos[i], np.float64) for i in si]
            )
            xmat = stack(
                [np.asarray(d.geom_xmat[i], np.float64).reshape(3, 3) for i in g]
                + [np.asarray(d.site_xmat[i], np.float64).reshape(3, 3) for i in si]
            )
            size = stack(
                [np.asarray(m.geom_size[i], np.float64) for i in g]
                + [np.asarray(m.site_size[i], np.float64) for i in si]
            )
            kind = np.asarray([int(m.geom_type[i]) for i in g] + [int(m.site_type[i]) for i in si], np.int64)
            rgba = stack(
                [np.asarray(m.geom_rgba[i], np.float64) for i in g]
                + [np.asarray(m.site_rgba[i], np.float64) for i in si]
            )
            matid = np.asarray([int(m.geom_matid[i]) for i in g] + [int(m.site_matid[i]) for i in si], np.int64)
            out = {
                # Keep geom IDs before site IDs, matching the primitive ordering.
                "geom_names": names,
                "geom_is_sphere": kind == R3.SPHERE,
                "geom_xmat": xmat,
                "geom_xpos": xpos,
                "geom_size": size,
                "geom_type": kind,
                "geom_rgba": rgba,
                "geom_matid": matid,

                "mat_rgba": np.array(m.mat_rgba, copy=True),
                "mat_reflectance": np.array(m.mat_reflectance, copy=True),
                "mat_specular": np.array(m.mat_specular, copy=True),
                "mat_shininess": np.array(m.mat_shininess, copy=True),
                # Texture assignments affect pixels and must participate in invariance checks.
                "mat_texid": np.array(m.mat_texid, copy=True),
                "light_pos": np.array(m.light_pos, copy=True),
                "light_dir": np.array(m.light_dir, copy=True),
                "light_diffuse": np.array(m.light_diffuse, copy=True),
                "light_specular": np.array(m.light_specular, copy=True),
                "light_ambient": np.array(m.light_ambient, copy=True),
                "light_castshadow": np.array(m.light_castshadow, copy=True),
                "cam_pos": np.array(m.cam_pos, copy=True),
                "cam_quat": np.array(m.cam_quat, copy=True),
                "cam_fovy": np.array(m.cam_fovy, copy=True),
            }
        finally:
            # Restore both episode entries and appearance after probing the shared model.
            for arr in {a for a, _ in slots}:
                getattr(m, arr)[:] = pr[arr]
            self._apply_variation(m, pr, None)
            # Lazy static depth reads live model arrays; discard scenes after restoring defaults
            # to prevent later depth initialization from using this probe’s appearance.
            _SCENE_CACHE.clear()
        return out


    def _fk(self, states: torch.Tensor, sc: _Scene):
        """Return per-row mover positions, rotations, sizes and colours as device tensors.

        Source kinematics runs on CPU. Tendons need mj_forward for constraint-dependent
        colour; their polylines are grouped by length for batched capsule construction.
        Restore episode model entries before returning, including on failure."""
        import mujoco

        gmov, smov = sc.movers
        td = sc.tendon
        nmov = len(gmov) + len(smov)
        P = nmov + (td.nseg if td is not None else 0)
        B = states.shape[0]
        st = states.detach().cpu().numpy().astype(np.float64)
        q, ep = st[:, : self.nq], st[:, self.nq :]
        xp = np.empty((B, P, 3))
        xm = np.empty((B, P, 3, 3))
        size = np.tile(np.asarray(sc.prims.size, np.float64), (B, 1, 1))
        rgb = np.tile(np.asarray(sc.prims.mat_rgb, np.float64), (B, 1, 1))
        m, d = sc.model, sc.data
        slots = self._slots()
        pr = self._pristine()
        gid = np.asarray(gmov, np.intp)
        sid = np.asarray(smov, np.intp)
        polys: list = []
        try:
            for i in range(B):
                # Episode entries use parent-body coordinates. Apply them before kinematics
                # rather than adding them to already transformed world poses.
                for (arr, k), v in zip(slots, ep[i]):
                    getattr(m, arr).reshape(-1)[k] = v
                d.qpos[:] = q[i]
                if td is None:
                    mujoco.mj_kinematics(m, d)
                else:
                    d.qvel[:] = 0
                    mujoco.mj_forward(m, d)
                # Write into preallocated rows instead of allocating fancy-indexed pose arrays.
                if gmov:
                    np.take(d.geom_xpos, gid, axis=0, out=xp[i, : len(gmov)])
                    np.take(d.geom_xmat, gid, axis=0, out=xm[i, : len(gmov)].reshape(-1, 9))
                if smov:
                    np.take(d.site_xpos, sid, axis=0, out=xp[i, len(gmov) : nmov])
                    np.take(d.site_xmat, sid, axis=0, out=xm[i, len(gmov) : nmov].reshape(-1, 9))
                if td is not None:
                    # Collect source polylines here; build capsule chains across rows below.
                    pts, trgb = _tendon_polyline(m, d, td)
                    polys.append(pts)
                    rgb[i, nmov:] = trgb
        finally:
            for arr in {a for a, _ in slots}:  # never leave a state row in the model
                getattr(m, arr)[:] = pr[arr]
        if td is not None:
            # Catenary branches produce different lengths; batch only matching point counts.
            by_len: dict = {}
            for i, pts in enumerate(polys):
                by_len.setdefault(len(pts), []).append(i)
            for npts, rows in by_len.items():
                idx = np.asarray(rows, np.intp)
                tp, tm, half = _chain_capsules_batch(np.stack([polys[i] for i in rows]), td.nseg)
                xp[idx, nmov:] = tp
                xm[idx, nmov:] = tm
                size[idx, nmov:, 1] = half
        t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)  # noqa: E731
        return t(xp), t(xm), t(size), t(rgb)


    @staticmethod
    def _cast(sc, o, d, xpos, xmat, sizes):
        """Return nearest mover distance, normal and primitive index for each ray."""

        return R3.cast_nearest(sc.prims.kind, sizes, xpos, xmat, o, d)

    @staticmethod
    def _occluded(sc, p, L, sl, xpos, xmat, sizes, skip=None, eps: float = 1e-6):
        """Return whether movers block rays from ``p`` to light ``sl``.

        ``skip`` excludes each ray’s originating convex primitive. Its front-facing
        surface cannot occlude itself; back-facing illumination is already zero."""
        shape = (1,) * (p.dim() - 1) + (3,)
        if L["kind"][sl] == LIGHT_DIRECTIONAL:
            d = (-_norm(L["dir"][sl].view(shape))).expand_as(p)
            lim = torch.full(p.shape[:-1], R3.INF, device=p.device)
        else:
            v = L["pos"][sl].view(shape) - p
            lim = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
            d = v / lim.unsqueeze(-1)
        o = p + d * eps
        return R3.any_blocking(sc.prims.kind, sizes, xpos, xmat, o, d, lim, skip=skip)


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
        """Render compact rows to uint8 frames, grouping styles to share baked plates."""
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
        offs, o = [], 0
        for p in self.PLATE_PATHS:
            w = int(sty.layout[p][1])
            offs.append((o, w))
            o += w
        for gi in range(len(uniq)):
            sub = np.flatnonzero(inv == gi)
            over = {p: uniq[gi][a : a + w] for p, (a, w) in zip(self.PLATE_PATHS, offs)}
            sc = self._scene(size, over)
            idx = torch.as_tensor(sub, device=self.device, dtype=torch.long)
            step = max(1, int(self.RAY_BUDGET // max(1, size * size * ss * ss)))
            for a in range(0, len(sub), step):
                part = idx[a : a + step]
                buf[part] = self._render_group(sc, s[part], size, ss)
        return buf.view(*lead, size, size, 3) if lead else buf

    # Window divisors, ordered from smallest footprint to largest.
    MOVER_TILES = (8, 6, 4, 3, 2, 1.5, 1.25)

    # Disable in differential tests to compare against full static-surface shadow checks.
    SHADOW_COMPACTION = True
    # Disable in differential tests to shade every candidate ray.
    COVER_COMPACTION = True

    def _mover_idx(self, sc: _Scene, xpos: torch.Tensor, size: int, ss: int):
        """Return a conservative mover window, or None to cast the full frame.

        Project bounds once, then choose the smallest fitting host-side window.
        Padding must preserve every sample that a mover can reach."""
        span = R3.screen_span(sc.cam, sc.prims, xpos, size, size, xpos.device)
        if span is None:
            return None
        x0, y0, w, h = span
        for div in self.MOVER_TILES:
            tile = max(1, int(size / div))
            if tile >= size or w > tile or h > tile:
                continue
            org = (x0.clamp(0, size - tile), y0.clamp(0, size - tile))
            return R3._expand_tile(org, tile, ss, size * ss, xpos.device)
        return None

    def _per_chunk_constants(self, sc: _Scene, ss: int):
        """Cache rays, plate colours, material parameters and lights per supersampling grid."""
        hit = sc.const.get(int(ss))
        if hit is None:
            dev = self.device
            R = sc.size * int(ss)
            o, d = sc.cam.rays(R, R, dev)
            hit = sc.const[int(ss)] = (
                o,
                d,
                _upsample_rgb(sc.plate, ss).reshape(1, -1, 3).to(torch.float32),
                torch.tensor(sc.prims.tex_rgb, device=dev, dtype=torch.float32),
                torch.tensor(sc.prims.spec, device=dev, dtype=torch.float32),
                torch.tensor(sc.prims.shininess, device=dev, dtype=torch.float32),
                sc.lights.to(dev),
            )
        return hit

    def _render_group(self, sc: _Scene, s: torch.Tensor, size: int, ss: int) -> torch.Tensor:
        dev = self.device
        B = s.shape[0]
        R = size * ss
        N = R * R
        o, d, plate_f, tex_rgb, spec, shin, L = self._per_chunk_constants(sc, ss)
        oo = o.expand(B, N, 3)
        dd = d.view(1, N, 3).expand(B, N, 3)

        xpos, xmat, sizes, mat_rgb = self._fk(s, sc)
        P = xpos.shape[1]
        sl = sc.shadow_light
        st = self._static_t(sc, ss).view(1, N).expand(B, N)
        # Tendon colour varies by row, so material lookup needs both row and primitive indices.
        bi = torch.arange(B, device=dev).unsqueeze(1)

        # Conservative windows exclude only rays movers cannot hit; fall back to a full
        # frame if none fits.
        widx = self._mover_idx(sc, xpos, size, ss)
        base = plate_f.expand(B, N, 3).clone()
        if widx is None:
            oo_w, dd_w = oo, dd
            t_w, n_w, i_w = self._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
            cover_w = t_w < st
        else:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
            oo_w, dd_w = oo.gather(1, w3), dd.gather(1, w3)
            t_w, n_w, i_w = self._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
            cover_w = t_w < st.gather(1, widx)

        # Compact the union of covered window positions across rows. Outside it, every
        # row selects the plate, so skipping shading cannot change the result.
        if self.COVER_COMPACTION:
            js = torch.nonzero(cover_w.any(0), as_tuple=False).reshape(-1)
            widx = js.view(1, -1).expand(B, -1) if widx is None else widx[:, js]
            cover_w, t_w, n_w, i_w = cover_w[:, js], t_w[:, js], n_w[:, js], i_w[:, js]
            oo_w, dd_w = oo_w[:, js], dd_w[:, js]
        if widx is not None:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
        pm_w = oo_w + t_w.clamp(max=1e6).unsqueeze(-1) * dd_w
        unlit_w = None
        if sl >= 0 and P > 1:
            unlit_w = self._occluded(sc, pm_w, L, sl, xpos, xmat, sizes, skip=i_w, eps=sc.eps)
        col_w = _shade(pm_w, n_w, -dd_w, L, mat_rgb[bi, i_w], spec[i_w], shin[i_w], unlit_w) * tex_rgb[i_w] * 255.0
        if sl >= 0:
            cand, dark = self._shadow_candidates(sc, ss)
            if cand.numel():
                stc = st[:, cand]
                ps = oo[:, cand] + stc.clamp(max=1e6).unsqueeze(-1) * dd[:, cand]
                occ = self._occluded(sc, ps, L, sl, xpos, xmat, sizes, eps=sc.eps) & (stc < R3.INF)
                base[:, cand] = torch.where(occ.unsqueeze(-1), dark, base[:, cand])
        if sc.reflectance > 0:
            base = self._reflect(sc, base, xpos, xmat, sizes, mat_rgb, tex_rgb, spec, shin, L, ss, bi)

        # Composite movers after background shadows and reflections; untouched rays retain the plate.
        if widx is None:  # nothing was restricted: no index to scatter through
            frame = torch.where(cover_w.unsqueeze(-1), col_w, base)
        else:
            frame = base.scatter_(1, w3, torch.where(cover_w.unsqueeze(-1), col_w, base.gather(1, w3)))
        frame = frame.view(B, size, ss, size, ss, 3).sum((2, 4)) / (ss * ss)
        return torch.clamp(torch.round(frame), 0, 255).to(torch.uint8)

    def _floor_tensors(self, sc: _Scene):
        """Cache the reflective plane’s size, position and rotation on the render device."""
        hit = sc.const.get("floor")
        if hit is None:
            fsize, fpos, fmat = sc.floor
            t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)  # noqa: E731
            hit = sc.const["floor"] = (t(fsize), t(fpos), t(fmat))
        return hit

    def _reflect(self, sc, base, xpos, xmat, sizes, mat_rgb, tex_rgb, spec, shin, L, ss, bi):
        """Replace mirror hits using ``r * mover_colour + (1-r) * unreflected_surface``.

        Blending onto the reflected plate would retain the old reflection underneath.
        Only hit samples are shaded; other pixels keep the existing background."""
        idx, ro, rd, surf = self._floor_rays(sc, ss)
        if idx.numel() == 0:
            return base
        B = base.shape[0]
        t_r, n_r, i_r = self._cast(sc, ro.expand(B, -1, 3), rd.expand(B, -1, 3), xpos, xmat, sizes)
        seen = t_r < R3.INF
        widx = idx.view(1, -1).expand(B, -1)
        # Shade only the union of mirror hits across rows; outside it the background is unchanged.
        if self.COVER_COMPACTION:
            js = torch.nonzero(seen.any(0), as_tuple=False).reshape(-1)
            widx, seen, ro, rd = widx[:, js], seen[:, js], ro[:, js], rd[:, js]
            surf, t_r, n_r, i_r = surf[:, js], t_r[:, js], n_r[:, js], i_r[:, js]
        pr = ro + t_r.clamp(max=1e6).unsqueeze(-1) * rd
        colr = _shade(pr, n_r, -rd, L, mat_rgb[bi, i_r], spec[i_r], shin[i_r], None) * tex_rgb[i_r] * 255.0
        r = float(sc.reflectance)
        val = r * colr + (1.0 - r) * surf
        w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
        # base is this render’s private canvas; in-place scatter avoids another full-frame copy.
        return base.scatter_(1, w3, torch.where(seen.unsqueeze(-1), val, base.gather(1, w3)))
