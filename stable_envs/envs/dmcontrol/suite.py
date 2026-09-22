"""The PARITY tier: a fixed model camera, three baked plates, and an analytic ray-cast of the movers.

One module because it is ONE renderer. Twelve cells are drawn by it and they differ only in the
declarations their leaf class carries (``acrobot.py``, ``cartpole.py``, ...); nothing below is
per-domain, and a domain that needed something here would be a domain that does not belong in this
tier. See the package docstring for what splits the two tiers and for what the plates cover.

``_chain_capsules_batch`` is imported BY NAME from ``._tendon`` and is therefore one of this
module's globals, which is where ``tests/unit/test_dmcontrol_tendon.py`` substitutes it when it
checks the vectorised chain against the scalar reference (``catenary`` is substituted on
``._tendon``, which is where ``_tendon_polyline`` calls it from).
"""

from __future__ import annotations

import numpy as np
import torch

from ...render import raycast3d as R3
from ...spaces import Box, Dict
from .._plate_texels import substitute as _substitute_texels
from .._plate_texels import texel_key as _texel_key
from .._plate_texels import texel_slots as _texel_slots
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
    """The fixed-camera DM Control domains: three baked plates and an analytic ray-cast.

    ``DOMAIN``/``TASK``/``CAMERA`` are set by the generated subclasses.
    """

    #: ONE ray per pixel, and that is a measurement rather than a saving. The reference is MSAA'd at
    #: MuJoCo's ``offsamples=4`` default, so a hard silhouette is the wrong answer at a mover's edge,
    #: and this was 2 for exactly that reason. What was never checked is whether the PARITY CLAIM
    #: needs it, and it does not:
    #:
    #: MSAA moves SILHOUETTE pixels. The strong assertion in
    #: ``test_render_parity_with_official_renderer`` is about INTERIOR pixels -- those whose 3x3
    #: neighbourhood in the official frame is one flat colour -- and averaging four samples inside a
    #: flat region returns the byte one sample returns. Measured over 201 seeded draws of 64 states on
    #: every cell of this family, the count of disagreeing interior pixels is IDENTICAL at ss=1 and
    #: ss=2; on ``DMCReacherEasy`` at 256 px ss=1 is strictly better (76 against 1,634). The two
    #: assertions supersampling does improve were never close to their bounds: the all-pixel floor is
    #: 0.90 and the worst cell here sits at 0.9103 at ss=1 (and 0.9053 at ss=2 -- ss=1 is the BETTER
    #: one on 11 of 15 cells, because an aliased edge is bit-identical to MuJoCo wherever MuJoCo's own
    #: MSAA happened to be fully covered, while a blended edge almost never is), and the mean ceiling
    #: is 4.0/255 against a worst of 1.085.
    #:
    #: WHAT THIS GIVES UP, stated because nothing asserts it: the worst single pixel roughly doubles
    #: (``max_abs`` 52 -> 134 on cartpole, 95 -> 179 on finger), and mean |diff| rises 1.5-2.3x while
    #: staying a quarter of its bound. Both are reported by ``pixel_agreement`` and neither is a
    #: threshold. In exchange the family renders 1.4-1.9x faster at 84 px and 2.8-3.4x at 256 px,
    #: which takes five of these cells over their own stored-frame throughput bar; none cleared it at
    #: ss=2. Every other conformance check -- batch-size invariance, chunked-render agreement, order
    #: independence, render purity, input-form invariance, determinism -- is BYTE-IDENTICAL at ss=1.
    #:
    #: ``supersample=2`` remains available per call and per instance for a caller who wants the
    #: smaller worst-pixel; it is the same kwarg it always was and it is not deprecated. See
    #: the frozen measurement record for the 615-draw survey, the throughput table and the conformance diff.
    default_supersample = 1
    #: Rays cast in one pass, so a large batch bounds its own memory. Every quantity here is
    #: (B, H*W*ss^2, 3) float32, and the arithmetic is per-ray, so chunking changes nothing about the
    #: answer -- which is asserted by ``test_frame_i_is_independent_of_batch_size`` rather than
    #: assumed. At 84 px and ss=2 this is ~580 environments a pass, at 256 px 64.
    #:
    #: Raised from 1<<23 once the casts were fused and the shadow and reflection passes were bounded,
    #: because what the chunk count costs is no longer memory, it is PYTHON: every chunk re-issues the
    #: whole render as a couple of hundred torch calls and synchronises twice to pick its screen
    #: windows, and these families run at 100% of one core. Swept at 256 px, batch 1024, f/s and peak
    #: device memory:
    #:
    #:     budget  envs/chunk  Cartpole    FingerTurn   BallInCup    Pusher       Reacher
    #:     1<<23   32          5,038/1.1G  2,330/2.5G   1,995/1.3G   4,156/1.8G   7,599/1.5G
    #:     1<<24   64          6,530/1.9G  2,502/4.8G   2,170/2.4G   4,839/3.3G   9,458/2.7G
    #:     1<<25   128         7,332/3.6G  2,500/9.4G   2,094/4.5G   5,256/6.2G  10,761/5.1G
    #:     1<<26   256         7,870/7.0G  2,419/18.5G  2,103/8.6G   5,165/12.2G 11,529/10.0G
    #:
    #: 1<<24 is where the trade turns: it is 1.09-1.31x over 1<<23 for half the memory of 1<<25, and
    #: the two envs that gain nothing past it (finger, ball_in_cup) are exactly the two whose mover
    #: window declines at 256 px, so their reflection pass runs over the FULL frame and a chunk holds
    #: a dozen (B, H*W*ss^2, 3) tensors at once. Going further buys 12% on three cells and doubles the
    #: memory on all of them.
    RAY_BUDGET = 1 << 24

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        sc = self._scene()
        assert sc.model.nq == self.nq, f"{type(self).__name__}: nq changed under us"

    @classmethod
    def _grid_slots(cls):
        """This cell's ``grid`` texel slots, or ``None`` if it has no grid texture.

        Built from the PRISTINE compiled texture -- ``_bundle``'s model is shared by every instance
        and is mutated by every bake, so masks taken from the live texture would describe whatever
        the last bake left. The slots are positional (texel indices), so they stay correct however
        many colours have been written since.
        """
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
        """The base family's knobs plus ``floor.color``, the suite's checker texture.

        A TEXTURE leaf, and the only one here. Every other knob in this family is a material colour,
        a light scale, a camera scale or a visibility bit, all of which are read per frame -- which is
        why ``_bundle`` could keep ONE ``mujoco.Renderer`` per (domain, size) for the life of the
        process and say in its own comment that "only a TEXTURE edit would [need a second GL
        context], and none is offered". This is that edit, and the comment's conclusion does not
        follow from it: the escape it does not consider is an in-place texel write on the EXISTING
        context, which is what ``_plate_texels.substitute`` does. No second context is created and
        the one-Renderer-per-(domain, size) policy is unchanged.

        ``(2, 3)`` and in that order because that is the registered shape and order for
        the leaf, in all twelve of the DM Control wrappers: ``rgb1`` then ``rgb2`` of the builtin
        ``grid`` texture. The default is read from ``dm_control``'s ``common/materials.xml`` (see
        ``_grid_defaults``), which is the same file the registered ``init_value`` transcribes -- so the
        registered default is the compiled appearance BY CONSTRUCTION and the default plate cannot
        move. That is asserted rather than trusted: ``texel_key`` returns ``None`` at the default, so
        a default render keeps exactly the plate-cache key it had before this leaf existed.
        """
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
        """Style leaves that change the STATIC scene and therefore the three baked plates.

        EVERY leaf is one, and that is a property of the suite rather than laziness: DM Control
        shares a material between static and moving geoms (pendulum's ``decoration`` is on a moving
        base while point_mass's is on static walls), so there is no colour knob that can be applied
        to the movers alone. Declared here so the reproducibility suite can see, from the class, that
        this family re-bakes per appearance and cover it with one representative.
        """
        return tuple(p for p, _, _ in self.variation_space.flat_layout())

    # ---- scene bake -------------------------------------------------------

    def _plate_key(self, size, variation):
        """Every plate leaf, rounded -- except the texture leaf, which is keyed on the BYTES.

        ``floor.color`` is quantised rather than rounded because ``int(255 * v)`` is what MuJoCo
        actually stores: 0.1 and 0.1003 compile to the same texture and must share one bake, and at
        the registered default ``texel_key`` returns ``None``, which is exactly the key this tuple
        carried before the leaf existed. A default render therefore hits a cache that was warm before
        this patch and bakes nothing extra.
        """
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
        """Absolute assignment from the pristine snapshot, never an increment, so baking B after A
        gives the same model as baking B first. ``cam_fovy`` and the light scale are MULTIPLIERS and
        are exactly the two that would otherwise compound."""
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
        # and the episode-drawn arrays, for the same reason one rung up: ``_fk`` writes a STATE ROW
        # into them, and a bake that ran after a render would otherwise be taken at whatever target
        # position the last environment of the last batch happened to hold.
        for arr in {a for a, _ in self._slots()}:
            getattr(m, arr)[:] = pr[arr]
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                m.mat_rgba[i, :3] = self._var(variation, f"material.{name}")
        m.cam_fovy[:] = pr["cam_fovy"] * float(self._var(variation, "camera.fovy_scale")[0])
        m.light_diffuse[:] = np.clip(pr["light_diffuse"] * float(self._var(variation, "light.diffuse_scale")[0]), 0, 1)
        m.light_specular[:] = pr["light_specular"]
        # SHAPE, before visibility so that hiding still wins. Absolute assignment from the pristine
        # arrays restored above, so baking B after A gives the same model as baking B first.
        #
        # WHAT THIS LEAVES STALE, recorded rather than discovered later: ``geom_rbound`` and
        # ``geom_aabb`` are COMPILED from geom_type/geom_size and MuJoCo does not recompute them when
        # the arrays are written, so after a shape change they describe the compiled shape. That is
        # harmless for every consumer here and it is worth saying why rather than asserting it.
        # RENDERING does not read them: ``mjv_updateScene`` builds an ``mjvGeom`` from type, size and
        # the xpos/xmat this file supplies, and ``_prims`` reads the same two arrays directly, so the
        # ray-caster's own bound (``R3.screen_span``'s bounding sphere from ``sc.prims.size``) is
        # rebuilt at every bake. COLLISION does read rbound, through the broadphase -- but the only
        # geom any ``SHAPE_KNOBS`` entry touches is a ``ghost``-class target with ``contype=0
        # conaffinity=0``, and nothing in this file ever steps the physics: ``_fk`` runs
        # ``mj_kinematics`` (or ``mj_forward`` for the tendon domains, which have no shape leaf).
        # A shape leaf on a COLLIDING geom would need ``mj_setConst``/recompilation, and would be
        # changing the dynamics as well as the picture, which is not what an appearance knob is.
        for gname, alts in self.SHAPE_KNOBS.items():
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            k = int(round(float(self._var(variation, f"{gname}.shape")[0])))
            if not 0 <= k < len(alts):
                raise ValueError(f"{type(self).__name__}: {gname}.shape = {k} is outside [0, {len(alts)})")
            t, s = alts[k]
            m.geom_type[gid] = int(t)
            m.geom_size[gid] = np.asarray(s, np.float64)
        # VISIBILITY, last, and expressed the way MuJoCo expresses it: the element's own rgba alpha.
        # Everything downstream already reads that — ``_effective_alpha`` resolves it, ``_visible``
        # drops a zero-alpha element, ``_movers`` therefore never lists it, and the three plate shots
        # are taken by MuJoCo itself, which skips it too. So the knob needs no code in the renderer;
        # it needed a way IN. Zeroing the element's own alpha rather than its MATERIAL's is the finer
        # of the two instruments and measurably matters: on finger-turn the ``target`` material is worn
        # by both the ``target`` and ``tip`` sites, so a material-alpha knob hides 58 px where the
        # element knob hides 43 (84 px). On reacher
        # the target geom is the material's only wearer and the two coincide exactly, which is what
        # makes this a faithful reading of the registered ``render_target = 0``.
        for kind, name, eids in self._hideable():
            # ``>= 0.5`` rather than ``!= 0``: the value arrives as a float32 style column, and a
            # Discrete leaf only ever holds 0 or 1, so the threshold is a guard on a caller who writes
            # a style row by hand rather than a semantic.
            if float(self._var(variation, f"visibility.{name}")[0]) < 0.5:
                for eid in eids:  # one id for an element leaf, N for a body leaf
                    getattr(m, f"{kind}_rgba")[eid, 3] = 0.0

    @staticmethod
    def _visible(m, kind: str) -> list[int]:
        """Drawn geoms or sites: in a visible GROUP and with a non-zero resolved ALPHA.

        Both conditions, because the two filters hide different things and either alone is silent.
        Group hides ``acrobot``'s ``target`` and ``finger``'s ellipsoids; alpha hides ``finger``'s
        ``target`` and ``tip`` sites, ``fish``'s ``target`` geom, and ``swimmer``'s six ``inertial``
        geoms and six sites — every one of them written by ``initialize_episode`` rather than by the
        XML, which is why ``_bundle`` runs the episodes before this is ever read.
        """
        grp = m.geom_group if kind == "geom" else m.site_group
        n = m.ngeom if kind == "geom" else m.nsite
        alpha = _effective_alpha(m, kind)
        return [i for i in range(n) if int(grp[i]) in VISIBLE_GROUPS and alpha[i] > 0]

    def _movers(self, m) -> tuple[list[int], list[int]]:
        """``(geom ids, site ids)`` whose pose depends on the STATE: off the world body, or drawn.

        Two sources, and the second is what lets ``reacher`` and ``finger-turn`` ship at all:

          * anything on a body other than the world's, whose pose is a function of ``qpos``. Sites are
            in here because they are DRAWN at the default ``MjvOption`` and several domains hang one
            off a moving body — hiding only the moving GEOMS left acrobot's plate varying by 28/255;
          * anything the EPISODE draws. ``reacher``'s target geom and ``finger-turn``'s target site
            sit on the WORLD body, so the first rule calls them static and they go into the plate,
            which is exactly why the plate was constant within an episode (0/255) and not across them
            (215/255 on reacher, 224/255 on finger-turn). Their position is in the state, so they are
            movers, and the plate is then 0/255 across episodes too.

        Treating them as movers rather than re-baking per draw is the difference between one bake per
        appearance and one bake per episode: a batch of 1024 states from 1024 episodes would otherwise
        cost 3072 MuJoCo renders.

        A BODY OR LIGHT SLOT CONTRIBUTES NOTHING HERE, and it is skipped explicitly rather than falling
        through. The second rule exists only to rescue an element the FIRST rule misclassifies, and the
        first rule misclassifies exactly one thing: an element hanging off the WORLD body whose pose the
        episode nevertheless moves (``reacher``'s target geom, ``finger-turn``'s target site). A body
        slot cannot be in that position -- ``manipulator``'s ``target_ball`` is its own body, so every
        geom on it already has ``geom_bodyid != 0`` and the first rule takes it -- and a light is not
        geometry at all. Letting them fall through was the bug this branch shipped and review caught:
        ``(ep_g if kind == "geom" else ep_s).add(eid)`` filed manipulator's BODY id 11 into the SITE
        set against ``nsite = 9``, which was inert only because no site happened to carry that id. An
        id-space confusion that is inert by arithmetic accident is a defect, not a style point: a
        domain whose drawn body id collided with a real site id would silently promote an unrelated
        site to a mover and take it out of the plate.
        """
        ep_g, ep_s = set(), set()
        for arr, i in self._slots():
            kind, eid = _slot_element(m, arr, i)
            if kind == "geom":
                ep_g.add(eid)
            elif kind == "site":
                ep_s.add(eid)
            # body / light: see the docstring -- different id space, and nothing to rescue
        g = [i for i in self._visible(m, "geom") if int(m.geom_bodyid[i]) != 0 or i in ep_g]
        s = [i for i in self._visible(m, "site") if int(m.site_bodyid[i]) != 0 or i in ep_s]
        return g, s

    def _apply_floor(self, m, r, variation) -> None:
        """Put one ``floor.color`` into this (domain, size)'s texture and onto its GL context.

        Absolute, from the pristine masks, on EVERY bake, with no "it already looks like that"
        shortcut -- the same rule ``_apply_variation`` follows one line above, and for the same
        reason: the model and the renderer are shared by every instance of this class and carry
        whatever the last bake left, so a bake that skipped the write would inherit a previous
        caller's floor. That is not hypothetical here, because the plate cache is keyed on the
        appearance and a MISS is the only thing that reaches this function: two misses in a row with
        different colours are the normal case.

        ``substitute`` makes the renderer's own GL context current before uploading. This family
        keeps one ``Renderer`` per (domain, SIZE) and a process rendering at two sizes therefore
        holds two live contexts; GL texture names are per-context, so an upload under the wrong one
        silently overwrites an unrelated texture object in it. Measured exactly that on
        gpu2264 in the OGB port -- two 128 px plates differing by 219/255, read as "the plate depends
        on the state" -- before the ``make_current`` line existed. It is the reason this goes through
        the shared helper instead of two lines here.
        """
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
            # The tendon is a MOVER and has to leave the plate with the rest of them, and it is the
            # one mover that cannot be hidden by a geom group because it is not a geom. Measured:
            # ball_in_cup's plate varies by 97/255 over 0.78% of the frame with every moving geom and
            # site hidden, and by 0/255 once mjVIS_TENDON goes off as well
            # . That 97/255 is the whole of what
            # ``docs/research/dmcontrol.md`` S8 recorded as this domain's blocker.
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
        # The STATIC set, recorded here rather than re-derived later. ``_static_t`` used to recompute
        # it from the LIVE shared model at render time, and the live model carries whatever the LAST
        # bake applied -- so a scene baked with an element hidden, whose static depth was first cast
        # after some other appearance had been baked, saw that element visible again, did not find it
        # in ``sc.movers``, and promoted it to a STATIC primitive at whatever ``geom_pos`` ``_fk`` had
        # last written. Measured on CPU over the 64 frozen states: ``render(hidden, ss=2)`` alone
        # against ``render(hidden) -> render(other appearance) -> render(hidden, ss=2)`` differed by
        # 18 px / max 215 on DMCReacherHard and 88 px / max 140 on DMCFingerTurnHard. It needed a
        # SUPERSAMPLE change between the calls to show, because ``static_t`` is cached per (scene, ss)
        # and the default-ss path had always cached it during the bake's own render.
        #
        # This is the same live-read class as the ``_used_materials`` fix in this patch's first
        # commit, one call site further on: anything a scene is a function of must be read while that
        # scene's appearance is ON the model, which is here and nowhere else.
        sc.statics = (gstat, sstat)
        sc.tendon = tendon
        sc.plate = torch.tensor(plate, device=self.device, dtype=torch.uint8)
        sc.plate_shadow = torch.tensor(plate_shadow, device=self.device, dtype=torch.uint8)
        sc.plate_norefl = torch.tensor(plate_norefl, device=self.device, dtype=torch.uint8)

        # the static depth is NOT cast here: it is needed at the supersampled grid, which depends on a
        # render-time argument, and `check_style_is_state` bakes one scene per sampled appearance --
        # so casting a full frame per bake would be paid 16 times for a buffer nothing had asked for.
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
        """``(Primitives, xpos (P,3), xmat (P,3,3))`` for a set of geoms AND sites, then the tendon.

        Sites carry their own type/size/rgba arrays, so they go through a parallel path rather than
        being pretended into geoms; the tendon's capsules go through a third, because they are not in
        the model at all — MuJoCo synthesises them per frame from the wrap points and the tendon's
        length. Their poses and half-lengths here are placeholders overwritten by ``_fk``; what is
        fixed at bake time is their COUNT, which is what lets the mover set stay a constant.
        """
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
        """``(idx, plate_shadow_rows)``: the STATIC samples a mover's shadow could change, cached.

        Exact rather than approximate, and that is the whole reason it is allowed. A shadowed sample
        selects ``plate_shadow`` instead of ``plate``, so wherever the two plates carry the SAME
        colour the occlusion test cannot change the frame whatever it returns; and a ray that hits no
        static surface has nothing to shadow, which the ``st < INF`` term in the composite already
        said. Restricting the SET the test runs on therefore cannot change a VALUE — every ray's
        answer is computed from that ray alone.

        This is ``mujoco_locomotion``'s ``_shadow_candidates`` verbatim in intent; DM Control simply
        never had it, and its shadow pass was still starting from every sample in the frame.
        """
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
        """``(idx, dark, o, d, lim, valid)`` for the shadow test on static surfaces, cached.

        THE SHADOW RAY IS A FUNCTION OF THE SCENE, NOT OF THE STATE. It starts at ``o + t*d`` for the
        STATIC ``t``, points at a light that does not move, and ends at that light — so its origin,
        direction and length are the same on every frame, and the only thing a state changes is
        whether a mover happens to be in the way. Recomputing them per chunk cost a dozen elementwise
        passes over a (B, samples, 3) tensor to rebuild numbers that were already known; what is left
        is the occlusion query itself, which is one fused kernel.
        """
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
        """``(idx, o, d, surf)``: the REFLECTED ray at every sample that shows the mirror. Cached.

        The same observation as ``_static_shadow_rays`` and a bigger payoff, because this prologue was
        the largest single term left in a DM Control chunk (37-53%, more than every ray cast put
        together). Which samples are the mirror, where the eye ray meets it, and which way it bounces
        are all fixed by the camera and the floor; only what the bounced ray HITS depends on the
        state. So the plane intersection, the ``|t_f - st|`` test, the hit point and the reflection
        are computed once per (scene, supersample) and the per-frame cost is one fused cast.

        This also retires the mirror WINDOW. A rectangle around the movers' reflected bound measured
        44-64% of the frame at 256 px — the reflections of a spread-out body are spread out — so it
        was bounding the pass by less than the floor mask does, at the cost of a projection and a
        synchronisation per chunk.
        """
        key = ("floor", int(ss))
        hit = sc.const.get(key)
        if hit is not None:
            return hit
        o, d = self._per_chunk_constants(sc, ss)[:2]
        N = (sc.size * int(ss)) ** 2
        st = self._static_t(sc, ss).view(1, N)
        oo, dd = o.expand(1, N, 3), d.view(1, N, 3)
        fsize, fpos, fmat = self._floor_tensors(sc)
        # WHICH samples are the mirror is answered by re-casting the plane and comparing ray
        # parameters, not by testing whether the hit point lies in the plane: at cartpole's 8 m scale
        # a float32 hit point carries ~1e-6 of slack and any absolute tolerance either misses the
        # floor or claims the rails as well.
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
        """The static ray parameter at the SUPERSAMPLED grid, cached per (scene, ss).

        Cast rather than upsampled from the output-resolution buffer. The shadow test starts from
        ``o + t*d`` at every sample, so a nearest-neighbour ``t`` would put the sub-samples around a
        silhouette on the wrong surface and scatter shadow pixels along every static edge.
        """
        hit = sc.static_t.get(int(ss))
        if hit is not None:
            return hit
        m, d = sc.model, sc.data
        # From the BAKE, never from the live model -- see ``_bake``. The shared ``MjModel`` carries
        # the last-baked appearance, so re-deriving the static set here silently used a different
        # scene's visibility.
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

    # ---- render-invariance evidence --------------------------------------

    #: State dimensions this cell OFFERS to ``check_state_dims_matter`` as render-invariant. Declaring
    #: one excludes NOTHING by itself: the check validates every candidate against the compiled model
    #: through ``render_scene_signature`` and refuses any whose perturbation moves anything but the
    #: orientation of SPHERE geoms. Set per cell in ``_make_classes``; empty everywhere else.
    RENDER_INVARIANT_DIMS: tuple[str, ...] = ()

    def render_invariant_dims(self) -> tuple[str, ...]:
        return tuple(self.RENDER_INVARIANT_DIMS)

    def render_scene_signature(self, states, variation=None) -> dict:
        """Everything a frame is a function of, for ONE state, read off the model after kinematics.

        The evidence half of the render-invariance contract: this supplies MODEL FACTS and
        ``conformance._validate_render_invariant`` owns the RULE, so there is one definition of what
        the exclusion means and it lives with the check that grants it.

        Built the way ``_fk`` builds a frame -- the episode slots written into the model, then
        ``mj_kinematics`` -- so what is returned is the scene the renderer would actually draw and not
        a restatement of the state.

        ONLY DRAWN ELEMENTS ARE REPORTED, through the same ``_visible`` census the bake and the mover
        pass use (a visible GROUP and a non-zero resolved ALPHA). That is not a loosening, it is the
        definition of the render-side scene: an element MuJoCo does not draw cannot reach a pixel, so
        including it would let an invisible element veto a claim about visible ones. It is also the
        difference between a correct refusal and a wrong one -- manipulator's ``ball`` and
        ``target_ball`` bodies each carry a group-3 site that rotates with them, and comparing those
        refused the exclusion for a reason that has nothing to do with the picture.

        Geoms and sites are concatenated into ONE primitive list, exactly as ``_prims`` builds the
        mover set, so a drawn spherical SITE gets the same treatment as a drawn spherical geom rather
        than a stricter one by accident.

        ``prim_is_sphere`` is read from the LIVE type arrays, which is what makes the exclusion
        evaporate under ``target_ball.shape = 0``: with a BOX target the quaternion slots rotate a
        non-sphere, the validation refuses, and the dimensions are live again.
        """
        import mujoco

        m, d = self._bundle(self.size)[1], self._bundle(self.size)[2]
        slots = self._slots()
        pr = self._pristine()
        st = np.asarray(states, np.float64).reshape(-1, self.state_dim)[0]
        try:
            # APPLY THE APPEARANCE HERE rather than reading it off ``_scene(variation)``. The scene
            # cache means a hit does NOT re-bake, so the shared ``MjModel`` carries whatever the LAST
            # bake applied -- and this function reads ``geom_type``/``geom_size``/``geom_rgba`` off
            # that model. Asking for the default scene after a box-target scene had been baked
            # therefore reported the BOX, and the validation silently refused a claim it should have
            # granted. Measured while writing this: ball_y excluded, then the two quaternion dims
            # refused, purely because of call ORDER. Same live-model class as the ``_used_materials``
            # and ``_static_t`` defects this file already records; ``_apply_variation`` restores from
            # the pristine snapshot first, so this is absolute rather than incremental.
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
                # the DRAWN primitive list, geoms then sites, as _prims assembles it
                "geom_names": names,
                "geom_is_sphere": kind == R3.SPHERE,
                "geom_xmat": xmat,
                "geom_xpos": xpos,
                "geom_size": size,
                "geom_type": kind,
                "geom_rgba": rgba,
                "geom_matid": matid,
                # and everything else a pixel is a function of
                "mat_rgba": np.array(m.mat_rgba, copy=True),
                "mat_reflectance": np.array(m.mat_reflectance, copy=True),
                "mat_specular": np.array(m.mat_specular, copy=True),
                "mat_shininess": np.array(m.mat_shininess, copy=True),
                # mat_texid too: a texture swap is a colour change on every geom wearing the material,
                # and the rule REFUSES a signature that omits any frozen field rather than skipping it
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
            # never leave a probe's state row OR its appearance on the shared model -- the same rule
            # ``_fk`` follows for the state, extended to the appearance this function now applies
            for arr in {a for a, _ in slots}:
                getattr(m, arr)[:] = pr[arr]
            self._apply_variation(m, pr, None)
            # AND DROP THE SCENE CACHE, which is the part that is easy to miss. ``_static_t`` is cast
            # LAZILY, per (scene, supersample), and it re-derives its primitives from the LIVE model --
            # the hazard ``_bake`` documents. A cached scene whose static depth had not yet been cast
            # would therefore be finished against whatever appearance this probe last left behind.
            # Today the bake is the only other writer and it re-applies its own appearance first, so
            # clearing the cache restores exactly the invariant that held before this method existed:
            # the first ``_static_t`` of a scene runs while that scene's appearance is on the model.
            # The cost is a re-bake per probed dimension inside one conformance check, which is
            # milliseconds, and the alternative is a silent wrong static depth.
            _SCENE_CACHE.clear()
        return out

    # ---- forward kinematics ----------------------------------------------

    def _fk(self, states: torch.Tensor, sc: _Scene):
        """``(B, state_dim)`` -> per-row mover poses, sizes and colours.

        ``(xpos (B,P,3), xmat (B,P,3,3), sizes (B,P,3), rgb (B,P,3))``. Three of the four are constant
        across the batch on every domain but ``ball_in_cup``, and they are still returned per row
        because the tendon's capsules change LENGTH and COLOUR with the state and there is no useful
        way to say that only sometimes.

        A pure function of the state, and a Python loop over the batch. On the domains without a
        tendon that loop is cheap -- 17 ms of cartpole's 232 ms at B=1024 -- and what costs the rest
        is that three of the ray passes are FULL-FRAME: the movers can be tiled (they are 0.2-3.0% of
        the frame), but the shadow test starts from every STATIC pixel and the reflection test from
        every floor pixel. Bounding those two to the movers' light-space and screen-space footprints
        is still the next thing worth doing here.

        ON ``ball_in_cup`` THE LOOP WAS THE BOTTLENECK, and this docstring used to say it was not.
        Measured on a 3090 at batch 4096: the render was 1,314 ms of which the card did 66.7 ms
        (5.1% utilised) and this function was 97%. Inside it, ``mj_forward`` -- which the paragraph
        below blames, correctly, for being necessary -- was 3%, and the tendon's pure-Python/numpy
        geometry was 78%: ``_chain_capsules`` once per environment (512 ``numpy.cross`` and 768
        ``numpy.linalg.norm`` dispatches on ``(28,3)`` arrays at batch 256) and ``catenary``'s
        29-point interior loop once per environment. Both are array code with no batch axis, so the
        loop now collects POLYLINES and ``_chain_capsules_batch`` builds every chain at once, grouped
        by point count. 3,272 -> 9,816 fps at batch 4096, bit-identical, and the row goes
        GPU_PARITY/BELOW_G4 -> GPU_FAST. A batched GPU forward kinematics is still not the answer;
        giving the numpy a batch axis was.

        A scene with a tendon runs ``mj_forward`` rather than ``mj_kinematics``, and pays for it: the
        colour of a taut tendon is its limit constraint's impedance, which only exists once the
        constraint set has been built. ``ball_in_cup`` is 4 dof and 6 geoms, so the extra work really
        is microseconds (3.50 ms over 256 environments); a domain where it were not could not use
        this path.
        """
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
                # The episode's draw goes into the MODEL and then through the same kinematics as the
                # joints, rather than being added to the mover's pose afterwards. That is not tidiness:
                # ``geom_pos`` is expressed in the parent BODY's frame, so a mover the episode draws on
                # a rotated body would land in the wrong place under any shortcut, and running it
                # through ``mj_kinematics`` is what makes the widened state mean exactly what it means
                # to ``dm_control``.
                for (arr, k), v in zip(slots, ep[i]):
                    getattr(m, arr).reshape(-1)[k] = v
                d.qpos[:] = q[i]
                if td is None:
                    mujoco.mj_kinematics(m, d)
                else:
                    d.qvel[:] = 0
                    mujoco.mj_forward(m, d)
                # ``np.take(..., out=)`` rather than ``d.geom_xpos[ids]``: fancy indexing ALLOCATES a
                # fresh array per call and the reshape allocates a second, where this writes the same
                # bytes straight into the destination row. Identical values, no temporaries; the same
                # change ``ogbench._fk`` already documents having made.
                if gmov:
                    np.take(d.geom_xpos, gid, axis=0, out=xp[i, : len(gmov)])
                    np.take(d.geom_xmat, gid, axis=0, out=xm[i, : len(gmov)].reshape(-1, 9))
                if smov:
                    np.take(d.site_xpos, sid, axis=0, out=xp[i, len(gmov) : nmov])
                    np.take(d.site_xmat, sid, axis=0, out=xm[i, len(gmov) : nmov].reshape(-1, 9))
                if td is not None:
                    # ONLY the polyline here. The chain of capsules it becomes is the same arithmetic
                    # for every row and is built once for the whole batch below -- see
                    # ``_chain_capsules_batch``.
                    pts, trgb = _tendon_polyline(m, d, td)
                    polys.append(pts)
                    rgb[i, nmov:] = trgb
        finally:
            for arr in {a for a, _ in slots}:  # never leave a state row in the model
                getattr(m, arr)[:] = pr[arr]
        if td is not None:
            # Grouped by point count, because the three branches of ``mjv_catenary`` produce polylines
            # of different lengths and a batch holds states from all three at once. ball_in_cup has
            # exactly three possible counts, so this is at most three batched calls in place of B.
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

    # ---- ray-cast helpers -------------------------------------------------

    @staticmethod
    def _cast(sc, o, d, xpos, xmat, sizes):
        """Nearest mover hit over a ray bundle: ``(t, normal, primitive index)``."""
        # One call, primitives batched and rays tiled. The Python loop this replaces cost 79% of a
        # BallInCupCatch render, whose tendon is 28 capsules re-tested by every pass.
        return R3.cast_nearest(sc.prims.kind, sizes, xpos, xmat, o, d)

    @staticmethod
    def _occluded(sc, p, L, sl, xpos, xmat, sizes, skip=None, eps: float = 1e-6):
        """Does any mover block the segment from ``p`` to light ``sl``? ``(B,N)`` bool.

        ``skip`` is the primitive each ray already sits on, excluded so a convex mover does not
        shadow its own front face. That is exact rather than an epsilon: a ray leaving a convex
        surface toward the light re-enters the same primitive only where ``n.l <= 0``, which the
        diffuse term already zeroes.
        """
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

    # ---- render -----------------------------------------------------------

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
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8. Style may differ per environment.

        The batch is grouped by style, because every leaf changes the baked plates. Environments that
        share an appearance share one bake and are rendered in one call.
        """
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

    #: Screen windows tried, as a divisor of the output edge. Smallest first: a window that fits is
    #: always better than a larger one, and one that does not fit costs a projection and nothing else.
    #:
    #: The ladder runs past 2 on purpose. ``ball_in_cup``'s movers span 44x50 of an 84px frame -- 31% of
    #: it, and 1.03% actually covered -- so a half-frame window misses by eight pixels in y and the
    #: environment fell all the way back to full-frame casting. Measured with
    #: the frozen measurement record, which also confirms the per-primitive bound is correct: 0 of 36
    #: primitives draw outside the box projected for them.
    MOVER_TILES = (8, 6, 4, 3, 2, 1.5, 1.25)

    #: Restrict the shadow-from-static pass to the samples where the two plates disagree. A class flag
    #: so ``tests/gpu/test_fast_paths.py`` can turn it off and compare against the pass it replaced;
    #: there is no reason for a caller to.
    SHADOW_COMPACTION = True
    #: Shade only the rays that landed on a mover / in the mirror, as the union over the chunk, when
    #: no screen window fits. Same reason for the flag: with it off the two paths must agree.
    COVER_COMPACTION = True

    def _mover_idx(self, sc: _Scene, xpos: torch.Tensor, size: int, ss: int):
        """Flat ray indices covering every pixel the movers can touch, or None for the full frame.

        The bound is ``screen_span``'s: the world-space AABB of each primitive's bounding sphere, all
        eight corners projected, padded a pixel each way. It is conservative on purpose -- a window that
        clipped a mover would drop pixels silently, and a faster renderer that can disagree with the
        slower one is worth nothing here.

        The span is projected ONCE and the ladder is then walked on the host. Asking for each rung
        separately re-projected the corners and synchronised per rung, which cost more than the ray
        passes the window exists to shrink.
        """
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
        """``(o, d, plate_f, tex, spec, shin, L)``: everything a chunk needs that no state changes.

        Cached per (scene, ss), because the batch is rendered in ``RAY_BUDGET``-sized chunks and every
        one of these was being rebuilt per chunk — at 256 px and ss=2 that is 32 rebuilds of a
        512x512 ray grid per 1024 frames. ``Camera.rays`` builds that grid in NUMPY, in float64, as
        half a dozen (512,512,3) temporaries, and then copies it to the device; it is a pure function
        of (camera, resolution) and it was the single largest term left in a DM Control render once
        the casts were fused. ``mujoco_locomotion`` has cached its rays all along and says so in
        ``_rays``; this family simply never did.
        """
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
        # ``mat_rgb`` is PER ROW, so the gather needs the batch index alongside the primitive index.
        # Only ball_in_cup's tendon actually varies with the state -- it is grey while the string
        # hangs and red while it is taut -- but a colour that is a tensor either way costs one gather
        # and removes the branch.
        bi = torch.arange(B, device=dev).unsqueeze(1)

        # The movers cover 0.2-3.3% of the frame (measured: 1.03% on ball_in_cup, 1.51% on cartpole,
        # 3.29% on reacher), so casting them against every ray spends ~99% of the pass on pixels they
        # cannot reach. Restrict the two mover passes to a conservative screen window; outside it
        # ``t_m`` is INF and ``cover`` is False, so the composite below is unchanged BY CONSTRUCTION
        # rather than by tolerance. ``_mover_idx`` returns None when no window fits and the full-frame
        # path runs, which is what keeps the two paths substitutable.
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

        # A WINDOW IS NOT A COVERAGE MASK, and the difference is most of the cost. The window is a
        # rectangle around the movers' projected bound; the movers cover 1.6-8.4% of the frame but
        # their rectangle is 25-100% of it, because a reacher arm or a finger spinner is long and
        # thin and diagonal. The cast is one fused kernel and does not care, but everything after it
        # -- the shadow ray and ~75 elementwise ops of shading -- is paid per ray. So narrow the
        # window down to the rays that ACTUALLY landed on a mover, as the union over the chunk.
        #
        # Restricting the SET cannot change a VALUE: every ray's colour is a function of that ray
        # alone, and outside the union ``cover`` is False for EVERY row, so the composite below takes
        # the plate there whatever the shader would have said. Note the union is over window
        # POSITIONS, not pixels -- position j is a different pixel in each row — which is exactly as
        # sound, for the same reason.
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

        # The movers go on LAST, over the shadowed and mirrored plate, and they go straight into it:
        # ``where(cover, col, base)`` is the identity wherever ``cover`` is False, and outside the
        # index set — a screen window or the union of covered rays — it provably is. So the composite
        # is a scatter of the set, without a second full-size canvas and a full-frame select.
        if widx is None:  # nothing was restricted: no index to scatter through
            frame = torch.where(cover_w.unsqueeze(-1), col_w, base)
        else:
            frame = base.scatter_(1, w3, torch.where(cover_w.unsqueeze(-1), col_w, base.gather(1, w3)))
        frame = frame.view(B, size, ss, size, ss, 3).sum((2, 4)) / (ss * ss)
        return torch.clamp(torch.round(frame), 0, 255).to(torch.uint8)

    def _floor_tensors(self, sc: _Scene):
        """``(size, pos, mat)`` of the reflective plane, as device tensors, built once per scene.

        Three ``torch.tensor(..., device=cuda)`` calls per chunk otherwise, and a pageable
        host-to-device copy blocks until the stream drains — see the note in ``raycast3d``.
        """
        hit = sc.const.get("floor")
        if hit is None:
            fsize, fpos, fmat = sc.floor
            t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)  # noqa: E731
            hit = sc.const["floor"] = (t(fsize), t(fpos), t(fmat))
        return hit

    def _reflect(self, sc, base, xpos, xmat, sizes, mat_rgb, tex_rgb, spec, shin, L, ss, bi):
        """Put the movers into the reflective floor, using MuJoCo's own compositing rule.

        ``frame = reflectance * mirror + (1 - reflectance) * surface``. The un-reflected plate IS
        ``surface``, so a mover that appears in the mirror replaces the mirror term outright instead
        of being blended into a number that would have to be guessed. The alternative — scaling the
        mover's colour into the existing plate — cannot be right, because the plate already holds
        whatever the mirror showed at that pixel (skybox, a rail, the far floor) and there is no way
        to subtract it back out.

        Everything geometric is precomputed by ``_floor_rays``; what is left per frame is one fused
        cast down the reflected rays and a shade of the samples that came back with a hit.
        """
        idx, ro, rd, surf = self._floor_rays(sc, ss)
        if idx.numel() == 0:
            return base
        B = base.shape[0]
        t_r, n_r, i_r = self._cast(sc, ro.expand(B, -1, 3), rd.expand(B, -1, 3), xpos, xmat, sizes)
        seen = t_r < R3.INF
        widx = idx.view(1, -1).expand(B, -1)
        # Narrow to the samples where a mover is ACTUALLY in the mirror. The floor is a third to a
        # half of the frame and the reflection reaches a few percent of it; ``seen`` is False for
        # every row outside the union, so the select below is the identity there and the shade —
        # ~75 elementwise ops — never runs on it.
        if self.COVER_COMPACTION:
            js = torch.nonzero(seen.any(0), as_tuple=False).reshape(-1)
            widx, seen, ro, rd = widx[:, js], seen[:, js], ro[:, js], rd[:, js]
            surf, t_r, n_r, i_r = surf[:, js], t_r[:, js], n_r[:, js], i_r[:, js]
        pr = ro + t_r.clamp(max=1e6).unsqueeze(-1) * rd
        colr = _shade(pr, n_r, -rd, L, mat_rgb[bi, i_r], spec[i_r], shin[i_r], None) * tex_rgb[i_r] * 255.0
        r = float(sc.reflectance)
        val = r * colr + (1.0 - r) * surf
        w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
        # in place: ``base`` is this render's own float canvas, and the out-of-place ``scatter``
        # would copy the whole frame to change a few percent of it
        return base.scatter_(1, w3, torch.where(seen.unsqueeze(-1), val, base.gather(1, w3)))
