"""``_DMCModel`` — everything a DM Control cell is before a renderer touches it, and how a cell is declared.

ONE SMALL BASE. The model plumbing, the widened state, the appearance knobs, the visited-state
distribution and the bridge to ``dm_control``'s own renderer are properties of the DOMAIN and are
written once here. Which rasterizer draws the frame is NOT a property of the domain, and the two
tiers disagree about it completely, so they are two modules beside this one (``suite``, ``trackcom``)
rather than two more rungs of a hierarchy. Below those there is exactly one class per registered
cell, and those classes are DECLARATIONS — ``DOMAIN``/``TASK``/``CAMERA`` and the measured numbers —
produced by ``parity_cell``/``trackcom_cell`` from the data in the eighteen domain modules.

The Public State additions live here too, and there are only three:

  * ``num_envs=`` (contract §2);
  * a two-key public ``State`` (contract §3) — ``{"qpos": (nq,), "episode": (k,)}`` — whose
    ``state_layout`` is assigned PER INSTANCE, because a DM Control cell's ``state_dim`` is a
    property of the compiled model and is not known until one is constructed;
  * ``close()`` refusing to render afterwards, without releasing the shared EGL context, which this
    family must never release.

NO DYNAMICS, and that is a declaration rather than an omission: these cells ship a renderer over
``dm_control``'s own physics and there is no batched simulator in this package for any of them, so
``step``/``transition`` raise ``Env``'s precise refusal and ``has_dynamics`` is False. ``reset`` and
``render`` work as usual; to produce trajectories, roll out the official CPU env and render the
states it gives you.
"""

from __future__ import annotations

import numpy as np

from ...env import Env
from ...spaces import Box, Dict, Discrete
from ._model import (  # noqa: F401
    _ENV_CACHE,
    N_DRAWS,
    VISIBLE_GROUPS,
    _effective_alpha,
    _episode_slots,
    _slot_element,
    _slot_name,
    _suite,
)


class _DMCModel(Env):
    """What every DM Control cell shares, whichever renderer draws it.

    The model plumbing, the widened state, the appearance knobs, the visited-state distribution and
    the bridge to ``dm_control``'s own renderer are properties of the DOMAIN. Which rasterizer draws
    the frame is not, and the two tiers below disagree about it entirely: ``DMControlSuite`` bakes a
    plate and ray-casts the movers (parity), ``DMControlTrackCom`` hands the whole scene to MJWarp
    every frame (visual). Everything that would otherwise be written twice, and drift, lives here.
    """

    family = "dmcontrol"

    DOMAIN: str = ""
    TASK: str = ""
    CAMERA: int = 0
    default_size = 84
    default_supersample = 1
    #: Whether the appearance space offers a field-of-view knob. It is a class flag rather than an
    #: always-on knob because whether it can be honoured is a property of the RENDERER: the ray-caster
    #: re-bakes per appearance and gets it for free, while MJWarp would need ``cam_fovy`` per world and
    #: ``put_model`` does not carry that array per world.
    CAMERA_KNOB = True
    #: Whether the appearance space offers a per-element VISIBILITY knob. A class flag for exactly the
    #: reason ``CAMERA_KNOB`` is one — whether it can be honoured is a property of the RENDERER, not of
    #: the domain. The ray-caster re-bakes per appearance and gets it for free because MuJoCo's own
    #: alpha filter is already what ``_visible`` reads; MJWarp's shader never reads alpha and hides
    #: zero-alpha elements by moving them out of the enabled geom groups at CONTEXT-CREATION time
    #: (``MW.prepare_model`` in ``_tc_context``), which is once per (domain, res, worlds) and not per
    #: world. See ``DMControlTrackCom``.
    VISIBILITY_KNOB = True
    #: Discrete SHAPE alternatives for a drawn geom: ``{geom name: ((mjtGeom, (sx, sy, sz)), ...)}``,
    #: declared per cell and empty on all but ``DMCManipulatorBringBall``. The leaf is
    #: ``<geom name>.shape`` and it is a PLATE path like every other leaf, so a group of environments
    #: that share it share one bake; ``_prims`` reads ``geom_type`` and ``geom_size`` off the model AT
    #: BAKE TIME, so the alternative reaches the mover set with no per-row machinery. That is the whole
    #: reason it can exist while a per-EPISODE size still cannot (``_episode_slots``): a style is
    #: constant over a bake group and an episode draw is not.
    #:
    #: EXACTLY ONE alternative must match the compiled model, and that index becomes the leaf's
    #: ``init_value``. What that buys, stated precisely because it is easy to overclaim: the DEFAULT
    #: STYLE renders the compiled model rather than a transcribed constant, so the appearance the parity
    #: numbers are measured under is the one ``suite.load`` compiles, and a model whose target changes
    #: shape upstream fails LOUDLY here instead of silently redefining what "default" means.
    #: ``check_default_style_is_official`` does NOT establish that: it compares ``render(states)``
    #: against ``render(states, style=default_style())`` and never touches the official renderer, so a
    #: default that had drifted off the compiled model would pass it. This assertion is the cover.
    SHAPE_KNOBS: dict[str, tuple[tuple[int, tuple[float, float, float]], ...]] = {}

    #: The reason ``device="cpu"`` is refused for this cell, or ``""`` when it is not. A DECLARATION
    #: on the cell rather than a name in a table beside it: the fact is "this cell has no CPU
    #: certification", and the cell is where a reader looks for what a cell claims. The package
    #: re-exports the collected mapping as ``CPU_UNCERTIFIED`` for the scripts that read it.
    CPU_UNCERTIFIED_REASON: str = ""

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int | None = None,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        """``num_envs`` environments on ``device`` (contract §2).

        ``state_dim`` and ``state_names`` are read off the COMPILED MODEL here and not declared on the
        class, because they are: ``nq`` plus however many model entries this task draws per episode,
        and both are discovered (``_episode_slots``) rather than transcribed. ``state_layout`` is
        likewise built here rather than declared on the class, for the same reason: the public
        State's shape is not known until a model has been compiled.
        """
        m = self._model()
        self.nq = int(m.nq)
        self.state_dim = self.nq + len(self._slots())
        self.state_names = self._qpos_names(m) + self._slot_names(m)
        # ``state_layout`` is set PER INSTANCE, which is the whole of this family's public State
        # machinery since the foundation made ``Env._layout`` an instance method (integration
        # f24d94c, this workstream's request R1). The two keys are what the halves ARE, and the
        # ``episode`` key is absent on the eleven cells that draw nothing per episode.
        k = self.state_dim - self.nq
        self.state_layout = ((("qpos",), self.nq, 0),) + (((("episode",), k, self.nq),) if k else ())
        super().__init__(
            num_envs=num_envs,
            size=size,
            supersample=self.default_supersample if supersample is None else supersample,
            device=device,
            backend=backend,
            seed=seed,
        )
        assert len(self.state_names) == self.state_dim

    # ---- state -----------------------------------------------------
    #
    # The packed float32 row is unchanged: ``qpos`` then the episode draw, exactly the packed float32 layout
    # every frozen corpus, digest and certification check speaks. The public State adds the entity
    # grouping over it, and the grouping is the state's own structure rather than an invention --
    # ``qpos`` is the simulator's configuration and ``episode`` is what ``initialize_episode`` wrote
    # into the MODEL, which is a different KIND of quantity and is the whole subject of the widening
    # ``_episode_slots`` performs. Eleven of the eighteen cells draw nothing per episode and get the
    # single key alone, which is what contract §3 means by no ceremony for a state that is one vector.
    #
    # NOTHING IS HIDDEN. Every column of the packed row is carried by one of the two keys, so no
    # field has offset ``None``, ``_hidden_state`` is never consulted, and ``pack``/``unpack`` round
    # trip the frozen corpus bit for bit. There is no ``_live_state`` either: the packed row IS this
    # family's live state -- ``set_state`` writes it and ``render`` reads it -- so there is no more
    # precise buffer to answer from, unlike classic control's float64 simulator.
    #
    # ``state_layout`` is an INSTANCE attribute, assigned in ``__init__`` above, and there is no
    # ``_layout`` override any more: ``Env._layout`` reads ``self.state_layout``, so a family whose
    # width comes from a compiled model declares it where the model is compiled. That is request R1,
    # done by the foundation at f24d94c.

    def _close(self) -> None:
        """Release what this INSTANCE owns, which is the per-instance CPU scene cache and nothing else.

        The ``mujoco.Renderer`` in ``_ENV_CACHE`` and the captured MJWarp context in ``_TC_CACHE`` are
        NOT released, and that is the family's policy rather than an oversight: they are shared by
        every instance of the class and by every class at the same (domain, size), and releasing an
        EGL context corrupts every other live one in the process -- the rule
        ``stable_envs/testing/conformance.py`` carries and the reason ``reference_env`` says "never
        closed". What ``close()`` buys here is the contract's other half: a closed env REFUSES to
        render rather than handing back frames drawn through something a caller believes is gone.
        """
        self.__dict__.pop("_cpu_scenes", None)

    # ---- model / reference plumbing --------------------------------------

    @classmethod
    def _load_env(cls, seed: int):
        """The dm_control env this cell is built from: ``suite.load`` by default.

        THE ONE PLACE A CELL'S MODEL COMES FROM. ``_bundle`` (the canonical model), ``sample_states``
        (the visited distribution) and ``reference_env`` (the official frames) all load through
        it, so a cell whose model is not ``suite.load``'s -- ``DMCQuadrupedWalk`` draws the env
        ``stable_worldmodel`` registers, built through ``dm_control.mjcf`` with SWM's default
        appearance -- overrides this and nothing else (integration finding INTEG-1: a copied
        ``_bundle`` drifted the moment the base one changed).
        """
        return _suite().load(cls.DOMAIN, cls.TASK, task_kwargs={"random": seed})

    @classmethod
    def _bundle(cls, size: int):
        """``(dm_control env, MjModel, MjData, pristine, Renderer, slots)``, one per (domain, size).

        The model is put through ``N_DRAWS`` real episodes before anything is read off it, and that is
        load-bearing twice over. ``initialize_episode`` is where a DM Control task edits its model, and
        a model straight out of ``suite.load`` has had none of those edits applied — ``finger-spin``
        hides its ``target`` and ``tip`` sites by setting their alpha to zero there, so a renderer
        reading the freshly loaded model paints two markers the reference does not draw. Running the
        episodes also SEPARATES the constant edits from the per-episode draw, which is the distinction
        ``_episode_slots`` needs and which one reset cannot make.

        The draws are deterministic: ``task_kwargs={"random": 0}`` seeds the task's own
        ``RandomState`` and the count is fixed, so a fresh interpreter reaches the same canonical
        model. That matters for the same reason ``_pristine`` does — the cross-process bit-identity
        test is the only thing that can see a model that differs between two processes.
        """
        key = (cls.DOMAIN, cls.TASK, int(size))
        hit = _ENV_CACHE.get(key)
        if hit is not None:
            return hit
        import mujoco

        env = cls._load_env(0)
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(size))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(size))
        slots = _episode_slots(env, m)
        d = mujoco.MjData(m)
        pristine = dict(
            mat_rgba=np.array(m.mat_rgba, copy=True),
            geom_rgba=np.array(m.geom_rgba, copy=True),
            site_rgba=np.array(m.site_rgba, copy=True),
            geom_group=np.array(m.geom_group, copy=True),
            site_group=np.array(m.site_group, copy=True),
            light_diffuse=np.array(m.light_diffuse, copy=True),
            light_specular=np.array(m.light_specular, copy=True),
            mat_reflectance=np.array(m.mat_reflectance, copy=True),
            cam_fovy=np.array(m.cam_fovy, copy=True),
            # geom_type/geom_size are here for ``SHAPE_KNOBS``, and unconditionally, so
            # that ``_apply_variation``'s restore is one shape for every cell. They are two
            # small arrays and the alternative is a branch in the hottest correctness path
            # in this file.
            geom_type=np.array(m.geom_type, copy=True),
            geom_size=np.array(m.geom_size, copy=True),
        )
        for arr in {a for a, _ in slots}:
            pristine[arr] = np.array(getattr(m, arr), copy=True)
        # ONE Renderer, kept for the life of the process. Material colours, light colours and the
        # camera are read per frame, so no variation here needs a second GL context — only a TEXTURE
        # edit would, and none is offered. Closing it would corrupt every other live context.
        r = mujoco.Renderer(m, height=int(size), width=int(size))
        _ENV_CACHE[key] = (env, m, d, pristine, r, slots)
        return _ENV_CACHE[key]

    @classmethod
    def _slots(cls) -> tuple[tuple[str, int], ...]:
        """``((model array, flat index), ...)`` the episode draws — the widening of the state."""
        return cls._bundle(cls.default_size)[5]

    @classmethod
    def _slot_names(cls, m) -> tuple[str, ...]:
        return tuple(_slot_name(m, arr, i) for arr, i in cls._slots())

    @classmethod
    def _hideable(cls) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
        """``((kind, name, (id, ...)), ...)`` the visibility knob covers: what the EPISODE DRAWS.

        DERIVED, never a name list. ``_episode_slots`` already measured which model entries the task
        writes per episode, by diffing real episodes rather than transcribing a task's source, and
        those entries ARE the scene's task-relevant landmarks: they are the reason the state is widened
        (``_DMCModel.__init__``) and the reason ``_movers`` has to treat a world-body geom as a mover
        at all. Measured over all seventeen shipped cells: the two reachers draw a ``target`` GEOM, the
        two finger-turn cells a ``target`` SITE, swimmer a ``target`` geom and a ``target_light``,
        ``manipulator`` a ``target_ball`` BODY, and the other eleven draw nothing — so eleven cells gain
        no leaf and their ``describe()`` does not move.

        A BODY SLOT IS AN INDIRECTION, and it is the reason the third field is a tuple rather than an
        id. ``reacher`` and ``finger-turn`` draw the element that is drawn; ``manipulator``'s
        ``initialize_episode`` writes ``body_pos``/``body_quat`` of ``target_ball``, which is a FRAME
        and has no rgba of its own — what MuJoCo draws is the geoms hanging off it. So a body slot
        resolves to every DRAWN geom on that body, named by the BODY, and hiding the leaf zeroes each
        of their alphas. That is the faithful reading of ``stable_worldmodel``'s own
        ``render_target = 0``, which sets the ``target`` MATERIAL's alpha to zero and hides exactly the
        ``target_ball`` geom.

        The census is taken against the PRISTINE snapshot, for the reason ``_used_materials`` records:
        the shared ``MjModel`` carries the last bake's appearance, so reading the live one would make
        the DECLARED space depend on what the previous env most recently drew.

        Lights are excluded because visibility is not what an rgba alpha means for one: MuJoCo draws no
        geometry for a light, it only lights the scene, and "hide the light" is ``light_diffuse``,
        which the appearance space already carries. Order follows ``_slots()`` so the declared space is
        a function of the model and not of a set's iteration order.
        """
        import mujoco

        m, pr = cls._model(), cls._pristine()
        galpha = _effective_alpha(m, "geom", rgba=pr["geom_rgba"], mat_rgba=pr["mat_rgba"])
        salpha = _effective_alpha(m, "site", rgba=pr["site_rgba"], mat_rgba=pr["mat_rgba"])
        out: list[tuple[str, str, tuple[int, ...]]] = []
        seen: dict[str, tuple[str, tuple[int, ...]]] = {}
        for arr, k in cls._slots():
            kind, eid = _slot_element(m, arr, k)
            if kind == "body":
                # The drawn geoms on the body, from the pristine groups and alphas. A body that also
                # carried a drawn SITE would need two leaves or one ambiguous ``kind``; nothing in the
                # shipped set does (manipulator's ``target_ball`` site is in group 3), so it raises
                # rather than silently hiding half the marker.
                ids = tuple(
                    i
                    for i in range(m.ngeom)
                    if int(m.geom_bodyid[i]) == eid and int(pr["geom_group"][i]) in VISIBLE_GROUPS and galpha[i] > 0
                )
                bad = [
                    i
                    for i in range(m.nsite)
                    if int(m.site_bodyid[i]) == eid and int(pr["site_group"][i]) in VISIBLE_GROUPS and salpha[i] > 0
                ]
                if bad:
                    raise RuntimeError(
                        f"{cls.__name__}: the episode draws body "
                        f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, eid)!r}, which carries both "
                        f"drawn geoms {list(ids)} and drawn sites {bad}; one visibility leaf cannot "
                        f"name both kinds"
                    )
                if not ids:
                    # A drawn body with nothing drawn on it is not hideable, and advertising a leaf for
                    # it would fail check_variations_do_something on a knob that was never real.
                    continue
                kind, name = "geom", (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, eid) or f"body{eid}")
            elif kind in ("geom", "site"):
                obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE}[kind]
                name = mujoco.mj_id2name(m, obj, eid) or f"{kind}{eid}"
                ids = (eid,)
            else:
                continue
            if name in seen and seen[name] != (kind, ids):
                # Two different elements would claim one variation path, and the second would silently
                # win. A domain that does this needs a naming rule chosen deliberately, not here.
                raise RuntimeError(
                    f"{cls.__name__}: the episode draws two different elements both named {name!r} "
                    f"({seen[name]} and {(kind, ids)}); visibility.{name} would be ambiguous"
                )
            if name in seen:
                continue
            seen[name] = (kind, ids)
            out.append((kind, name, ids))
        return tuple(out)

    @classmethod
    def _model(cls):
        return cls._bundle(cls.default_size)[1]

    @classmethod
    def _pristine(cls) -> dict:
        """The model AS COMPILED, which is the only thing an appearance default may be read from.

        The ``MjModel`` in ``_bundle`` is shared by every instance of this class and is mutated by every
        bake, and ``_apply_variation`` restores it from this snapshot at the START of a bake rather than
        at the end — so between bakes it carries the LAST appearance rendered, not the official one.
        Reading ``make_variation_space``'s ``init_value`` off the live model therefore made a new env's
        default appearance depend on what the previous env had most recently drawn.

        Measured, because it is invisible from inside one env: the default plate's sha256 changed after
        ``check_variations_do_something`` ran, and the only test that could see it was the cross-process
        one — a fresh interpreter has no previous bake, so the subprocess and the parent disagreed on
        what "default" meant while every same-process check passed.
        """
        return cls._bundle(cls.default_size)[3]

    @staticmethod
    def _qpos_names(m) -> tuple[str, ...]:
        """One label per ``qpos`` entry, read off the model's joint layout rather than transcribed —
        a state vector whose fields are mislabelled is worse than one with no labels."""
        import mujoco

        free = ("_x", "_y", "_z", "_qw", "_qx", "_qy", "_qz")
        ball = ("_qw", "_qx", "_qy", "_qz")
        out: list[str] = []
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            t = int(m.jnt_type[j])
            if t == int(mujoco.mjtJoint.mjJNT_FREE):
                out += [nm + s for s in free]
            elif t == int(mujoco.mjtJoint.mjJNT_BALL):
                out += [nm + s for s in ball]
            else:
                out.append(nm)
        return tuple(out)

    # ---- variation --------------------------------------------------------

    @classmethod
    def _used_materials(cls) -> list[str]:
        """Materials carried by a VISIBLE geom or site, in model order.

        Declared from the model rather than from a fixed list because
        ``check_variations_do_something`` moves every declared leaf and fails if the frame does not
        change: cartpole has no ``effector`` and acrobot no ``target``, so a shared knob set would
        advertise support that does not exist in three of the five domains.

        Every array is read from the PRISTINE snapshot, alpha included. The groups always were; alpha
        was not, and it had to become so the moment a bake could legitimately leave an element hidden
        (``visibility.*``) — otherwise this census, and therefore the DECLARED variation space of the
        next env built from this class, would depend on what the previous env last drew. The same
        failure ``_pristine`` documents, one array further along.
        """
        import mujoco

        m, pr = cls._model(), cls._pristine()
        used = []
        for kind, arr, grp, n in (
            ("geom", m.geom_matid, pr["geom_group"], m.ngeom),
            ("site", m.site_matid, pr["site_group"], m.nsite),
        ):
            alpha = _effective_alpha(m, kind, rgba=pr[f"{kind}_rgba"], mat_rgba=pr["mat_rgba"])
            for i in range(n):
                # alpha as well as group: ``finger-spin`` hides its ``target`` and ``tip`` sites by
                # setting their alpha to zero, and they are the ONLY wearers of the ``target``
                # material — so a group-only census declares a knob that cannot change any pixel and
                # ``check_variations_do_something`` fails on a knob that was never real.
                if int(grp[i]) not in VISIBLE_GROUPS or alpha[i] <= 0:
                    continue
                mid = int(arr[i])
                if mid >= 0 and mid not in used:
                    used.append(mid)
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}" for i in used]

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are read off the compiled model, so "the default style is
        the official appearance" is true by construction rather than by a transcribed constant.

        The colours are float ``Box`` in [0,1] rather than ``RGBBox`` for the reason the OGBench
        mazes are: MuJoCo's material colours are floats (0.7, 0.5, 0.3) that a uint8 channel cannot
        hold, and in a lit 3-D scene no material colour appears verbatim in any pixel anyway, because
        a pixel is the material colour times the lighting.

        VISIBILITY is a ``Discrete(2)`` and deliberately not an alpha ``Box``. ``stable_worldmodel``'s
        reacher hides its target with a material alpha of zero (``render_target = 0``) and MuJoCo then
        declines to draw the geom; the thing that has to be representable is therefore exactly "drawn
        or not". MuJoCo BLENDS an alpha strictly between 0 and 1 and the analytic ray-caster is
        opaque-only, so a continuous knob would advertise a range two thirds of which this renderer
        cannot honour, and no conformance check in the suite could catch it. A binary knob says what
        is true. Its default is read off the pristine model's own resolved alpha, so the default
        appearance is still the compiled model's by construction.
        """
        import mujoco

        m, pr = self._model(), self._pristine()
        mats = {}
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            # from the PRISTINE snapshot, never the live model — see _pristine for what that cost
            mats[name] = Box(low=0.0, high=1.0, shape=(3,), init_value=[float(x) for x in pr["mat_rgba"][i][:3]])
        space = {
            "material": Dict(mats),
            "light": Dict({"diffuse_scale": Box(low=0.2, high=1.8, shape=(1,), init_value=[1.0])}),
        }
        if self.CAMERA_KNOB:
            space["camera"] = Dict({"fovy_scale": Box(low=0.55, high=1.8, shape=(1,), init_value=[1.0])})
        vis = {}
        if self.VISIBILITY_KNOB:
            for kind, name, eids in self._hideable():
                a = _effective_alpha(m, kind, rgba=pr[f"{kind}_rgba"], mat_rgba=pr["mat_rgba"])
                # ANY of the leaf's elements drawn means the leaf's default is "drawn". A body leaf
                # covers every drawn geom on the body, and ``_hideable`` has already dropped the ones
                # the pristine model does not draw, so this is 1 for a body and is still read from the
                # PRISTINE alphas rather than from whatever the last bake left on the model.
                vis[name] = Discrete(2, init_value=int(any(float(a[i]) > 0 for i in eids)))
        if vis:
            space["visibility"] = Dict(vis)
        for gname, alts in self.SHAPE_KNOBS.items():
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid < 0:
                raise RuntimeError(
                    f"{type(self).__name__}: SHAPE_KNOBS names geom {gname!r}, which this model does not have"
                )
            cur = [
                k
                for k, (t, s) in enumerate(alts)
                if int(t) == int(pr["geom_type"][gid]) and np.allclose(np.asarray(s, np.float64), pr["geom_size"][gid])
            ]
            if len(cur) != 1:
                raise RuntimeError(
                    f"{type(self).__name__}: {len(cur)} of the {len(alts)} declared shapes for "
                    f"{gname!r} match the compiled model (type {int(pr['geom_type'][gid])}, size "
                    f"{list(pr['geom_size'][gid])}); exactly one must, or this env's DEFAULT STYLE "
                    f"stops being a render of the compiled model — which is the appearance every "
                    f"parity number here is measured under, and which "
                    f"check_default_style_is_official cannot see (it compares the default style "
                    f"against itself, never against the official renderer)"
                )
            if gname in space:
                # The leaf lives at ``<geom name>.shape``, so a geom named after one of the groups
                # above would REPLACE that group and silently delete every leaf in it. Nothing in the
                # suite is named ``material``/``light``/``camera``/``visibility``, and a domain that
                # is must choose a naming rule deliberately rather than lose its colour knobs here.
                raise RuntimeError(
                    f"{type(self).__name__}: geom {gname!r} would put its shape leaf at "
                    f"{gname}.shape and overwrite the existing {gname!r} variation group "
                    f"({sorted(space[gname])}); rename the group or the leaf"
                )
            space[gname] = Dict({"shape": Discrete(len(alts), init_value=cur[0])})
        return Dict(space)

    def _var(self, variation, path):
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)

    # ---- states -----------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """The domain's OWN visited distribution: its ``initialize_episode`` plus random actions.

        Not a convenient box, and not the reset distribution alone: cartpole-balance starts near
        upright and would never show the renderer a hanging pole, so a parity number measured there
        would be a statement about one pose. Every conformance number in the suite is measured on
        these states, and a parity test on states the env never reaches proves very little.

        The episode half of the state is read out of the model the episode wrote it into, so a
        sampled state is a pairing the domain genuinely produced rather than a ``qpos`` glued to a
        target drawn somewhere else.
        """
        rng = rng or self.rng
        seed = int(rng.integers(0, 2**31 - 1))
        env = self._load_env(seed)
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        spec = env.action_spec()
        r2 = np.random.default_rng(seed)
        slots = self._slots()
        out = np.empty((int(batch), self.state_dim), np.float64)
        for i in range(int(batch)):
            env.reset()
            for _ in range(int(r2.integers(0, 40))):
                env.step(r2.uniform(spec.minimum, spec.maximum))
            out[i, : self.nq] = np.asarray(p.data.qpos, np.float64)
            for j, (arr, k) in enumerate(slots):
                out[i, self.nq + j] = np.asarray(getattr(m, arr)).reshape(-1)[k]
        return out.astype(np.float32)

    # ---- reference bridge -------------------------------------------------

    @classmethod
    def reference_env(cls, size: int | None = None):
        """A FRESH official env, carrying the size it must render at (``default_size`` if none).

        Never closed. ``test_state_is_sufficient`` drives one through two histories, and a
        dm_control ``Physics`` holds the EGL context its renderer created — releasing it corrupts
        rendering in every other live context in the process.
        """
        env = cls._load_env(0)
        env.reset()
        env._se_size = int(size or cls.default_size)
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for the given state, with the round-trip ASSERTED.

        ``qvel`` is set to zero and is provably irrelevant: scaling it by 100 changes zero pixels on
        every domain measured. The assertion is not decoration — without it, a state the reference did
        not actually reach produces a plausible parity number instead of a failure, and the mistake
        reads as a rendering discrepancy.
        """
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        size = int(getattr(env, "_se_size", 0) or cls.default_size)
        slots = cls._slots()
        nq = int(m.nq)
        out = []
        for st in np.asarray(states, np.float64):
            # The episode half of the state goes into the reference's MODEL, which is where
            # ``initialize_episode`` put it. Written BEFORE the reset context so the ``mj_forward``
            # that closes the context sees it -- ``docs/research/metaworld.md`` S1.1 is the recorded
            # form of the opposite mistake, where a model edit after the last forward leaves
            # ``site_xpos`` holding the previous draw and no stored state can reproduce the frame.
            for (arr, k), v in zip(slots, st[nq:]):
                getattr(m, arr).reshape(-1)[k] = v
            with p.reset_context():
                p.data.qpos[:] = st[:nq]
                p.data.qvel[:] = 0
            got = np.concatenate(
                [np.asarray(p.data.qpos, np.float64), [np.asarray(getattr(m, arr)).reshape(-1)[k] for arr, k in slots]]
            )
            if np.abs(got - st).max() > 1e-9:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state "
                    f"(max {np.abs(got - st).max():.3e}); every parity number computed from it would "
                    f"be comparing two different scenes"
                )
            out.append(p.render(height=size, width=size, camera_id=cls.CAMERA).copy())
        return np.stack(out)
