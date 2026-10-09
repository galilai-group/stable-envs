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
    """Share compiled models, appearance defaults and source rendering for DM Control.

    Compact render rows contain qpos and episode-specific model entries. They are
    not restorable simulator State; registered recovered environments compose a
    separate simulation adapter that defines the full State schema."""

    family = "dmcontrol"

    DOMAIN: str = ""
    TASK: str = ""
    CAMERA: int = 0
    default_size = 84
    default_supersample = 1
    # Renderer capability: MJWarp does not batch cam_fovy per world.
    CAMERA_KNOB = True
    # Renderer capability: cached MJWarp contexts fix visibility at construction.
    VISIBILITY_KNOB = True
    # Per-geom alternatives: ((mjtGeom, (sx, sy, sz)), ...). Exactly one must
    # match the compiled model; shape changes are shared within an appearance group.
    SHAPE_KNOBS: dict[str, tuple[tuple[int, tuple[float, float, float]], ...]] = {}

    # Empty when CPU rendering is supported; otherwise names the certification gap.
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
        """Derive compact-row names and widths from the compiled model."""
        m = self._model()
        self.nq = int(m.nq)
        self.state_dim = self.nq + len(self._slots())
        self.state_names = self._qpos_names(m) + self._slot_names(m)
        # The compiled model determines both widths; omit an empty episode field.
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


    def _close(self) -> None:
        """Drop instance CPU scenes; shared GL and MJWarp contexts remain cached."""
        self.__dict__.pop("_cpu_scenes", None)


    @classmethod
    def _load_env(cls, seed: int):
        """Load the common source used for models, sampling and reference frames.

        Override here when a registration requires a model other than suite.load’s."""
        return _suite().load(cls.DOMAIN, cls.TASK, task_kwargs={"random": seed})

    @classmethod
    def _bundle(cls, size: int):
        """Cache ``(env, model, data, pristine, renderer, slots)`` by domain, task and size.

        Episode probing applies task model edits and separates constant appearance
        from sampled fields. Its fixed seed makes the snapshot reproducible."""
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
            # Restore shape arrays for every appearance, including domains without shape knobs.
            geom_type=np.array(m.geom_type, copy=True),
            geom_size=np.array(m.geom_size, copy=True),
        )
        for arr in {a for a, _ in slots}:
            pristine[arr] = np.array(getattr(m, arr), copy=True)
        # Reuse this GL context for the process lifetime; teardown can corrupt other contexts.
        r = mujoco.Renderer(m, height=int(size), width=int(size))
        _ENV_CACHE[key] = (env, m, d, pristine, r, slots)
        return _ENV_CACHE[key]

    @classmethod
    def _slots(cls) -> tuple[tuple[str, int], ...]:
        """Return ``(model-array name, flat index)`` pairs carried in compact render rows."""
        return cls._bundle(cls.default_size)[5]

    @classmethod
    def _slot_names(cls, m) -> tuple[str, ...]:
        return tuple(_slot_name(m, arr, i) for arr, i in cls._slots())

    @classmethod
    def _hideable(cls) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
        """Resolve episode landmarks to ``(kind, name, element IDs)`` visibility leaves.

        A body slot covers its visible geoms; mixed geom/site bodies are rejected.
        Use pristine alpha and groups so previous renders cannot change declarations."""
        import mujoco

        m, pr = cls._model(), cls._pristine()
        galpha = _effective_alpha(m, "geom", rgba=pr["geom_rgba"], mat_rgba=pr["mat_rgba"])
        salpha = _effective_alpha(m, "site", rgba=pr["site_rgba"], mat_rgba=pr["mat_rgba"])
        out: list[tuple[str, str, tuple[int, ...]]] = []
        seen: dict[str, tuple[str, tuple[int, ...]]] = {}
        for arr, k in cls._slots():
            kind, eid = _slot_element(m, arr, k)
            if kind == "body":
                # A single leaf cannot represent both geom and site IDs; refuse ambiguous bodies.
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

                    continue
                kind, name = "geom", (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, eid) or f"body{eid}")
            elif kind in ("geom", "site"):
                obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE}[kind]
                name = mujoco.mj_id2name(m, obj, eid) or f"{kind}{eid}"
                ids = (eid,)
            else:
                continue
            if name in seen and seen[name] != (kind, ids):
                # Duplicate names would otherwise silently overwrite a visibility leaf.
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
        """Return the canonical snapshot from before appearance changes.

        The shared model retains the last rendered style. Read defaults from this
        snapshot to keep new instances independent of previous rendering."""
        return cls._bundle(cls.default_size)[3]

    @staticmethod
    def _qpos_names(m) -> tuple[str, ...]:
        """Name scalar positions and quaternion components from the model’s joint layout."""
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


    @classmethod
    def _used_materials(cls) -> list[str]:
        """List materials on visible geoms/sites using pristine groups and resolved alpha."""
        import mujoco

        m, pr = cls._model(), cls._pristine()
        used = []
        for kind, arr, grp, n in (
            ("geom", m.geom_matid, pr["geom_group"], m.ngeom),
            ("site", m.site_matid, pr["site_group"], m.nsite),
        ):
            alpha = _effective_alpha(m, kind, rgba=pr[f"{kind}_rgba"], mat_rgba=pr["mat_rgba"])
            for i in range(n):
                # Group membership alone includes transparent markers with no visible pixels.
                if int(grp[i]) not in VISIBLE_GROUPS or alpha[i] <= 0:
                    continue
                mid = int(arr[i])
                if mid >= 0 and mid not in used:
                    used.append(mid)
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}" for i in used]

    @property
    def dynamics_variation_leaves(self):
        """Gravity and body densities change the physics, never a single rendered frame."""
        return frozenset(
            n for n in self.variation_names if n.startswith(("gravity.", "agent_physics.")) or n.endswith("_density")
        )

    def make_variation_space(self) -> Dict:
        """Derive supported appearance leaves and defaults from the pristine model.

        Materials use float colours in [0, 1]. Visibility is binary because the
        analytic renderer does not blend partial alpha. Shape choices must contain
        exactly one match for the compiled geometry, which becomes the default."""
        import mujoco

        m, pr = self._model(), self._pristine()
        mats = {}
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)

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
                # A body landmark is visible if any of its pristine geoms are visible.
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
                # A geom name must not overwrite an existing variation group.
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


    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample compact render rows from source resets followed by random actions.

        Read qpos and episode-specific model entries from the same trajectory."""
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


    @classmethod
    def reference_env(cls, size: int | None = None):
        """Construct a fresh source with its render resolution attached.

        Keep its GL context alive for the process lifetime to avoid cross-context
        renderer corruption on teardown."""
        env = cls._load_env(0)
        env.reset()
        env._se_size = int(size or cls.default_size)
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore compact rows in the source, verify them, and render the requested camera.

        Velocity is set to zero because these compact rows describe geometry only."""
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        size = int(getattr(env, "_se_size", 0) or cls.default_size)
        slots = cls._slots()
        nq = int(m.nq)
        out = []
        for st in np.asarray(states, np.float64):
            # Write model entries before reset_context runs forward kinematics, so derived
            # world poses use this row rather than the previous render’s episode fields.
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
