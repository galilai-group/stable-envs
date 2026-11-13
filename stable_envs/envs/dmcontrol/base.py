from __future__ import annotations

import numpy as np

from ..._core.mujoco_gl import mujoco_renderer
from ...env import Env
from ...render.mjwarp import effective_alpha
from ...spaces import Box, Dict, Discrete
from . import warp_render
from ._model import (
    _ENV_CACHE,
    VISIBLE_GROUPS,
    _episode_slots,
    _row_width,
    _slot_element,
    _slot_name,
    _suite,
)


class _DMCModel(Env):
    """Share compiled models, appearance defaults and source rendering for DM Control.

    Compact render rows contain qpos and episode-specific model entries. They are
    not restorable simulator state; registered environments compose a separate
    simulation adapter that defines the full state schema.
    """

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
    # Per-geom or per-site alternatives: ((mjtGeom, (sx, sy, sz)), ...). Exactly one must
    # match the compiled model; shape changes are shared within an appearance group.
    SHAPE_KNOBS: dict[str, tuple[tuple[int, tuple[float, float, float]], ...]] = {}
    # ``group -> geom names`` whose explicit rgba a ``<group>.color`` leaf writes, overriding their material.
    GEOM_COLORS: dict[str, tuple[str, ...]] = {}
    # ``physics leaf -> sphere geom`` drawn at the leaf's radius per row.
    SPHERE_SIZES: dict[str, str] = {}

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
        render_options: dict | None = None,
    ):
        """Derive compact-row names and widths from the compiled model.

        ``backend="mjwarp"`` draws with MuJoCo Warp's ray tracer on CUDA. ``render_options`` sets its
        speed and fidelity options (see ``warp_render.Options``).
        """
        self.render_options = warp_render.Options.parse(render_options)
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
        if self.backend == "mjwarp" and self.device.type != "cuda":
            raise ValueError(f"{type(self).__name__}: backend='mjwarp' draws on CUDA; got device={self.device}")
        if self.backend == "mjwarp" and self.DOMAIN == "manipulator":
            raise ValueError(f"{type(self).__name__}: backend='mjwarp' does not draw the Manipulator cells")

    def render_semantic(self, states):
        """Per-pixel ``(objid, objtype)`` pairs of the cell's model and planar depth of compact rows.

        MuJoCo Warp's ray tracer draws them on CUDA at ``self.size``. Return ``(seg, depth)``: seg is
        (B, H, W, 2) int64 with -1 for the sky, depth is (B, H, W) float32.
        """
        if self.device.type != "cuda":
            raise ValueError(f"{type(self).__name__}.render_semantic draws with MuJoCo Warp on CUDA")
        return warp_render.semantic(self, states)

    def _ray_backend(self) -> str:
        """The CPU ray backend: ``"numba"`` or ``"torch"`` when the env names one, else ``"auto"``."""
        return self.backend if self.backend in ("numba", "torch") else "auto"

    def _close(self) -> None:
        """Drop instance CPU scenes and live MJWarp contexts; shared GL and MJWarp contexts remain cached."""
        self.__dict__.pop("_cpu_scenes", None)
        warp_render.release(self)

    @classmethod
    def _load_env(cls, seed: int):
        """Load the common source used for models, sampling and reference frames.

        Override here when a registration requires a model other than suite.load's.
        """
        return _suite().load(cls.DOMAIN, cls.TASK, task_kwargs={"random": seed})

    @classmethod
    def _bundle(cls, size: int):
        """Cache ``(env, model, data, pristine, renderer, slots)`` by domain, task and size.

        Episode probing applies task model edits and separates constant appearance
        from sampled fields. Its fixed seed makes the snapshot reproducible.
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
            # Restore shape arrays for every appearance, including domains without shape knobs.
            geom_type=np.array(m.geom_type, copy=True),
            geom_size=np.array(m.geom_size, copy=True),
            site_type=np.array(m.site_type, copy=True),
            site_size=np.array(m.site_size, copy=True),
        )
        for arr in {a for a, _ in slots}:
            pristine[arr] = np.array(getattr(m, arr), copy=True)
        # Reuse this GL context for the process lifetime; teardown can corrupt other contexts.
        r = mujoco_renderer(m, height=int(size), width=int(size))
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
        Use pristine alpha and groups so previous renders cannot change declarations.
        """
        import mujoco

        m, pr = cls._model(), cls._pristine()
        galpha = effective_alpha(m, "geom", rgba=pr["geom_rgba"], mat_rgba=pr["mat_rgba"])
        salpha = effective_alpha(m, "site", rgba=pr["site_rgba"], mat_rgba=pr["mat_rgba"])
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
    def _static_slot_rows(cls) -> dict:
        """Map world-body position slots to derived data rows requiring direct uploads.

        MJWarp kinematics skips world-body geoms/sites. Updating only their model
        positions leaves stale world poses; the identity parent frame permits direct copies.
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
                continue  # kinematics recomputes poses on non-world bodies
            rows.setdefault((kind, eid), [np.asarray(pr[arr], np.float64)[eid].copy(), []])[1].append(
                (j, int(k % _row_width(m, arr)))
            )
        cls._slot_rows_cache = rows
        return rows

    @classmethod
    def _shape_element(cls, name: str) -> tuple[str, int]:
        """Resolve a SHAPE_KNOBS name to a geom, or to a site when no geom has it."""
        import mujoco

        m = cls._model()
        for kind, obj in (("geom", mujoco.mjtObj.mjOBJ_GEOM), ("site", mujoco.mjtObj.mjOBJ_SITE)):
            ident = mujoco.mj_name2id(m, obj, name)
            if ident >= 0:
                return kind, ident
        raise RuntimeError(
            f"{cls.__name__}: SHAPE_KNOBS names {name!r}, which this model has as neither geom nor site"
        )

    @classmethod
    def _model(cls):
        return cls._bundle(cls.default_size)[1]

    @classmethod
    def _pristine(cls) -> dict:
        """Return the canonical snapshot from before appearance changes.

        The shared model retains the last rendered style. Read defaults from this
        snapshot to keep new instances independent of previous rendering.
        """
        return cls._bundle(cls.default_size)[3]

    @staticmethod
    def _qpos_names(m) -> tuple[str, ...]:
        """Name scalar positions and quaternion components from the model's joint layout."""
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
            alpha = effective_alpha(m, kind, rgba=pr[f"{kind}_rgba"], mat_rgba=pr["mat_rgba"])
            for i in range(n):
                # Group membership alone includes transparent markers with no visible pixels.
                if int(grp[i]) not in VISIBLE_GROUPS or alpha[i] <= 0:
                    continue
                mid = int(arr[i])
                if mid >= 0 and mid not in used:
                    used.append(mid)
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}" for i in used]

    #: Physics variation leaves of the simulator; the renderer keys appearance on the others.
    _physics_paths: frozenset[str] = frozenset()
    #: Physics leaves that also move pixels at a fixed state.
    FRAME_PHYSICS: frozenset[str] = frozenset()

    @property
    def dynamics_variation_leaves(self):
        return frozenset(self._physics_paths) - self.FRAME_PHYSICS

    def make_variation_space(self) -> Dict:
        """Derive supported appearance leaves and defaults from the pristine model.

        Materials use float colors in [0, 1]. Visibility is binary because the
        analytic renderer does not blend partial alpha. Shape choices must contain
        exactly one match for the compiled geometry, which becomes the default.
        """
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
                a = effective_alpha(m, kind, rgba=pr[f"{kind}_rgba"], mat_rgba=pr["mat_rgba"])
                # A body landmark is visible if any of its pristine geoms are visible.
                vis[name] = Discrete(2, init_value=int(any(float(a[i]) > 0 for i in eids)))
        if vis:
            space["visibility"] = Dict(vis)
        for group, geoms in self.GEOM_COLORS.items():
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geoms[0])
            mid, rgba = int(m.geom_matid[gid]), pr["geom_rgba"][gid]
            rgb = pr["mat_rgba"][mid][:3] if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]) else rgba[:3]
            space[group] = Dict({"color": Box(low=0.0, high=1.0, shape=(3,), init_value=[float(x) for x in rgb])})
        for gname, alts in self.SHAPE_KNOBS.items():
            kind, gid = self._shape_element(gname)
            ptype, psize = pr[f"{kind}_type"][gid], pr[f"{kind}_size"][gid]
            cur = [
                k
                for k, (t, s) in enumerate(alts)
                if int(t) == int(ptype) and np.allclose(np.asarray(s, np.float64), psize)
            ]
            if len(cur) != 1:
                raise RuntimeError(
                    f"{type(self).__name__}: {len(cur)} of the {len(alts)} declared shapes for "
                    f"{gname!r} match the compiled model (type {int(ptype)}, size "
                    f"{list(psize)}); exactly one must, so the default style renders the compiled model"
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

        Read qpos and episode-specific model entries from the same trajectory.
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

    @classmethod
    def reference_env(cls, size: int | None = None):
        """Construct a fresh source with its render resolution attached.

        Keep its GL context alive for the process lifetime to avoid cross-context
        renderer corruption on teardown.
        """
        env = cls._load_env(0)
        env.reset()
        env._se_size = int(size or cls.default_size)
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Restore compact rows in the source, verify them, and render the requested camera.

        Velocity is set to zero because these compact rows describe geometry only.
        """
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        size = int(getattr(env, "_se_size", 0) or cls.default_size)
        slots = cls._slots()
        nq = int(m.nq)
        out = []
        for st in np.asarray(states, np.float64):
            # Write model entries before reset_context runs forward kinematics, so derived
            # world poses use this row rather than the previous render's episode fields.
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
                    f"(max {np.abs(got - st).max():.3e})"
                )
            out.append(p.render(height=size, width=size, camera_id=cls.CAMERA).copy())
        return np.stack(out)
