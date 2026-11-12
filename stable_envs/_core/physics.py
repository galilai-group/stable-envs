"""Per-lane MuJoCo model parameters driven by Variation leaves.

Values travel in State under ``("physics", *leaf.split("."))``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

PHYSICS_KEY = "physics"

# Model fields whose change invalidates quantities that mj_setConst derives.
_SETCONST_INPUTS = frozenset(
    {
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "body_gravcomp",
        "dof_armature",
        "qpos0",
        "qpos_spring",
    }
)
# Statistics that depend on geometry only; mj_setConst would replace XML-declared values.
_GEOMETRIC_STATISTICS = ("center", "extent")
_OBJECTS = {
    "body": "mjOBJ_BODY",
    "geom": "mjOBJ_GEOM",
    "joint": "mjOBJ_JOINT",
    "site": "mjOBJ_SITE",
    "actuator": "mjOBJ_ACTUATOR",
    "tendon": "mjOBJ_TENDON",
    "light": "mjOBJ_LIGHT",
    "camera": "mjOBJ_CAMERA",
}


@dataclass(frozen=True)
class ModelField:
    """Write ``value * scale + offset`` into one model field.

    ``field`` is an ``MjModel`` array name or ``"opt.<name>"``. ``obj`` and
    ``name`` select an element row; a joint selects its dof row for ``dof_*``
    arrays and its qpos row for ``qpos*`` arrays. ``index`` selects components
    within the row; its length is the leaf width.
    """

    leaf: str
    field: str
    obj: str | None = None
    name: str | None = None
    index: int | tuple[int, ...] | None = None
    scale: float = 1.0
    offset: float = 0.0

    @property
    def width(self) -> int:
        return len(self.index) if isinstance(self.index, tuple) else 1


@dataclass(frozen=True)
class GeomMass:
    """Scale one primitive geom's mass by ``value / nominal``.

    ``per="density"`` reads the leaf as a density and ``per="mass"`` as the
    geom's explicit mass; both scale the geom's contribution to its body
    linearly. The body's mass, center of mass and principal inertia are
    recombined from that contribution and the rest of the body, so a body with
    several geoms moves its center of mass as a recompile would.
    """

    leaf: str
    geom: str
    nominal: float
    per: str = "density"
    width = 1


Knob = ModelField | GeomMass


@dataclass(frozen=True)
class PhysicsLayout:
    """Leaf names, widths, row offsets and nominal values of a physics value row."""

    leaves: tuple[str, ...]
    widths: tuple[int, ...]
    offsets: tuple[int, ...]
    nominal: tuple[float, ...]


def state_key(leaf: str) -> tuple[str, ...]:
    """Return the State key path that stores ``leaf``."""
    return (PHYSICS_KEY, *leaf.split("."))


def state_fields(knobs: Sequence[Knob]) -> dict[tuple[str, ...], int]:
    """Return ``{state key: width}`` for the leaves the knobs declare."""
    out: dict[tuple[str, ...], int] = {}
    for knob in knobs:
        out.setdefault(state_key(knob.leaf), int(knob.width))
    return out


def _mujoco():
    import mujoco

    return mujoco


def _skew_inertia(mass: float, com: np.ndarray) -> np.ndarray:
    return mass * (float(com @ com) * np.eye(3) - np.outer(com, com))


def _quat_mat(quat) -> np.ndarray:
    mj = _mujoco()
    out = np.empty(9)
    mj.mju_quat2Mat(out, np.asarray(quat, np.float64))
    return out.reshape(3, 3)


def _body_moments(mass, ipos, iquat, inertia):
    """Return mass, first moment and inertia tensor about the body frame origin."""
    rot = _quat_mat(iquat)
    about_com = rot @ np.diag(np.asarray(inertia, np.float64)) @ rot.T
    com = np.asarray(ipos, np.float64)
    return float(mass), mass * com, about_com + _skew_inertia(float(mass), com)


def _principal(mass, moment, tensor):
    """Return ``(mass, ipos, iquat, inertia)`` for moments about the body frame origin."""
    mj = _mujoco()
    com = moment / mass
    about_com = tensor - _skew_inertia(mass, com)
    about_com = 0.5 * (about_com + about_com.T)
    values, vectors = np.linalg.eigh(about_com)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] *= -1.0
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, np.ascontiguousarray(vectors).reshape(-1))
    return mass, com, quat, values


def _refresh_bvh(m, body: int) -> None:
    """Recompute a body's BVH boxes in its inertial frame, keeping the tree topology."""
    adr, num = int(m.body_bvhadr[body]), int(m.body_bvhnum[body])
    if adr < 0 or num <= 0:
        return
    inertial = _quat_mat(m.body_iquat[body])
    ipos = np.asarray(m.body_ipos[body], np.float64)
    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def box(local: int):
        node = adr + local
        geom = int(m.bvh_nodeid[node])
        left, right = (int(c) for c in m.bvh_child[node])
        if left < 0 and right < 0:
            rot = inertial.T @ _quat_mat(m.geom_quat[geom])
            centre = inertial.T @ (np.asarray(m.geom_pos[geom], np.float64) - ipos) + rot @ m.geom_aabb[geom, :3]
            half = np.abs(rot) @ np.asarray(m.geom_aabb[geom, 3:], np.float64)
            lo, hi = centre - half, centre + half
        else:
            parts = [box(c) for c in (left, right) if c >= 0]
            lo = np.min([p[0] for p in parts], axis=0)
            hi = np.max([p[1] for p in parts], axis=0)
        boxes[node] = (lo, hi)
        return lo, hi

    box(0)
    for node, (lo, hi) in boxes.items():
        m.bvh_aabb[node, :3] = 0.5 * (lo + hi)
        m.bvh_aabb[node, 3:] = 0.5 * (hi - lo)


def _unit_geom(m, geom: int):
    """Compile the geom alone at density 1 and return its moments in its body frame."""
    mj = _mujoco()
    gtype = int(m.geom_type[geom])
    if gtype in (int(mj.mjtGeom.mjGEOM_MESH), int(mj.mjtGeom.mjGEOM_SDF), int(mj.mjtGeom.mjGEOM_HFIELD)):
        raise NotImplementedError("GeomMass supports primitive geoms; mesh, sdf and hfield inertia are not derived")
    if gtype == int(mj.mjtGeom.mjGEOM_PLANE):
        raise ValueError("a plane has no mass")
    spec = mj.MjSpec()
    body = spec.worldbody.add_body()
    body.add_freejoint()
    g = body.add_geom()
    g.type = mj.mjtGeom(gtype)
    g.size = np.asarray(m.geom_size[geom], np.float64)
    g.pos = np.asarray(m.geom_pos[geom], np.float64)
    g.quat = np.asarray(m.geom_quat[geom], np.float64)
    g.density = 1.0
    unit = spec.compile()
    return _body_moments(unit.body_mass[1], unit.body_ipos[1], unit.body_iquat[1], unit.body_inertia[1])


class _ResolvedField:
    __slots__ = ("knob", "row", "index", "option")

    def __init__(self, m, knob: ModelField):
        mj = _mujoco()
        self.knob = knob
        self.index = knob.index
        self.option = knob.field.startswith("opt.")
        self.row = None
        if self.option:
            if not hasattr(m.opt, knob.field[4:]):
                raise KeyError(f"{knob.leaf}: MjOption has no field {knob.field[4:]!r}")
            return
        if not hasattr(m, knob.field):
            raise KeyError(f"{knob.leaf}: MjModel has no field {knob.field!r}")
        if knob.obj is None:
            return
        objtype = getattr(mj.mjtObj, _OBJECTS[knob.obj])
        ident = mj.mj_name2id(m, objtype, knob.name)
        if ident < 0:
            raise KeyError(f"{knob.leaf}: no {knob.obj} named {knob.name!r}")
        if knob.obj == "joint" and knob.field.startswith("dof_"):
            ident = int(m.jnt_dofadr[ident])
        elif knob.obj == "joint" and knob.field.startswith("qpos"):
            ident = int(m.jnt_qposadr[ident])
        self.row = ident

    def write(self, m, value) -> None:
        v = np.asarray(value, np.float64).reshape(-1) * self.knob.scale + self.knob.offset
        if self.option:
            name = self.knob.field[4:]
            current = getattr(m.opt, name)
            if np.ndim(current) == 0:
                setattr(m.opt, name, type(current)(v[0]))
                return
            target = current
        else:
            target = getattr(m, self.knob.field)
            if self.row is not None:
                if target.ndim == 1:
                    target[self.row] = v[0]
                    return
                target = target[self.row]
        if self.index is None:
            target[...] = v.reshape(np.shape(target)) if v.size == np.size(target) else v[0]
        elif isinstance(self.index, tuple):
            target[list(self.index)] = v
        else:
            target[self.index] = v[0]


class _ResolvedMass:
    __slots__ = ("knob", "body", "part", "unscaled_mass")

    def __init__(self, m, knob: GeomMass, compile_scale: float):
        mj = _mujoco()
        if knob.per not in ("density", "mass"):
            raise ValueError(f"{knob.leaf}: per must be 'density' or 'mass', got {knob.per!r}")
        if not float(knob.nominal) > 0:
            raise ValueError(f"{knob.leaf}: the nominal value must be positive")
        geom = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, knob.geom)
        if geom < 0:
            raise KeyError(f"{knob.leaf}: no geom named {knob.geom!r}")
        self.knob = knob
        self.body = int(m.geom_bodyid[geom])
        unit = _unit_geom(m, geom)
        k = float(knob.nominal) if knob.per == "density" else float(knob.nominal) / unit[0]
        self.unscaled_mass = k * unit[0]
        self.part = tuple(compile_scale * k * np.asarray(x) for x in unit)


class PhysicsKnobs:
    """Resolved physics declarations for one compiled model.

    A model compiled with ``settotalmass`` passes ``total_mass`` and the body mass
    sum the compiler saw before rescaling (``unscaled_total``); a mass change then
    rescales every body as that compiler option does.
    """

    def __init__(
        self, knobs: Sequence[Knob], model, *, total_mass: float | None = None, unscaled_total: float | None = None
    ):
        if (total_mass is None) != (unscaled_total is None):
            raise ValueError("total_mass and unscaled_total are declared together")
        self.total_mass = None if total_mass is None else float(total_mass)
        self._compile_scale = 1.0 if total_mass is None else float(total_mass) / float(unscaled_total)
        self._unscaled_total = None if unscaled_total is None else float(unscaled_total)
        self.knobs = tuple(knobs)
        leaves: dict[str, int] = {}
        for knob in self.knobs:
            width = int(knob.width)
            if leaves.setdefault(knob.leaf, width) != width:
                raise ValueError(f"{knob.leaf}: declarations disagree on the leaf width")
        self.leaves = tuple(leaves)
        self.widths = tuple(leaves.values())
        self.offsets = tuple(int(x) for x in np.cumsum((0, *self.widths))[:-1])
        self.dim = int(sum(self.widths))
        self._fields = [_ResolvedField(model, k) for k in self.knobs if isinstance(k, ModelField)]
        self._masses = [_ResolvedMass(model, k, self._compile_scale) for k in self.knobs if isinstance(k, GeomMass)]
        # Bodies whose whole mass comes from declared geoms; one shared factor then scales them exactly.
        declared: dict[int, list] = {}
        for mass in self._masses:
            declared.setdefault(mass.body, []).append(float(mass.part[0]))
        self._whole = {
            body: len(parts)
            for body, parts in declared.items()
            if abs(sum(parts) - float(model.body_mass[body])) <= 1e-9 * float(model.body_mass[body])
        }
        self._derived = bool(self._masses) or any(f.knob.field in _SETCONST_INPUTS for f in self._fields)
        self._nominal_model = copy.copy(model)
        self.nominal = self._read_nominal(model)

    def _read_nominal(self, m) -> np.ndarray:
        row = np.zeros(self.dim)
        for f in self._fields:
            k, off = f.knob, self.offsets[self.leaves.index(f.knob.leaf)]
            source = getattr(m.opt, k.field[4:]) if f.option else getattr(m, k.field)
            if f.row is not None:
                source = source[f.row]
            raw = np.asarray(source, np.float64)
            raw = raw[list(f.index)] if isinstance(f.index, tuple) else raw if f.index is None else raw[f.index]
            row[off : off + k.width] = (np.asarray(raw).reshape(-1)[: k.width] - k.offset) / k.scale
        for mass in self._masses:
            row[self.offsets[self.leaves.index(mass.knob.leaf)]] = float(mass.knob.nominal)
        return row

    def as_row(self, values: Mapping[str, object] | np.ndarray | None) -> np.ndarray:
        """Return a ``(dim,)`` row: the nominal values, a full row, or nominal values with mapping entries replaced."""
        if values is None:
            return self.nominal.copy()
        if not isinstance(values, Mapping):
            row = np.asarray(values, np.float64).reshape(self.dim)
            return row.copy()
        row = self.nominal.copy()
        unknown = set(values) - set(self.leaves)
        if unknown:
            raise KeyError(f"not physics leaves: {sorted(unknown)}; declared: {list(self.leaves)}")
        for leaf, value in values.items():
            i = self.leaves.index(leaf)
            row[self.offsets[i] : self.offsets[i] + self.widths[i]] = np.asarray(value, np.float64).reshape(-1)
        return row

    def apply(self, model, row, data=None) -> None:
        """Write ``row`` into ``model`` and recompute the constants that depend on the changed fields."""
        mj = _mujoco()
        row = np.asarray(row, np.float64).reshape(self.dim)
        if not np.isfinite(row).all():
            raise ValueError(f"physics values must be finite, got {row}")
        for f in self._fields:
            off = self.offsets[self.leaves.index(f.knob.leaf)]
            f.write(model, row[off : off + f.knob.width])
        if not self._derived:
            return
        self._apply_masses(model, row)
        keep = {name: np.array(getattr(model.stat, name)) for name in _GEOMETRIC_STATISTICS}
        data = mj.MjData(model) if data is None else data
        mj.mj_setConst(model, data)
        if mj.mj_version() >= 3017000:
            # Since 3.17, setConst leaves compiler statistics unchanged. Match a
            # recompile after mass variations (also consumed by MJX Reacher).
            full = np.zeros((model.nv, model.nv))
            mj.mj_fullM(model, data, full)
            if model.nv:
                model.stat.meaninertia = np.cumsum(np.diag(full))[-1] / model.nv
            if model.nbody > 1:
                model.stat.meanmass = np.cumsum(model.body_mass[1:])[-1] / (model.nbody - 1)
        for name, value in keep.items():
            if value.ndim:
                getattr(model.stat, name)[:] = value
            else:
                setattr(model.stat, name, float(value))

    def _apply_masses(self, model, row) -> None:
        if not self._masses:
            return
        n = self._nominal_model
        reset = range(1, n.nbody) if self.total_mass is not None else {mass.body for mass in self._masses}
        for b in reset:
            for name in ("body_mass", "body_ipos", "body_iquat", "body_inertia"):
                getattr(model, name)[b] = getattr(n, name)[b]
            adr, num = int(n.body_bvhadr[b]), int(n.body_bvhnum[b])
            if adr >= 0 and num > 0:
                model.bvh_aabb[adr : adr + num] = n.bvh_aabb[adr : adr + num]
        changed: dict[int, list] = {}
        extra = 0.0
        for mass in self._masses:
            value = float(row[self.offsets[self.leaves.index(mass.knob.leaf)]])
            if not value > 0:
                raise ValueError(f"{mass.knob.leaf} must be positive, got {value}")
            s = value / float(mass.knob.nominal)
            if s != 1.0:
                changed.setdefault(mass.body, []).append((mass, s))
                extra += (s - 1.0) * mass.unscaled_mass
        for body, entries in changed.items():
            factors = {s for _, s in entries}
            if len(entries) == self._whole.get(body) and len(factors) == 1:
                # Scaling every contribution keeps the inertial frame; re-diagonalizing could rotate it.
                factor = factors.pop()
                model.body_mass[body] = n.body_mass[body] * factor
                model.body_inertia[body] = n.body_inertia[body] * factor
                continue
            total = list(_body_moments(n.body_mass[body], n.body_ipos[body], n.body_iquat[body], n.body_inertia[body]))
            for mass, s in entries:
                total = [t + (s - 1.0) * p for t, p in zip(total, mass.part, strict=True)]
            body_mass, ipos, iquat, inertia = _principal(*total)
            model.body_mass[body] = body_mass
            model.body_ipos[body] = ipos
            model.body_iquat[body] = iquat
            model.body_inertia[body] = inertia
        if self.total_mass is not None and extra:
            ratio = self._unscaled_total / (self._unscaled_total + extra)
            model.body_mass[1:] *= ratio
            model.body_inertia[1:] *= ratio
        for body in changed:
            _refresh_bvh(model, body)


class PhysicsLanes:
    """One physics value row per CPU model."""

    def __init__(self, knobs: Sequence[Knob], models: Sequence, **compile_options):
        if not models:
            raise ValueError("PhysicsLanes needs at least one model")
        self.spec = PhysicsKnobs(knobs, models[0], **compile_options)
        self.models = list(models)
        self._data = [None] * len(self.models)
        self.values = np.tile(self.spec.nominal, (len(self.models), 1))

    @property
    def num_envs(self) -> int:
        return len(self.models)

    @property
    def layout(self) -> PhysicsLayout:
        spec = self.spec
        return PhysicsLayout(spec.leaves, spec.widths, spec.offsets, tuple(float(x) for x in spec.nominal))

    def _apply(self, lane: int, row: np.ndarray) -> None:
        mj = _mujoco()
        if self._data[lane] is None and self.spec._derived:
            self._data[lane] = mj.MjData(self.models[lane])
        self.spec.apply(self.models[lane], row, self._data[lane])

    def configure(self, values) -> None:
        """Install ``(num_envs, dim)`` rows, one shared row, or a mapping of leaf overrides."""
        if isinstance(values, Mapping):
            rows = self.values.copy()
            for leaf, value in values.items():
                i = self.spec.leaves.index(leaf)
                o, w = self.spec.offsets[i], self.spec.widths[i]
                rows[:, o : o + w] = np.broadcast_to(np.asarray(value, np.float64).reshape(-1, w), (len(rows), w))
        else:
            rows = np.broadcast_to(np.asarray(values, np.float64).reshape(-1, self.spec.dim), self.values.shape)
        if not np.isfinite(rows).all():
            raise ValueError("physics values must be finite")
        for lane, row in enumerate(rows):
            if not np.array_equal(row, self.values[lane]):
                self._apply(lane, row)
        self.values = np.array(rows, np.float64)

    def state_fields(self) -> dict[tuple[str, ...], int]:
        """Return ``{state key: width}`` for each physics leaf."""
        return {state_key(leaf): w for leaf, w in zip(self.spec.leaves, self.spec.widths, strict=True)}

    def export(self) -> dict[tuple[str, ...], np.ndarray]:
        """Return a copy of the installed values as ``{state key: (num_envs, width)}``."""
        return {
            state_key(leaf): self.values[:, o : o + w].copy()
            for leaf, o, w in zip(self.spec.leaves, self.spec.offsets, self.spec.widths, strict=True)
        }

    def rows_from_state(self, state) -> np.ndarray:
        """Read ``(num_envs, dim)`` physics rows from State; every physics field must be present."""
        keys = set(state.keys(True, True))
        missing = [".".join(k) for k in self.state_fields() if k not in keys]
        if missing:
            raise ValueError(
                f"State has no {missing}: it does not record the physics it was produced under, and "
                f"restoring it would keep this environment's current physics instead. Add the fields "
                f"(the registered values are {dict(zip(self.spec.leaves, self.spec.nominal.tolist()))}) "
                f"or restore a State this environment produced."
            )
        rows = np.empty((len(self.models), self.spec.dim))
        for leaf, o, w in zip(self.spec.leaves, self.spec.offsets, self.spec.widths, strict=True):
            value = state[state_key(leaf)]
            value = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            rows[:, o : o + w] = np.asarray(value, np.float64).reshape(len(self.models), w)
        return rows
