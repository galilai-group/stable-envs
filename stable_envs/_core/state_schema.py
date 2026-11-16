"""State schemas: the named fields of a State, stored in one tensor per group.

A State with one leaf per named field costs one tensor and one TensorDict node per field on every export, slice,
copy and device move. With a ``StateSchema``, a State has one ``(*batch, size)`` leaf per group. A group holds
fields of one dtype side by side, and a field is a slice of its group. A field whose shape changes from one State
to the next, such as an action history, is a "kept" field: it stays a leaf of its own, and ``set`` can give it a
new width.

The schema is the only code that knows the layout. Everything else reads and writes fields by name, with
``StateSchema.get`` / ``StateSchema.set``, or with ``get_field`` / ``set_field`` when it has only the State.

Adoption
--------

1. Declare the fields once, in State order. A field names its group; a group path ends in a dtype name by
   convention, so one section can hold several dtypes::

       from stable_envs._core.state_schema import Field, StateSchema

       STATE = StateSchema(
           "Reacher",
           [
               Field("physics/qpos", (2,), torch.float64, "physics/float64"),
               Field("physics/qvel", (2,), torch.float64, "physics/float64"),
               Field("physics/time", (), torch.float64, "physics/float64"),
               Field("task/target", (2,), torch.float32, "task/float32"),
               Field("rng/key", (2,), torch.int64, "rng/int64"),
           ],
       )

   A layout that depends on the configuration (for example on a shape count) declares a schema per
   configuration. Equal declarations share one registry entry. ``schema_of`` finds only declared schemas, and a
   State loaded from disk carries no schema. So declare the schema when the module loads, or, for a layout that
   depends on the configuration, when the environment is built, for every layout that a saved State can have.

2. Keep the live simulator state in group buffers and write kernels into them. Cut the field views once, when
   the buffers are made, not on every step::

       live = STATE.new(num_envs, device)            # one zero tensor per group
       buffers = STATE.buffers(live)                 # {"physics/float64": (num_envs, 5) tensor, ...}
       qpos = STATE.views(live, ["physics/qpos"])["physics/qpos"]   # a view; writes change ``live``
       kernel(buffers["physics/float64"], ...)        # a kernel writes its whole group

   ``get_state`` then returns ``live.clone()``: one copy per group. A family that exports through
   ``StateLayout`` passes ``STATE.leaves()`` as its leaves and ``STATE.pack(values)`` as its values.

3. Replace raw leaf access. ``state["physics", "qpos"]`` becomes ``STATE.get(state, "physics/qpos")``, or
   ``get_field(state, "physics.qpos")`` in code that has only the State (renderers, tests, examples). Writes use
   ``STATE.set`` / ``set_field``. ``set_state`` calls ``STATE.validate(state)`` first, which names a missing
   group, a wrong shape or a wrong dtype. ``tests/unit/test_state_schema.py`` fails on raw leaf indexing of an
   adopted family; add the family to ``ADOPTED`` there.

Batch indexing, slicing, stacking, ``torch.cat``, ``clone`` and ``.to(device)`` are plain TensorDict operations on
a folded State; every field read works on the result, with the extra batch axes in front.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from ..specs import CompositeSpec, Spec
from .tensordicts import build_tensordict

__all__ = [
    "Field",
    "FieldSlice",
    "Group",
    "StateSchema",
    "get_field",
    "raw_field_access",
    "schema_of",
    "set_field",
    "unfold",
]


class Field(NamedTuple):
    """The declaration of one named State field.

    Attributes:
        name: The field name: State key parts joined by ``/``, such as ``"current/body/agent/position"``. A part
            must not contain ``.``.
        shape: Per-environment shape, ``()`` for a scalar. None for a kept field whose shape changes.
        dtype: Torch dtype of the values.
        group: ``/``-joined path of the group leaf that holds the field, or None to keep the field as a leaf of its
            own. A field with ``shape`` None must have ``group`` None.
    """

    name: str
    shape: tuple[int, ...] | None
    dtype: torch.dtype
    group: str | None


class FieldSlice(NamedTuple):
    """Where a grouped field lives: ``group[..., start:stop]``, reshaped to ``(*batch, *shape)``.

    Attributes:
        name: The field name.
        group: The group key.
        start: First column of the field in its group.
        stop: Column after the last one.
        shape: Per-environment shape.
        dtype: Torch dtype, which is the group's dtype.
    """

    name: str
    group: str
    start: int
    stop: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class Group(NamedTuple):
    """One group leaf of a State: ``(*batch, size)`` values of one dtype.

    Attributes:
        key: ``/``-joined path of the leaf.
        path: The leaf's key as a tuple.
        dtype: Torch dtype of every field in the group.
        size: Number of columns.
        fields: The group's fields, in column order.
    """

    key: str
    path: tuple[str, ...]
    dtype: torch.dtype
    size: int
    fields: tuple[FieldSlice, ...]


#: One schema per distinct layout, and the schemas that hold each ``(group path, size, dtype)``.
_SCHEMAS: dict[tuple, StateSchema] = {}
_BY_GROUP: dict[tuple[tuple[str, ...], int, torch.dtype], list[StateSchema]] = {}


def _path(key: str) -> tuple[str, ...]:
    return tuple(key.split("/"))


def _dtype_of(value) -> torch.dtype | None:
    if torch.is_tensor(value):
        return value.dtype
    if isinstance(value, np.ndarray | np.generic):
        return torch.from_numpy(np.empty(0, value.dtype)).dtype
    return None


def _safe(source: torch.dtype, dtype: torch.dtype) -> bool:
    """Whether every ``source`` value has an exact ``dtype`` value (NumPy's ``"safe"`` casting rule)."""
    try:
        return np.can_cast(torch.empty(0, dtype=source).numpy().dtype, torch.empty(0, dtype=dtype).numpy().dtype)
    except TypeError:
        return False


def _cast(what: str, value, dtype: torch.dtype, device) -> torch.Tensor:
    """``value`` as a ``dtype`` tensor on ``device``; an array must cast without loss, and a Python value exactly.

    Raises:
        TypeError: If ``value`` would change.
    """
    source = _dtype_of(value)
    if source is not None:
        if source != dtype and not _safe(source, dtype):
            raise TypeError(f"{what} is {dtype}; {source} doesn't cast to it safely")
        return torch.as_tensor(value, device=device).to(dtype)
    given = torch.as_tensor(value, device=device)
    cast = given.to(dtype)
    back = cast.to(given.dtype)
    if not bool(((back == given) | (given != given)).all()):
        raise TypeError(f"{what} is {dtype}; {value!r} doesn't fit it exactly")
    return cast


class StateSchema:
    """The layout of a folded State: groups of fields, and kept fields.

    Fields are packed into their groups in declaration order. Groups and kept fields are State leaves in the order
    of their first field.

    Args:
        name: Name used in error messages, such as the family name.
        fields: Every field of the State, as ``Field`` or ``(name, shape, dtype, group)`` tuples.

    Raises:
        ValueError: If a name repeats or contains ``.``, a group mixes dtypes, a kept field is given a group, or a
            group key is also a field name.
    """

    def __init__(self, name: str, fields: Iterable[Field | tuple]):
        self.name = name
        columns: dict[str, list] = {}
        dtypes: dict[str, torch.dtype] = {}
        #: Every grouped field by name.
        self.slices: dict[str, FieldSlice] = {}
        #: Kept fields by name, with their dtypes.
        self.kept: dict[str, torch.dtype] = {}
        leaves: list[str] = []
        for declared in fields:
            item = Field(*declared)
            if "." in item.name or item.name in self.slices or item.name in self.kept:
                raise ValueError(f"{name} State field {item.name!r} repeats or contains '.'")
            if item.group is None:
                self.kept[item.name] = item.dtype
                leaves.append(item.name)
                continue
            if item.shape is None:
                raise ValueError(f"{name} State field {item.name!r} has no shape, so it can't be in a group")
            if dtypes.setdefault(item.group, item.dtype) != item.dtype:
                raise ValueError(f"{name} State group {item.group!r} mixes {dtypes[item.group]} and {item.dtype}")
            if item.group not in columns:
                columns[item.group] = []
                leaves.append(item.group)
            shape = tuple(int(n) for n in item.shape)
            start = sum(s.stop - s.start for s in columns[item.group])
            entry = FieldSlice(item.name, item.group, start, start + math.prod(shape), shape, item.dtype)
            columns[item.group].append(entry)
            self.slices[item.name] = entry
        clash = sorted(set(columns) & (set(self.slices) | set(self.kept)))
        if clash:
            raise ValueError(f"{name} State group keys are also field names: {clash}")
        #: Every group by key.
        self.groups: dict[str, Group] = {
            key: Group(key, _path(key), dtypes[key], columns[key][-1].stop if columns[key] else 0, tuple(columns[key]))
            for key in columns
        }
        #: Group keys and kept field names, in State order.
        self.order: tuple[str, ...] = tuple(leaves)
        #: The layout, which equal schemas share.
        self.signature = (
            tuple((g.key, g.dtype, g.fields) for g in self.groups.values()),
            tuple(self.kept.items()),
            self.order,
        )
        if _SCHEMAS.setdefault(self.signature, self) is self:
            for group in self.groups.values():
                _BY_GROUP.setdefault((group.path, group.size, group.dtype), []).append(self)

    def __eq__(self, other) -> bool:
        return isinstance(other, StateSchema) and self.signature == other.signature

    def __hash__(self) -> int:
        return hash(self.signature)

    def __repr__(self) -> str:
        return f"StateSchema({self.name!r}, {len(self.slices) + len(self.kept)} fields, {len(self.groups)} groups)"

    @property
    def names(self) -> list[str]:
        """Every field name: the grouped fields, then the kept ones."""
        return [*self.slices, *self.kept]

    def name_of(self, name: str) -> str:
        """The declared form of ``name``, which may join its parts with ``.`` instead of ``/``.

        Raises:
            KeyError: If the State has no such field.
        """
        if name in self.slices or name in self.kept:
            return name
        name = name.replace(".", "/")
        if name in self.slices or name in self.kept:
            return name
        raise KeyError(f"{self.name} State has no field {name!r}")

    def leaves(self) -> list[tuple[str, tuple[int, ...] | None, torch.dtype]]:
        """``(key, per-env shape, dtype)`` of every State leaf in State order, as ``StateLayout`` takes them.

        A kept field has shape None.
        """
        return [
            (key, (self.groups[key].size,), self.groups[key].dtype)
            if key in self.groups
            else (key, None, self.kept[key])
            for key in self.order
        ]

    def new(self, batch: int | Sequence[int], device: torch.device | str = "cpu") -> TensorDict:
        """A State of zeros on ``device``: one tensor per group, and kept fields with no columns.

        A kept field gets shape ``(*batch, 0)``; the family replaces it with its value.

        Args:
            batch: Number of rows, or the batch shape.
            device: Device of the tensors.
        """
        batch = (int(batch),) if isinstance(batch, int) else tuple(int(n) for n in batch)
        tree: dict = {}
        for key, shape, dtype in self.leaves():
            node = tree
            path = _path(key)
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = torch.zeros((*batch, *(shape or (0,))), dtype=dtype, device=device)
        return build_tensordict(tree, list(batch), device)

    def buffers(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        """``{group key: tensor}`` of the groups of ``state``; kernels write into these tensors.

        Raises:
            KeyError: If ``state`` lacks a group.
        """
        out = {}
        for key, group in self.groups.items():
            value = state.get(group.path, None)
            if value is None:
                raise KeyError(f"{self.name} State has no group {key!r}")
            out[key] = value
        return out

    def views(
        self,
        state: TensorDictBase | Mapping,
        names: Iterable[str] | None = None,
        *,
        numpy: bool = False,
    ) -> dict:
        """``{name: view}`` of the named fields of ``state``, by default every field.

        A view shares memory with its group, so writes to it change ``state``. Cut views once per buffer and keep
        them; each call slices again.

        Args:
            state: A folded State, or a ``{group key or kept field name: array}`` mapping such as ``buffers`` or
                ``pack`` returns.
            names: Field names (``/`` or ``.`` joined); None takes every field.
            numpy: Return NumPy views of host tensors, which cost less per field than tensor views.

        Raises:
            KeyError: If a name is unknown or ``state`` lacks its group.
        """
        cut: dict = {}

        def leaf(key):
            if key not in cut:
                if isinstance(state, TensorDictBase):
                    value = state.get(_path(key), None)
                else:
                    value = state.get(key)
                if value is None:
                    raise KeyError(f"{self.name} State has no leaf {key!r}")
                cut[key] = value.numpy() if numpy and torch.is_tensor(value) else value
            return cut[key]

        out = {}
        for name in self.names if names is None else names:
            name = self.name_of(name)
            if name in self.kept:
                out[name] = leaf(name)
                continue
            where = self.slices[name]
            block = leaf(where.group)
            if not where.shape:
                out[name] = block[..., where.start]
            else:
                out[name] = block[..., where.start : where.stop].reshape(*block.shape[:-1], *where.shape)
        return out

    def get(self, state: TensorDictBase, name: str) -> torch.Tensor:
        """The field ``name`` of ``state``, shaped ``(*batch, *shape)``, as a view: writes change ``state``.

        Args:
            state: A folded State.
            name: Field name, with parts joined by ``/`` or ``.``.

        Raises:
            KeyError: If the field or its group is missing.
        """
        return self.views(state, [name])[self.name_of(name)]

    def set(self, state: TensorDictBase, name: str, value) -> None:
        """Write ``value`` into the field ``name`` of ``state``, broadcast to the field's shape.

        A kept field whose ``value`` has as many dims as the field is replaced by a copy of ``value``, so its
        non-batch sizes can change, as when an action history grows. Its batch sizes must stay the same.

        Args:
            state: A folded State.
            name: Field name, with parts joined by ``/`` or ``.``.
            value: Tensor, array or number. Its dtype must cast to the field's dtype without loss.

        Raises:
            KeyError: If the field or its group is missing.
            TypeError: If ``value`` doesn't cast to the field's dtype without loss.
            ValueError: If ``value`` doesn't broadcast to the field's shape, or a kept field's new value changes
                the batch sizes.
        """
        field = self.get(state, name)
        name = self.name_of(name)
        if name in self.kept:
            value = _cast(f"{self.name} State field {name!r}", value, field.dtype, field.device)
            if value.ndim == field.ndim:
                rank = len(state.batch_size)
                if value.shape[:rank] != field.shape[:rank]:
                    raise ValueError(
                        f"{self.name} State field {name!r} has batch {tuple(field.shape[:rank])}; "
                        f"got {tuple(value.shape)}"
                    )
                state.set(_path(name), value.clone())
                return
        field.copy_(self._coerce(name, value, field.dtype, field.device, field.shape))

    def _coerce(self, name, value, dtype, device, shape) -> torch.Tensor:
        value = _cast(f"{self.name} State field {name!r}", value, dtype, device)
        try:
            return value.expand(shape)
        except RuntimeError:
            raise ValueError(
                f"{self.name} State field {name!r} has shape {tuple(shape)}; got {tuple(value.shape)}"
            ) from None

    def pack(self, values: Mapping[str, torch.Tensor | np.ndarray], groups: Iterable[str] | None = None) -> dict:
        """``{group key: (rows, size) array}`` of named ``values``, NumPy or torch like the values.

        This is the export path: it doesn't check shapes or dtypes. ``fold`` checks them.

        Args:
            values: ``{field name: (rows, *shape) array}`` for every field of each packed group.
            groups: Group keys to pack; None packs every group.
        """
        out = {}
        for key in self.groups if groups is None else groups:
            parts = [values[f.name] for f in self.groups[key].fields]
            rows = len(parts[0])
            flat = [part.reshape(rows, -1) for part in parts]
            out[key] = np.concatenate(flat, 1) if isinstance(parts[0], np.ndarray) else torch.cat(flat, 1)
        return out

    def fold(
        self,
        values: Mapping[str, torch.Tensor | np.ndarray] | TensorDictBase,
        batch_size: Sequence[int],
        device: torch.device | str = "cpu",
    ) -> TensorDict:
        """A new folded State from the named values of every field.

        Args:
            values: ``{field name: (*batch_size, *shape) value}``, or a TensorDict with one leaf per field such as
                ``unfold`` returns.
            batch_size: Batch shape of the State.
            device: Device of the State.

        Raises:
            KeyError: If a field is missing or a name is unknown.
            TypeError: If a value doesn't cast to its field's dtype without loss.
            ValueError: If a value has the wrong shape.
        """
        if isinstance(values, TensorDictBase):
            values = {"/".join(k if isinstance(k, tuple) else (k,)): v for k, v in values.items(True, True)}
        values = {self.name_of(k): v for k, v in values.items()}
        missing = [name for name in self.names if name not in values]
        if missing:
            raise KeyError(f"{self.name} State is missing fields {missing}")
        batch = tuple(int(n) for n in batch_size)
        state = self.new(batch, device)
        for name, value in values.items():
            if name in self.kept:
                kept = torch.as_tensor(value, device=device)
                if kept.dtype != self.kept[name] or tuple(kept.shape[: len(batch)]) != batch:
                    raise ValueError(f"{self.name} State field {name!r} must be {self.kept[name]} (*{batch}, ...)")
                state.set(_path(name), kept.clone())
                continue
            where = self.slices[name]
            field = self.get(state, name)
            value_shape = tuple(value.shape) if hasattr(value, "shape") else ()
            if value_shape != (*batch, *where.shape):
                raise ValueError(
                    f"{self.name} State field {name!r} must be {(*batch, *where.shape)}, got {value_shape}"
                )
            field.copy_(self._coerce(name, value, where.dtype, field.device, field.shape))
        return state

    def unfold(self, state: TensorDictBase) -> TensorDict:
        """A TensorDict with one leaf per field of ``state``; each leaf is a view of its group."""
        tree: dict = {}
        for name, value in self.views(state).items():
            node = tree
            path = _path(name)
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = value
        return build_tensordict(tree, state.batch_size, state.device)

    def validate(self, state: TensorDictBase, *, finite: bool = False) -> None:
        """Check that ``state`` has exactly this layout.

        Args:
            state: The State to check, with any batch shape.
            finite: Also require finite values in every floating-point field.

        Raises:
            TypeError: If ``state`` isn't a TensorDict.
            ValueError: If a group or kept field is missing or has the wrong size or dtype, ``state`` has leaves
                outside the schema, or ``finite`` is set and a field holds NaN or infinity. The message names the
                group or field.
        """
        if not isinstance(state, TensorDictBase):
            raise TypeError(f"{self.name} State must be a TensorDict, got {type(state).__name__}")
        rank = len(state.batch_size)
        for key, group in self.groups.items():
            value = state.get(group.path, None)
            if not torch.is_tensor(value):
                raise ValueError(f"{self.name} State is missing the group {key!r}")
            if value.dtype != group.dtype or value.ndim != rank + 1 or value.shape[-1] != group.size:
                raise ValueError(
                    f"{self.name} State group {key!r} must be {group.dtype} (*batch, {group.size}), "
                    f"got {value.dtype} {tuple(value.shape)}"
                )
        for name, dtype in self.kept.items():
            value = state.get(_path(name), None)
            if not torch.is_tensor(value) or value.dtype != dtype or value.ndim < rank:
                raise ValueError(f"{self.name} State field {name!r} must be a {dtype} tensor")
        known = set(map(_path, self.order))
        extra = sorted(
            "/".join(k)
            for k in (k if isinstance(k, tuple) else (k,) for k in state.keys(True, True))
            if k not in known
        )
        if extra:
            raise ValueError(f"{self.name} State has leaves outside its schema: {extra}")
        if finite:
            for key in self.order:
                value = state.get(_path(key))
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    names = [key] if key in self.kept else [f.name for f in self.groups[key].fields]
                    bad = next(n for n in names if not bool(torch.isfinite(self.get(state, n)).all()))
                    raise ValueError(f"{self.name} State field {bad!r} contains non-finite values")

    def spec(self, device: torch.device | str | None = None) -> CompositeSpec:
        """Specs of the group leaves, without the batch axis. Kept fields have no fixed shape, so they are left out.

        Args:
            device: Device recorded in the specs.
        """
        tree: dict = {}
        for group in self.groups.values():
            node = tree
            for part in group.path[:-1]:
                node = node.setdefault(part, {})
            node[group.path[-1]] = Spec((group.size,), dtype=group.dtype, device=device)
        return CompositeSpec(tree, device=device)


def schema_of(state: TensorDictBase) -> StateSchema | None:
    """The schema of a folded ``state``, found from its group leaves; None for a State with one leaf per field.

    The lookup needs no metadata in the State: each declared group is registered by ``(path, size, dtype)``.

    Raises:
        ValueError: If the groups of ``state`` fit declared schemas that disagree on the layout of those groups.
    """
    found: list[StateSchema] | None = None
    rank = len(state.batch_size)
    present = []
    for key in state.keys(True, True):
        path = key if isinstance(key, tuple) else (key,)
        value = state.get(path)
        if not torch.is_tensor(value) or value.ndim != rank + 1:
            continue
        schemas = _BY_GROUP.get((path, value.shape[-1], value.dtype))
        if schemas is None:
            continue
        present.append("/".join(path))
        found = list(schemas) if found is None else [s for s in found if any(s is t for t in schemas)]
    if not found:
        return None
    layouts = {tuple(s.groups[key] for key in present) for s in found}
    if len(layouts) > 1:
        raise ValueError(f"State groups {present} fit several StateSchemas: {[s.name for s in found]}")
    return found[0]


def get_field(state: TensorDictBase, name: str) -> torch.Tensor:
    """The field ``name`` of any State, shaped ``(*batch, *shape)``.

    On a folded State the result is a view of its group, so writes to it change the State. On a State with one
    leaf per field it is that leaf.

    Args:
        state: A State, with any batch shape.
        name: Field name, with parts joined by ``/`` or ``.``, such as ``"current.body.agent.position"``.

    Raises:
        KeyError: If ``state`` has no field ``name``.
    """
    schema = schema_of(state)
    if schema is not None:
        return schema.get(state, name)
    value = state.get(_path(name.replace(".", "/")), None)
    if not torch.is_tensor(value):
        raise KeyError(
            f"State has no field {name!r}, and its leaves match no StateSchema declared in this process; "
            "build the environment that made the State first"
        )
    return value


def set_field(state: TensorDictBase, name: str, value) -> None:
    """Write ``value`` into the field ``name`` of any State, broadcast to the field's shape.

    On a folded State, a kept field can take a new non-batch shape, as ``StateSchema.set`` says.

    Args:
        state: A State, with any batch shape.
        name: Field name, as for ``get_field``.
        value: Tensor, array or number. Its dtype must cast to the field's dtype without loss.

    Raises:
        KeyError: If ``state`` has no field ``name``.
        TypeError: If ``value`` doesn't cast to the field's dtype without loss.
        ValueError: If ``value`` doesn't broadcast to the field's shape, or a kept field's new value changes the
            batch sizes.
    """
    schema = schema_of(state)
    if schema is not None:
        schema.set(state, name, value)
        return
    field = get_field(state, name)
    field.copy_(_cast(f"State field {name!r}", value, field.dtype, field.device).expand(field.shape))


def unfold(state: TensorDictBase) -> TensorDictBase:
    """A TensorDict with one leaf per field of ``state``, each a view; a State that isn't folded comes back as is."""
    schema = schema_of(state)
    return state if schema is None else schema.unfold(state)


def raw_field_access(paths: Iterable[str | Path], sections: Iterable[str]) -> list[str]:
    """``"path:line"`` of each line in ``paths`` that indexes a State leaf by key under one of ``sections``.

    It finds ``state["section", ...]``, ``state["section"]`` and ``.get``, ``.set``, ``.select``, ``.exclude`` or
    ``.pop`` calls with such a key. A string key that joins names with ``/``, such as ``row["section/field"]``, reads
    a mapping of named fields, not a State, so it isn't reported. A line that also calls ``unfold(`` reads a State
    with one leaf per field and is allowed. The schema module itself is never reported.

    Args:
        paths: Python files to scan.
        sections: Top-level State keys of an adopted family, such as ``("current", "history")``.
    """
    names = "|".join(re.escape(s) for s in sections)
    pattern = re.compile(rf"""(\[|\.(get|set|select|exclude|pop)\(\(?)\s*(["'])({names})\3""")
    found = []
    for path in map(Path, paths):
        if path.resolve() == Path(__file__).resolve():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line) and "unfold(" not in line:
                found.append(f"{path}:{number}")
    return found
