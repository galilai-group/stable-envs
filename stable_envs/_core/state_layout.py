"""Packed State export: one new buffer per dtype, and the State's TensorDict tree built from views of it.

A family declares the leaves of its State once, as ``(key, per-row shape, dtype)`` in State order, and builds a
``StateLayout``. ``export`` copies the current values into one fresh buffer per dtype (on an accelerator, through one
host-to-device copy of all of them) and returns a TensorDict whose leaves are views of those buffers, so the State
owns its memory. Leaves named ``static`` change only on reset, restore or a Variation change: ``load_static`` packs
them once, and each export copies them with one copy per dtype. ``columns`` reads a State back for a restore.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from tensordict import TensorDict

from .tensordicts import nested_builder, new_node, to_device

#: A leaf key: a tuple of names, or names joined by ``/``. ``export`` reads values under the key as declared.
Key = tuple[str, ...] | str


def _path(key: Key) -> tuple[str, ...]:
    return tuple(key.split("/")) if isinstance(key, str) else tuple(key)


#: Size from which a new host buffer comes from fresh pages: glibc's default threshold for serving an allocation
#: with ``mmap``. A buffer this large is reused once no earlier State holds it. Smaller States don't pay page faults,
#: and reuse costs them more than it saves: the reused buffer needs a ``reshape`` per leaf and tensor views instead
#: of NumPy slices (measured on a PushT State of 239 leaves and a DMC Reacher State of 14 leaves).
_FRESH_PAGES = 128 * 1024
_CPU = torch.device("cpu")
#: Byte alignment of each dtype's buffer inside the pinned transfer buffer; a multiple of every dtype's size.
_ALIGN = 16


@functools.cache
def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    return torch.empty((), dtype=dtype).numpy().dtype


class StateLayout:
    """The leaves of a State with ``num_envs`` rows, grouped by dtype into packed buffers.

    Args:
        leaves: ``(key, per-row shape, dtype)`` of every leaf, in State order. A shape of None takes the value's
            shape at each export, for a leaf whose width changes, such as an action history.
        num_envs: Number of rows.
        static: Keys, as declared in ``leaves``, whose values ``load_static`` packs once instead of every export.
    """

    def __init__(
        self,
        leaves: Sequence[tuple[Key, tuple[int, ...] | None, torch.dtype]],
        num_envs: int,
        static: frozenset[Key] = frozenset(),
    ):
        self.num_envs = int(num_envs)
        self.keys: tuple[Key, ...] = tuple(key for key, _, _ in leaves)
        self.shapes = {key: None if shape is None else (self.num_envs, *shape) for key, shape, _ in leaves}
        self.dtypes = {key: dtype for key, _, dtype in leaves}
        self._kinds = {key: _numpy_dtype(dtype) for key, dtype in self.dtypes.items()}
        self._host_plans: dict[tuple[str, ...], list] = {}
        groups: dict[torch.dtype, tuple[list[int], list[int]]] = {}
        for index, (key, _, dtype) in enumerate(leaves):
            groups.setdefault(dtype, ([], []))[key not in static].append(index)
        #: ``(dtype, NumPy dtype, static leaf indices, moving leaf indices, runs)`` per dtype. A buffer holds the
        #: static leaves, then the moving ones, each grouped by shape. A run ``(indices, row shape)`` is leaves of
        #: one shape next to each other, which one ``unbind`` splits; a leaf without a shape is a run of its own.
        self.groups = []
        for dtype, (fixed, moving) in groups.items():
            fixed, moving = self._by_shape(fixed), self._by_shape(moving)
            self.groups.append((dtype, _numpy_dtype(dtype), fixed, moving, self._runs(fixed + moving)))
        #: Per group: the keys of its moving leaves, and the run sizes when no leaf of the group lacks a shape.
        self._moving_keys = [[self.keys[i] for i in moving] for _, _, _, moving, _ in self.groups]
        self._sizes = [
            None if any(shape is None for _, shape in runs) else [len(i) * math.prod(shape) for i, shape in runs]
            for *_, runs in self.groups
        ]
        self._static: dict[torch.dtype, torch.Tensor | np.ndarray] = {}
        self._buffers: dict[torch.dtype, list[torch.Tensor]] = {}
        self._build = nested_builder([_path(key) for key in self.keys])
        self._batch = torch.Size([self.num_envs])

    def _by_shape(self, indices: list[int]) -> list[int]:
        order = {}
        for i in indices:
            order.setdefault(self.shapes[self.keys[i]], []).append(i)
        return [i for run in order.values() for i in run]

    def _runs(self, indices: list[int]) -> list[tuple[list[int], tuple[int, ...] | None]]:
        runs: list[tuple[list[int], tuple[int, ...] | None]] = []
        for i in indices:
            shape = self.shapes[self.keys[i]]
            if shape is not None and runs and runs[-1][1] == shape:
                runs[-1][0].append(i)
            else:
                runs.append(([i], shape))
        return runs

    def load_static(self, values: Mapping[Key, torch.Tensor | np.ndarray]) -> None:
        """Pack the static leaves from ``values``; call it after a reset, restore or Variation change."""
        self._static = {
            dtype: self._pack(kind, [values[self.keys[i]] for i in fixed])
            for dtype, kind, fixed, _, _ in self.groups
            if fixed
        }

    @staticmethod
    def _pack(kind: np.dtype, sources: Sequence) -> torch.Tensor | np.ndarray:
        """One new flat buffer: a NumPy array when every source is one, else a tensor."""
        if all(isinstance(v, np.ndarray) for v in sources):
            return np.concatenate([v.reshape(-1) for v in sources], dtype=kind)
        tensors = [torch.as_tensor(v) for v in sources]
        if len(tensors) == 1:
            # Flattening one contiguous tensor returns a view of it, not a copy.
            return tensors[0].clone(memory_format=torch.contiguous_format).view(-1)
        # One call flattens and concatenates the tensors.
        return torch._C._nn.flatten_dense_tensors(tensors)

    @classmethod
    def of(cls, state: TensorDict) -> StateLayout:
        """The layout of the leaves of ``state``, in its order."""
        keys = list(state.keys(True, True))
        return cls(
            [((k,) if isinstance(k, str) else tuple(k), tuple(state[k].shape[1:]), state[k].dtype) for k in keys],
            state.batch_size[0],
        )

    def export(self, values: Mapping[Key, torch.Tensor | np.ndarray], device) -> TensorDict:
        """A State that owns its memory, from the current ``{key: (num_envs, ...) value}``.

        Host values are packed with one copy per dtype; on an accelerator the packed buffers move in one copy.
        Values already on ``device`` are packed there.
        """
        device = torch.device(device)
        fixed = all(sizes is not None for sizes in self._sizes)
        if device.type == "cuda" and fixed and _on_host(values.values()) and _on_host(self._static.values()):
            # Packed on the host, then one copy whose split into leaves is one call per dtype; a leaf whose width
            # changes would need a new transfer plan for each width.
            return to_device(self.export(values, _CPU), device)
        flats, layout = [], []
        for (dtype, kind, fixed, _, runs), keys, sizes in zip(
            self.groups, self._moving_keys, self._sizes, strict=True
        ):
            sources = [values[key] for key in keys]
            if fixed:
                if dtype not in self._static:
                    raise RuntimeError("StateLayout.load_static must run before export")
                sources.insert(0, self._static[dtype])
            widths = None
            if sizes is None:
                widths = {i[0]: tuple(values[self.keys[i[0]]].shape) for i, shape in runs if shape is None}
                sizes = [len(i) * math.prod(shape) if shape else math.prod(widths[i[0]]) for i, shape in runs]
            layout.append((runs, sizes, widths))
            numel = sum(sizes)
            if device.type == "cpu" and numel * dtype.itemsize >= _FRESH_PAGES:
                buffer = self._buffer(dtype, numel)
                if all(isinstance(v, np.ndarray) for v in sources):
                    np.concatenate([v.reshape(-1) for v in sources], out=buffer.numpy())
                else:
                    torch.cat([torch.as_tensor(v).reshape(-1) for v in sources], out=buffer)
                flats.append(buffer)
            else:
                flats.append(self._pack(kind, sources))
        if device.type != "cpu":
            flats = [torch.from_numpy(flat) if isinstance(flat, np.ndarray) else flat for flat in flats]
            if any(flat.device.type == "cpu" for flat in flats):
                flats = self._to_device(flats, device)
        leaves: list = [None] * len(self.keys)
        for (runs, sizes, widths), flat in zip(layout, flats, strict=True):
            if isinstance(flat, np.ndarray):
                # On the host, NumPy slices and from_numpy cost less per leaf than tensor views.
                start = 0
                for (indices, shape), size in zip(runs, sizes, strict=True):
                    block = flat[start : start + size]
                    start += size
                    if shape is None or len(indices) == 1:
                        leaves[indices[0]] = torch.from_numpy(block.reshape(shape or widths[indices[0]]))
                    else:
                        rows = torch.from_numpy(block.reshape(len(indices), *shape)).unbind(0)
                        for i, leaf in zip(indices, rows, strict=True):
                            leaves[i] = leaf
                continue
            for (indices, shape), block in zip(runs, flat.split_with_sizes(sizes), strict=True):
                if shape is None:
                    leaves[indices[0]] = block.view(widths[indices[0]])
                elif len(indices) == 1:
                    leaves[indices[0]] = block.view(shape)
                else:
                    for i, leaf in zip(indices, block.view(len(indices), *shape).unbind(0), strict=True):
                        leaves[i] = leaf
        batch, where = self._batch, _recorded(flats, device)
        return self._build(leaves, lambda node: new_node(node, batch, where))

    def host_leaves(self, values: Mapping[Key, torch.Tensor | np.ndarray], prefix: tuple[str, ...] = ()) -> dict:
        """``{prefix + key path: array}`` of the current host ``values`` with the declared shapes and dtypes.

        The arrays may share memory with ``values``; ``pack_to_device`` copies them at once. A layout with static
        leaves has no such form.
        """
        if any(fixed for _, _, fixed, _, _ in self.groups):
            raise RuntimeError("StateLayout.host_leaves needs a layout without static leaves")
        plan = self._host_plans.get(prefix)
        if plan is None:
            plan = self._host_plans[prefix] = [
                (key, (*prefix, *_path(key)), self._kinds[key], self.shapes[key]) for key in self.keys
            ]
        out = {}
        for key, path, kind, shape in plan:
            value = values[key]
            if type(value) is not np.ndarray:
                value = value.numpy() if torch.is_tensor(value) else np.asarray(value)
            if value.dtype != kind:
                value = value.astype(kind)
            if shape is not None and value.shape != shape:
                value = value.reshape(shape)
            out[path] = value
        return out

    def _buffer(self, dtype: torch.dtype, numel: int) -> torch.Tensor:
        """A flat host buffer that no earlier State still uses: a large new buffer costs page faults on each export."""
        if not _REUSE:
            return torch.empty(numel, dtype=dtype)
        pool = self._buffers.setdefault(dtype, [])
        for buffer in pool:
            # A buffer moved to shared memory may be read by another process, which the use count doesn't show.
            if buffer.numel() >= numel and _free(buffer) and not buffer.untyped_storage().is_shared():
                return buffer[:numel]
        buffer = torch.empty(numel + numel // 4, dtype=dtype)
        pool.append(buffer)
        if len(pool) > 2:
            pool.pop(0)
        return buffer[:numel]

    def _to_device(self, flats: list[torch.Tensor], device: torch.device) -> list[torch.Tensor]:
        """Move host buffers to ``device`` through one pinned buffer and one asynchronous copy."""
        offsets, total = [], 0
        for flat in flats:
            offsets.append(total)
            total += -(-flat.numel() * flat.element_size() // _ALIGN) * _ALIGN
        host = torch.empty(max(total, _ALIGN), dtype=torch.uint8, pin_memory=True)
        for flat, start in zip(flats, offsets, strict=True):
            n = flat.numel() * flat.element_size()
            host[start : start + n].view(flat.dtype).copy_(flat.cpu())
        # The pinned block isn't reused before this copy ends: the host allocator records the copy's stream.
        moved = host.to(device, non_blocking=True)
        return [
            moved[start : start + flat.numel() * flat.element_size()].view(flat.dtype)
            for flat, start in zip(flats, offsets, strict=True)
        ]

    def columns(self, state: TensorDict) -> dict[Key, torch.Tensor]:
        """``{key: leaf}`` of ``state``; raises if a leaf is missing or has another shape or dtype."""
        out = {}
        for key in self.keys:
            name = "/".join(_path(key))
            try:
                value = state[_path(key)]
            except KeyError:
                raise KeyError(f"State has no leaf {name}") from None
            shape = self.shapes[key]
            wrong = (shape is not None and tuple(value.shape) != shape) or value.shape[:1] != self._batch
            if wrong or value.dtype != self.dtypes[key]:
                raise ValueError(
                    f"State leaf {name} must be {self.dtypes[key]} {shape or (self.num_envs, '...')}, "
                    f"got {value.dtype} {tuple(value.shape)}"
                )
            out[key] = value
        return out


def _on_host(values) -> bool:
    return not any(torch.is_tensor(v) and v.device.type != "cpu" for v in values)


def _free(buffer: torch.Tensor) -> bool:
    """Whether no tensor but ``buffer`` uses its storage: the pool's tensor and the temporary storage object."""
    return torch._C._storage_Use_Count(buffer.untyped_storage()._cdata) == 2


def _reuse_works() -> bool:
    """Whether this torch counts the users of a storage as ``_free`` expects: a held view counts, a dropped one not."""
    if not hasattr(torch._C, "_storage_Use_Count"):
        return False
    try:
        buffer = torch.empty(8)
        view = buffer[:4]
        held = not _free(buffer)
        del view
        return held and _free(buffer)
    except Exception:  # noqa: BLE001 - any difference turns reuse off
        return False


#: Whether exports reuse free host buffers.
_REUSE = _reuse_works()


def _recorded(flats: Sequence, device: torch.device) -> torch.device:
    """The device the TensorDict records: ``device`` with its index."""
    if device.type == "cuda" and device.index is None:
        cuda = [f.device for f in flats if torch.is_tensor(f)]
        return cuda[0] if cuda else torch.device("cuda", torch.cuda.current_device())
    return device
