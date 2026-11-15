"""Packed State export: one new buffer per dtype, and the State's TensorDict tree built from views of it.

A family declares the leaves of its State once, as ``(key, per-row shape, dtype)`` in State order, and builds a
``StateLayout``. ``export`` copies the current values into one fresh buffer per dtype (on an accelerator, through one
host-to-device copy of all of them) and returns a TensorDict whose leaves are views of those buffers, so the State
owns its memory. Leaves named ``static`` change only on reset, restore or a Variation change: ``load_static`` packs
them once, and each export copies them with one copy per dtype. ``columns`` reads a State back for a restore.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from tensordict import TensorDict

#: A leaf key: a tuple of names, or names joined by ``/``. ``export`` reads values under the key as declared.
Key = tuple[str, ...] | str


def _path(key: Key) -> tuple[str, ...]:
    return tuple(key.split("/")) if isinstance(key, str) else tuple(key)


#: Byte alignment of each dtype's buffer inside the pinned transfer buffer; a multiple of every dtype's size.
_ALIGN = 16


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    return torch.empty((), dtype=dtype).numpy().dtype


def _builder(keys: Sequence[Key]):
    """``build(leaves, node)``: the nested State from leaves in ``keys`` order; ``node(dict)`` makes one TensorDict."""
    tree: dict = {}
    for index, key in enumerate(map(_path, keys)):
        node = tree
        for part in key[:-1]:
            node = node.setdefault(part, {})
        if not isinstance(node, dict) or key[-1] in node:
            raise ValueError(f"State key {key} clashes with another key")
        node[key[-1]] = index

    def source(node):
        items = (f"{name!r}: {source(c) if type(c) is dict else f'L[{c}]'}" for name, c in node.items())
        return "N({" + ", ".join(items) + "})"

    return eval("lambda L, N: " + source(tree))  # noqa: S307 - the source holds only State key literals


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
        groups: dict[torch.dtype, tuple[list[int], list[int]]] = {}
        for index, (key, _, dtype) in enumerate(leaves):
            groups.setdefault(dtype, ([], []))[key not in static].append(index)
        #: ``(dtype, NumPy dtype, static leaf indices, moving leaf indices)`` per dtype; a buffer holds the static
        #: leaves first.
        self.groups = [(dtype, _numpy_dtype(dtype), fixed, moving) for dtype, (fixed, moving) in groups.items()]
        self._static: dict[torch.dtype, torch.Tensor | np.ndarray] = {}
        self._build = _builder(self.keys)
        self._batch = torch.Size([self.num_envs])

    def load_static(self, values: Mapping[Key, torch.Tensor | np.ndarray]) -> None:
        """Pack the static leaves from ``values``; call it after a reset, restore or Variation change."""
        self._static = {
            dtype: self._pack(kind, [values[self.keys[i]] for i in fixed])
            for dtype, kind, fixed, _ in self.groups
            if fixed
        }

    @staticmethod
    def _pack(kind: np.dtype, sources: Sequence) -> torch.Tensor | np.ndarray:
        """One new flat buffer: a NumPy array when every source is one, else a tensor."""
        if all(isinstance(v, np.ndarray) for v in sources):
            return np.concatenate([v.reshape(-1) for v in sources], dtype=kind)
        return torch.cat([torch.as_tensor(v).reshape(-1) for v in sources])

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
        flats, shapes = [], [None] * len(self.keys)
        for dtype, kind, fixed, moving in self.groups:
            sources = [values[self.keys[i]] for i in moving]
            for i, value in zip(moving, sources, strict=True):
                shapes[i] = self.shapes[self.keys[i]] or tuple(value.shape)
            for i in fixed:
                shapes[i] = self.shapes[self.keys[i]]
            if fixed:
                if dtype not in self._static:
                    raise RuntimeError("StateLayout.load_static must run before export")
                sources.insert(0, self._static[dtype])
            flats.append(self._pack(kind, sources))
        host = device.type == "cpu"
        if not host:
            flats = [torch.from_numpy(f) if isinstance(f, np.ndarray) else f for f in flats]
            if any(flat.device.type == "cpu" for flat in flats):
                flats = self._to_device(flats, device)
        leaves: list = [None] * len(self.keys)
        for (_, _, fixed, moving), flat in zip(self.groups, flats, strict=True):
            start = 0
            for i in fixed + moving:
                stop = start + int(np.prod(shapes[i]))
                # NumPy slices and from_numpy cost less per leaf than tensor views.
                part = flat[start:stop].reshape(shapes[i])
                leaves[i] = torch.from_numpy(part) if isinstance(part, np.ndarray) else part
                start = stop
        batch, where = self._batch, _recorded(flats, device)
        return self._build(leaves, lambda node: TensorDict._new_unsafe(node, batch_size=batch, device=where))

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


def _recorded(flats: Sequence, device: torch.device) -> torch.device:
    """The device the TensorDict records: ``device`` with its index."""
    if device.type == "cuda" and device.index is None:
        cuda = [f.device for f in flats if torch.is_tensor(f)]
        return cuda[0] if cuda else torch.device("cuda", torch.cuda.current_device())
    return device
