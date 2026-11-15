"""Classic-control State held in one host buffer per dtype, exported with one copy per dtype.

Each State leaf is a contiguous ``(num_envs, width)`` segment of the buffer of its dtype. The host simulators read
and write the segments in place, so a State export copies three buffers and makes each leaf a view of the copies.
A sequence of recent such States stacks with one copy per dtype.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.tensordicts import nested_builder, to_device
from .local import CLASSIC_FIELDS

_DTYPES = (torch.float64, torch.int64, torch.bool)
_NUMPY = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}
#: Exports that ``stack`` can find; older States of a longer sequence stack leaf by leaf.
_RECENT = 32


def _getter(keys) -> Callable[[TensorDict], tuple]:
    """``get(state)``: the leaves of a State in ``keys`` order, read straight from its nodes; KeyError if one is
    missing."""
    paths = [(key,) if isinstance(key, str) else tuple(key) for key in keys]
    reads = []
    for path in paths:
        expr = "S"
        for part in path:
            expr = f"{expr}._tensordict[{part!r}]"
        reads.append(expr)
    return eval("lambda S: (" + ", ".join(reads) + ",)")  # noqa: S307 - the source holds only State key literals


class PackedState:
    """The State fields of ``num_envs`` lanes of ``domain`` in packed host buffers."""

    def __init__(self, domain: str, num_envs: int):
        self.num_envs = n = int(num_envs)
        fields = CLASSIC_FIELDS[domain]
        self.keys = tuple(fields)
        totals = dict.fromkeys(_DTYPES, 0)
        #: ``(dtype index, start, width)`` of each leaf, in State order.
        self.places = []
        for width, dtype in fields.values():
            self.places.append((_DTYPES.index(dtype), totals[dtype], width))
            totals[dtype] += n * width
        self.buffers = [np.zeros(totals[dtype], _NUMPY[dtype]) for dtype in _DTYPES]
        #: ``{key: (num_envs, width) view}`` of the live buffers.
        self.views = {
            key: self.buffers[group][start : start + n * width].reshape(n, width)
            for key, (group, start, width) in zip(self.keys, self.places, strict=True)
        }
        self._build = nested_builder([(k,) if isinstance(k, str) else tuple(k) for k in self.keys])
        self._leaves = _getter(self.keys)
        self._batch = torch.Size([n])
        self._itemsizes = [buffer.itemsize for buffer in self.buffers]
        #: ``{address of the copy: copies}`` of the most recent exports, oldest first, so ``stack`` can find them.
        self._recent: dict[int, tuple[list[np.ndarray], list[int]]] = {}

    def export(self, device: torch.device) -> TensorDict:
        """A State that owns its memory: one copy of each buffer, with each leaf a view of the copies."""
        copies = [buffer.copy() for buffer in self.buffers]
        addresses = [copy.ctypes.data for copy in copies]
        self._recent[addresses[0]] = (copies, addresses)
        if len(self._recent) > _RECENT:
            del self._recent[next(iter(self._recent))]
        n = self.num_envs
        leaves = [
            torch.from_numpy(copies[group][start : start + n * width].reshape(n, width))
            for group, start, width in self.places
        ]
        batch = self._batch
        state = self._build(leaves, lambda node: TensorDict._new_unsafe(node, batch_size=batch, device="cpu"))
        return state if device.type == "cpu" else to_device(state, device)

    def _copies_of(self, state: TensorDict) -> list[np.ndarray] | None:
        """The copies that a recent ``export`` made ``state`` from, or None for another State."""
        try:
            leaves = self._leaves(state)
        except (KeyError, AttributeError):
            return None
        group0, start0, _ = self.places[0]
        found = self._recent.get(leaves[0].data_ptr() - start0 * self._itemsizes[group0])
        if found is None or leaves[0].device.type != "cpu":
            return None
        copies, addresses = found
        n, sizes = self.num_envs, self._itemsizes
        for leaf, (group, start, width) in zip(leaves, self.places, strict=True):
            expected = addresses[group] + start * sizes[group]
            if leaf.data_ptr() != expected or leaf.shape != (n, width) or not leaf.is_contiguous():
                return None
        return copies

    def stack(self, states) -> TensorDict | None:
        """The States of a step sequence stacked along a new time axis with one copy per dtype, else None."""
        per_state = [self._copies_of(state) for state in states]
        if any(copies is None for copies in per_state):
            return None
        steps, n = len(states), self.num_envs
        stacked = [np.stack([copies[group] for copies in per_state]) for group in range(len(_DTYPES))]
        # NumPy slices and from_numpy cost less per leaf than tensor views.
        leaves = [
            torch.from_numpy(stacked[group][:, start : start + n * width].reshape(steps, n, width).transpose(1, 0, 2))
            for group, start, width in self.places
        ]
        batch = torch.Size([n, steps])
        return self._build(leaves, lambda node: TensorDict._new_unsafe(node, batch_size=batch, device="cpu"))
