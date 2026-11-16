"""Classic-control State held in one host array per group, exported with one copy per group.

The live State is the group arrays of its schema, and the host simulators read and write each field as a view of
its group. A State export copies the group arrays and makes each leaf of the State one of the copies. A sequence of
recent such States stacks with one copy per group.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.tensordicts import nested_builder, to_device
from .local import STATES

#: Exports that ``stack`` can find; older States of a longer sequence stack leaf by leaf.
_RECENT = 32


def _getter(paths) -> Callable[[TensorDict], tuple]:
    """``get(state)``: the leaves of a State at ``paths``, read straight from its nodes.

    ``get`` raises KeyError if one is missing.
    """
    reads = []
    for path in paths:
        expr = "S"
        for part in path:
            expr = f"{expr}._tensordict[{part!r}]"
        reads.append(expr)
    return eval("lambda S: (" + ", ".join(reads) + ",)")  # noqa: S307 - the source holds only State key literals


class PackedState:
    """The State fields of ``num_envs`` lanes of ``domain`` in host group arrays."""

    def __init__(self, domain: str, num_envs: int):
        self.num_envs = n = int(num_envs)
        schema = STATES[domain]
        live = schema.new(n, "cpu")
        #: ``{group key: (num_envs, size) array}``: the live State, which the host simulators write.
        self.arrays = {key: tensor.numpy() for key, tensor in schema.buffers(live).items()}
        #: ``{field name: (num_envs, width) view}`` of the live arrays, cut once.
        self.views = schema.views(self.arrays)
        paths = [tuple(key.split("/")) for key in self.arrays]
        self._build = nested_builder(paths)
        self._leaves = _getter(paths)
        self._batch = torch.Size([n])
        self._shapes = [array.shape for array in self.arrays.values()]
        #: ``{address of the copy: copies}`` of the most recent exports, oldest first, so ``stack`` can find them.
        self._recent: dict[int, tuple[list[np.ndarray], list[int]]] = {}

    def export(self, device: torch.device) -> TensorDict:
        """A State that owns its memory: one copy of each group, which is its leaf."""
        copies = [array.copy() for array in self.arrays.values()]
        addresses = [copy.ctypes.data for copy in copies]
        self._recent[addresses[0]] = (copies, addresses)
        if len(self._recent) > _RECENT:
            del self._recent[next(iter(self._recent))]
        batch = self._batch
        state = self._build(
            [torch.from_numpy(copy) for copy in copies],
            lambda node: TensorDict._new_unsafe(node, batch_size=batch, device="cpu"),
        )
        return state if device.type == "cpu" else to_device(state, device)

    def _copies_of(self, state: TensorDict) -> list[np.ndarray] | None:
        """The copies that a recent ``export`` made ``state`` from, or None for another State."""
        try:
            leaves = self._leaves(state)
        except (KeyError, AttributeError):
            return None
        found = self._recent.get(leaves[0].data_ptr())
        if found is None or leaves[0].device.type != "cpu":
            return None
        copies, addresses = found
        for leaf, address, shape in zip(leaves, addresses, self._shapes, strict=True):
            if leaf.data_ptr() != address or leaf.shape != shape or not leaf.is_contiguous():
                return None
        return copies

    def stack(self, states) -> TensorDict | None:
        """The States of a step sequence stacked along a new time axis with one copy per group, else None."""
        per_state = [self._copies_of(state) for state in states]
        if any(copies is None for copies in per_state):
            return None
        steps = len(states)
        # NumPy stacks and from_numpy cost less per leaf than tensor operations.
        leaves = [
            torch.from_numpy(np.stack([copies[group] for copies in per_state]).transpose(1, 0, 2))
            for group in range(len(self._shapes))
        ]
        batch = torch.Size([self.num_envs, steps])
        return self._build(leaves, lambda node: TensorDict._new_unsafe(node, batch_size=batch, device="cpu"))
