from __future__ import annotations

import torch
from tensordict import TensorDict

from ...._core.state_schema import get_field, schema_of
from ...._core.tensordicts import build_tensordict, stack_leaves
from .replay import HISTORY_KEYS


def _with_time_axis(state):
    """A copy of ``state`` with a time axis of length 1, with one copy per storage that its leaves share.

    An exported State's leaves on an accelerator are views of a few buffers, so this copies a few buffers instead
    of every leaf. A State on the CPU, or whose leaves don't share storage or view a larger buffer, is copied leaf
    by leaf.
    """
    leaves = list(state.items(True, True))
    if state.device.type == "cpu":
        # On the host, a NumPy copy per leaf costs less than a torch.stack.
        return _tree(((key, torch.from_numpy(leaf.numpy()[:, None].copy())) for key, leaf in leaves), state)
    used: dict = {}
    for _, leaf in leaves:
        storage = leaf.untyped_storage()
        source = (storage.data_ptr(), storage.nbytes())
        used[source] = used.get(source, 0) + leaf.numel() * leaf.element_size()
    if len(used) == len(leaves) or any(nbytes > 2 * size for (_, nbytes), size in used.items()):
        return stack_leaves([state])
    copies: dict = {}

    def copy(leaf):
        storage = leaf.untyped_storage()
        source = (storage.data_ptr(), storage.nbytes())
        if source not in copies:
            copies[source] = storage.clone()
        return leaf.new_empty(0).set_(copies[source], leaf.storage_offset(), leaf.shape, leaf.stride()).unsqueeze(1)

    return _tree(((key, copy(leaf)) for key, leaf in leaves), state)


def _tree(items, state):
    """A TensorDict with a time axis of length 1 from ``(key, value)`` pairs of ``state``'s leaves."""
    tree: dict = {}
    for key, value in items:
        *parents, name = (key,) if isinstance(key, str) else key
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
        node[name] = value
    return build_tensordict(tree, [*state.batch_size, 1], state.device)


def stack_states(states):
    """Return owned ``[B, selected_time]`` states with a common history width.

    Pad only the replay-history axis. Each snapshot retains its own valid mask
    and history length; the selected-time axis is independent.
    """
    if not states:
        raise ValueError("PushT State collation requires at least one snapshot")
    if len(states) == 1:
        return _with_time_axis(states[0])
    schema = schema_of(states[0])
    read = get_field if schema is None else schema.get
    widths = [read(state, "history/actions").shape[1] for state in states]
    capacity = max(widths)
    if min(widths) == capacity:
        return stack_leaves(states)
    padded = []
    for state in states:
        values = {}
        for key, value in state.items(True, True):
            if "/".join(key) in HISTORY_KEYS:
                padding = value.new_zeros((value.shape[0], capacity - value.shape[1], *value.shape[2:]))
                value = torch.cat((value, padding), dim=1)
            values[key] = value
        padded.append(TensorDict(values, state.batch_size, device=state.device))
    return stack_leaves(padded)
