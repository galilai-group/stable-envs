from __future__ import annotations

import torch
from tensordict import TensorDict

from ...._core.state_schema import get_field
from ...._core.tensordicts import stack_leaves
from .replay import HISTORY_KEYS


def stack_states(states):
    """Return owned ``[B, selected_time]`` states with a common history width.

    Pad only the replay-history axis. Each snapshot retains its own valid mask
    and history length; the selected-time axis is independent.
    """
    if not states:
        raise ValueError("PushT State collation requires at least one snapshot")
    widths = [get_field(state, "history/actions").shape[1] for state in states]
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
