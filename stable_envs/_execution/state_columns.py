"""Assemble and validate State columns from per-lane simulator rows."""

import numpy as np
import torch
from tensordict import TensorDict


def owned_columns(rows, batch, device):
    """Collate nonempty NumPy rows into owned tensor columns on the requested device.

    Rows must agree in keys, shapes and dtypes. Copies isolate returned State
    from reused simulator buffers; each column transfers once.
    """
    rows = iter(rows)
    first = next(rows)
    columns = {key: np.empty((batch, *value.shape), dtype=value.dtype) for key, value in first.items()}
    for key, value in first.items():
        columns[key][0] = value
    count = 1
    for index, row in enumerate(rows, 1):
        if row.keys() != columns.keys():
            raise ValueError("State rows must have identical fields")
        for key, value in row.items():
            if value.shape != columns[key].shape[1:] or value.dtype != columns[key].dtype:
                raise ValueError(f"Inconsistent State row field {key}")
            columns[key][index] = value
        count += 1
    if count != batch:
        raise ValueError("State row count differs from batch")
    return TensorDict(
        {key: torch.from_numpy(value).to(device) for key, value in columns.items()}, [batch], device=device
    )


def require_columns(state, shapes):
    """Check that State holds every ``{key: (shape, dtype)}`` column and that all are finite.

    Finiteness is reduced over every column before a single host read.
    """
    present = {(key,) if isinstance(key, str) else key for key in state.keys(True, True)}
    values = []
    for key, (shape, dtype) in shapes.items():
        if ((key,) if isinstance(key, str) else key) not in present:
            raise ValueError(f"Full State missing {key}")
        value = state[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"Invalid State field {key}")
        values.append((key, value))
    if values and not bool(torch.stack([torch.isfinite(value).all() for _, value in values]).all()):
        raise ValueError(
            f"Invalid State field {next(key for key, value in values if not torch.isfinite(value).all())}"
        )
