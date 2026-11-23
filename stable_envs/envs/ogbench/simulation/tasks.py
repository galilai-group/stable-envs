"""The ``task_id`` reset option of the OGBench classes."""

from __future__ import annotations

import numpy as np


def task_ids(value, batch: int | None, num_tasks: int) -> np.ndarray:
    """Check a ``task_id`` reset option and return one task per environment as an int64 array.

    Args:
        value: One task in ``1..num_tasks``, or a sequence with one task per environment.
        batch: The number of environments; None accepts only one task and returns it as a ``(1,)`` array.
        num_tasks: The number of evaluation tasks.

    Raises:
        ValueError: If a task is not a Python or NumPy integer (bools are refused) in ``1..num_tasks``, or a
            sequence does not have ``batch`` entries.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    sequence = isinstance(value, (list, tuple, np.ndarray))
    items = list(value) if sequence else [value]
    if (
        (sequence and (batch is None or len(items) != batch))
        or not items
        or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) or not 1 <= item <= num_tasks
            for item in items
        )
    ):
        many = "" if batch is None else f", or a sequence of {batch} such ints"
        raise ValueError(f"task_id must be an int in 1..{num_tasks}{many}; got {value!r}")
    return np.asarray(items * (batch or 1) if not sequence else items, np.int64)
