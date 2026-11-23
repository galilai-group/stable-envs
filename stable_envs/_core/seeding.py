"""Reset seeds: None, one int for the batch (row ``i`` gets ``seed + i``), or one int for each row."""

from __future__ import annotations

import numpy as np


def per_row(seed) -> bool:
    """Whether ``seed`` gives one seed to each row."""
    return seed is not None and not isinstance(seed, (int, np.integer))


def row_seed(seed, row: int) -> int | None:
    """The seed of batch row ``row``: ``seed + row`` for one int, ``seed[row]`` for a per-row seed."""
    if seed is None:
        return None
    if isinstance(seed, (int, np.integer)):
        return int(seed) + row
    return int(seed[row])


def rows_seed(seed, start: int, stop: int):
    """The seed of rows ``start:stop`` as a batch of their own: an int stays an int, a per-row seed is sliced."""
    if seed is None:
        return None
    if per_row(seed):
        return np.asarray(seed[start:stop], np.int64)
    return int(seed) + start


def row_seed_list(seed, count: int) -> list[int | None]:
    """The seed of each of ``count`` rows as Python ints, or ``count`` Nones for a None seed."""
    if seed is None:
        return [None] * count
    if isinstance(seed, (int, np.integer)):
        return list(range(int(seed), int(seed) + count))
    return np.asarray(seed, np.int64).tolist()


def row_seeds(seed, count: int) -> np.ndarray:
    """The ``(count,)`` int64 seeds of every row of a seed that is not None."""
    if per_row(seed):
        return np.asarray(seed, np.int64)
    return np.arange(count, dtype=np.int64) + int(seed)
