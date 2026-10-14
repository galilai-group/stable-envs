"""PCG64 generator state as fixed-width State columns."""

from __future__ import annotations

import numpy as np


def uint128_bytes(value: int) -> np.ndarray:
    """Return a 128-bit integer as 16 little-endian bytes."""
    return np.frombuffer(int(value).to_bytes(16, "little"), dtype=np.uint8).copy()


def bytes_uint128(value: np.ndarray) -> int:
    """Return the integer stored as 16 little-endian bytes."""
    return int.from_bytes(value.tobytes(), "little")


def pcg_state(state: int, inc: int, has_uint32: int, uinteger: int) -> dict:
    """Return a ``bit_generator.state`` dictionary for PCG64."""
    return {
        "bit_generator": "PCG64",
        "state": {"state": state, "inc": inc},
        "has_uint32": has_uint32,
        "uinteger": uinteger,
    }


def pcg_encode(rng):
    """Return ``rng``'s PCG64 state as byte and integer columns."""
    r = rng.bit_generator.state
    return {
        "state": uint128_bytes(r["state"]["state"]),
        "increment": uint128_bytes(r["state"]["inc"]),
        "has_uint32": np.array([r["has_uint32"]], np.int64),
        "uinteger": np.array([r["uinteger"]], np.int64),
    }


def pcg_decode(values):
    """Return the PCG64 state stored by :func:`pcg_encode`, rejecting an even increment or invalid cache."""
    state = bytes_uint128(values["state"])
    inc = bytes_uint128(values["increment"])
    flag, cached = int(values["has_uint32"][0]), int(values["uinteger"][0])
    if inc % 2 != 1 or flag not in (0, 1) or not 0 <= cached < 2**32:
        raise ValueError("Invalid PCG64 increment/cache")
    return pcg_state(state, inc, flag, cached)
