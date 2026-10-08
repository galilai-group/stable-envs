"""Craftax night-noise generation using JAX-compatible Threefry in Torch.

Matches JAX's partitionable threefry2x32 path with float32 output. The
non-partitionable path produces different values and is not supported.
Keys and counters use signed int32 tensors to hold uint32 bit patterns.
"""

from __future__ import annotations

import torch

_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
# Threefry key-schedule parity constant.
_PARITY = 0x1BD11BDA
# Float32 exponent for 1.0 and its mantissa mask.
_ONE_BITS = 0x3F800000
_MANTISSA = 0x7FFFFF


def _rotl32(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rotate a uint32 bit pattern stored in int32 left by ``d`` bits."""
    # Mask off sign extension from Torch's arithmetic right shift.
    return (x << d) | ((x >> (32 - d)) & ((1 << d) - 1))


def threefry2x32(
    k1: torch.Tensor, k2: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two output words of the 20-round Threefry-2x32 hash.

    Key words ``k1, k2`` and counter words ``x1, x2`` must be broadcastable
    int32 tensors. Their bit patterns are interpreted as unsigned words.
    """
    ks = [k1, k2, k1 ^ k2 ^ _PARITY]
    x = [x1 + ks[0], x2 + ks[1]]
    ks = ks[1:] + ks[:1]
    rots = [_ROTATIONS[0], _ROTATIONS[1]]
    for i in range(5):
        for r in rots[0]:
            x[0] = x[0] + x[1]
            x[1] = _rotl32(x[1], r)
            x[1] = x[0] ^ x[1]
        x = [x[0] + ks[0], x[1] + ks[1] + (i + 1)]
        ks = ks[1:] + ks[:1]
        rots = rots[1:] + rots[:1]
    return x[0], x[1]


def _counter(
    n: int, offset: int, device: torch.device, dtype: torch.dtype = torch.int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split counters ``offset`` through ``offset + n - 1`` into high and low words."""
    i = torch.arange(int(offset), int(offset) + int(n), dtype=torch.int64, device=device)
    # Narrowing preserves the low 32 bits, including the sign bit.
    return (i >> 32).to(dtype), (i & 0xFFFFFFFF).to(dtype)


def random_bits(keys: torch.Tensor, n: int, offset: int = 0) -> torch.Tensor:
    """Return ``(B, n)`` random uint32 bit patterns in an int32 tensor.

    ``keys`` has shape ``(B, 2)``. ``offset`` selects the first counter so
    successive chunks reproduce a single draw of the whole field.
    """
    if keys.ndim != 2 or keys.shape[1] != 2:
        raise ValueError(f"prng.random_bits: keys must be (B, 2), got {tuple(keys.shape)}")
    k = keys.to(torch.int32)
    B = k.shape[0]
    hi, lo = _counter(n, offset, k.device)
    b1, b2 = threefry2x32(
        k[:, 0].view(B, 1), k[:, 1].view(B, 1), hi.view(1, n).expand(B, n), lo.view(1, n).expand(B, n)
    )
    return b1 ^ b2


def uniform(keys: torch.Tensor, shape: tuple[int, ...], offset: int = 0) -> torch.Tensor:
    """Return float32 values in ``[0, 1)`` with shape ``(B, *shape)``.

    ``keys`` has shape ``(B, 2)``; ``offset`` counts flattened output elements.
    Apply Craftax's ``* 95 + 32`` scaling at the call site. Fusing that
    multiply and add can change rounding and break pixel agreement.
    """
    n = 1
    for d in shape:
        n *= int(d)
    bits = random_bits(keys, n, offset)
    floats = (((bits >> 9) & _MANTISSA) | _ONE_BITS).view(torch.float32) - 1.0
    return floats.reshape(keys.shape[0], *[int(d) for d in shape])


def key_from_halves(halves: torch.Tensor) -> torch.Tensor:
    """Rebuild ``(B, 2)`` int32 keys from ``(B, 4)`` 16-bit halves.

    Halves are ordered ``(hi0, lo0, hi1, lo1)`` and fit exactly in float32
    render projections; complete uint32 words do not.
    """
    h = halves.to(torch.int64) & 0xFFFF
    w0 = (h[:, 0] << 16) | h[:, 1]
    w1 = (h[:, 2] << 16) | h[:, 3]
    return torch.stack([w0, w1], dim=1).to(torch.int32)


def key_to_halves(keys: torch.Tensor) -> torch.Tensor:
    """Split ``(B, 2)`` keys into ``(B, 4)`` int64 halves ordered ``(hi0, lo0, hi1, lo1)``."""
    k = keys.to(torch.int64) & 0xFFFFFFFF
    return torch.stack([(k[:, 0] >> 16) & 0xFFFF, k[:, 0] & 0xFFFF, (k[:, 1] >> 16) & 0xFFFF, k[:, 1] & 0xFFFF], dim=1)
