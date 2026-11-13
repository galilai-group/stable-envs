"""Counter-based random numbers of the MovingBlockWorld reset, equal in Numba and in Torch.

Draw ``k`` of an environment's episode is a hash of ``(seed, episode, k)``: no generator state to carry, and
every environment draws its numbers independently on any device. Each named draw has a fixed slot ``k``.
"""

from __future__ import annotations

import torch
from numba import njit

_M32 = 0xFFFFFFFF
_C1, _C2 = 0x7FEB352D, 0x846CA68B

COUNT, FLOOR, WALL = 0, 1, 2
PERM = 16  # one key per colour of a permutation
COLOR = 32  # one draw per block colour picked with replacement
AGENT_DIR = 62
AGENT = 64  # placement candidates: two draws each
BLOCK = 128
BLOCK_STRIDE = 64
B_TEX, B_BALL, B_SIZE, B_DIR, B_VEL, B_GRID, B_THETA, B_PLACE = 0, 1, 2, 3, 4, 5, 30, 32


@njit(cache=True, inline="always")
def _mul(x, c):
    """``x * c mod 2**32`` for ``x, c < 2**32`` in int64 without overflow."""
    return (x * (c & 0xFFFF) + (((x * (c >> 16)) & 0xFFFF) << 16)) & _M32


@njit(cache=True, inline="always")
def mix(x):
    """The ``lowbias32`` integer hash."""
    x &= _M32
    x ^= x >> 16
    x = _mul(x, _C1)
    x ^= x >> 15
    x = _mul(x, _C2)
    return x ^ (x >> 16)


@njit(cache=True, inline="always")
def stream(seed, episode):
    s = mix((seed & _M32) ^ 0x3C6EF372)
    s = mix(s ^ ((seed >> 32) & _M32))
    return mix(s ^ (episode & _M32))


@njit(cache=True, inline="always")
def uniform(base, k):
    """Draw ``k`` of the stream ``base`` in [0, 1)."""
    return mix(base ^ mix(k)) * (1.0 / 4294967296.0)


def _mul_t(x: torch.Tensor, c: int) -> torch.Tensor:
    return (x * (c & 0xFFFF) + (((x * (c >> 16)) & 0xFFFF) << 16)) & _M32


def mix_t(x: torch.Tensor) -> torch.Tensor:
    x = x & _M32
    x = x ^ (x >> 16)
    x = _mul_t(x, _C1)
    x = x ^ (x >> 15)
    x = _mul_t(x, _C2)
    return x ^ (x >> 16)


def stream_t(seed: torch.Tensor, episode: torch.Tensor) -> torch.Tensor:
    s = mix_t((seed & _M32) ^ 0x3C6EF372)
    s = mix_t(s ^ ((seed >> 32) & _M32))
    return mix_t(s ^ (episode & _M32))


def uniform_t(base: torch.Tensor, k) -> torch.Tensor:
    """Draws ``k`` (an int or an int64 tensor that broadcasts with ``base``) as float64 in [0, 1)."""
    # A fill, not a host copy, so that a CUDA graph can capture the draw.
    k = torch.full((), k, dtype=torch.int64, device=base.device) if isinstance(k, int) else k
    return mix_t(base ^ mix_t(k)).to(torch.float64) * (1.0 / 4294967296.0)
