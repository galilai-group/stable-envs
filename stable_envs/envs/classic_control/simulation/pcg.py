"""NumPy's ``SeedSequence`` seeding of ``PCG64`` and its ``uniform`` draws, for many generators at once.

``reset(seed=s)`` of a Gymnasium environment seeds ``Generator(PCG64(SeedSequence(s)))``, and its initial state
comes from ``uniform`` draws. These functions repeat both on arrays of generators, one per lane, and give the same
128-bit states and the same doubles. The 128-bit values are ``(high, low)`` pairs of uint64 arrays.
"""

from __future__ import annotations

import numpy as np

_U32 = np.uint64(0xFFFF_FFFF)
_INIT_A, _MULT_A = 0x43B0D7E5, 0x931E8875
_INIT_B, _MULT_B = 0x8B51F9DD, 0x58F38DED
_MIX_L, _MIX_R = np.uint64(0xCA01F9DD), np.uint64(0x4973F715)
_POOL = 4
#: ``PCG_DEFAULT_MULTIPLIER_128`` as (high, low).
_MULT = (np.uint64(2549297995355413924), np.uint64(4865540595714422341))


def _constants(start: int, mult: int, count: int) -> list[np.uint64]:
    """The hash constant before each of ``count`` hash calls: it depends only on the call's index."""
    out, value = [], start
    for _ in range(count):
        out.append(np.uint64(value))
        value = (value * mult) & 0xFFFF_FFFF
    return out


def _hashmix(value: np.ndarray, const: np.uint64, mult: int) -> np.ndarray:
    value = value ^ const
    value = (value * np.uint64((int(const) * mult) & 0xFFFF_FFFF)) & _U32
    return value ^ (value >> np.uint64(16))


def _mix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = (_MIX_L * x - _MIX_R * y) & _U32
    return result ^ (result >> np.uint64(16))


def _words(seeds: np.ndarray, count: int) -> list[np.ndarray]:
    return [(seeds >> np.uint64(32 * i)) & _U32 for i in range(count)]


def _pool(entropy: list[np.ndarray]) -> list[np.ndarray]:
    """``SeedSequence.mix_entropy`` of ``entropy`` (uint32 words, least significant first) into the pool."""
    calls = _POOL + _POOL * (_POOL - 1) + max(0, len(entropy) - _POOL) * _POOL
    consts = iter(_constants(_INIT_A, _MULT_A, calls))
    zero = np.zeros_like(entropy[0])
    pool = [_hashmix(entropy[i] if i < len(entropy) else zero, next(consts), _MULT_A) for i in range(_POOL)]
    for src in range(_POOL):
        for dst in range(_POOL):
            if src != dst:
                pool[dst] = _mix(pool[dst], _hashmix(pool[src], next(consts), _MULT_A))
    for src in range(_POOL, len(entropy)):
        for dst in range(_POOL):
            pool[dst] = _mix(pool[dst], _hashmix(entropy[src], next(consts), _MULT_A))
    return pool


def _mul64(a: np.ndarray, b: np.uint64) -> tuple[np.ndarray, np.ndarray]:
    """The full 128-bit product of uint64 values ``a`` and ``b``, as (high, low)."""
    a_hi, a_lo = a >> np.uint64(32), a & _U32
    b_hi, b_lo = b >> np.uint64(32), b & _U32
    lo_lo, hi_lo, lo_hi, hi_hi = a_lo * b_lo, a_hi * b_lo, a_lo * b_hi, a_hi * b_hi
    cross = (lo_lo >> np.uint64(32)) + (hi_lo & _U32) + lo_hi
    high = hi_hi + (hi_lo >> np.uint64(32)) + (cross >> np.uint64(32))
    return high, (cross << np.uint64(32)) | (lo_lo & _U32)


def _step(state: tuple, inc: tuple) -> tuple[np.ndarray, np.ndarray]:
    """``state * PCG_DEFAULT_MULTIPLIER_128 + inc`` modulo 2**128."""
    s_hi, s_lo = state
    high, low = _mul64(s_lo, _MULT[1])
    high = high + s_hi * _MULT[1] + s_lo * _MULT[0]
    low_sum = low + inc[1]
    carry = (low_sum < low).astype(np.uint64)
    return high + inc[0] + carry, low_sum


def seed(seeds: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """``(state, inc)`` of ``PCG64(SeedSequence(s))`` for each non-negative integer seed ``s`` below 2**64."""
    seeds = np.asarray(seeds, dtype=np.uint64)
    state_hi, state_lo, inc_hi, inc_lo = (np.zeros_like(seeds) for _ in range(4))
    # SeedSequence takes the seed's 32-bit words, at least one; lanes with as many words hash together.
    lengths = np.where((seeds >> np.uint64(32)) == 0, 1, 2)
    for length in np.unique(lengths):
        rows = lengths == length
        pool = _pool(_words(seeds[rows], int(length)))
        consts = _constants(_INIT_B, _MULT_B, 8)
        words = [_hashmix(pool[i % _POOL], consts[i], _MULT_B) for i in range(8)]
        # ``generate_state(4, np.uint64)`` pairs the words, low word first.
        val = [words[2 * i] | (words[2 * i + 1] << np.uint64(32)) for i in range(4)]
        init, seq = (val[0], val[1]), (val[2], val[3])
        inc = ((seq[0] << np.uint64(1)) | (seq[1] >> np.uint64(63)), (seq[1] << np.uint64(1)) | np.uint64(1))
        zero = np.zeros_like(val[0])
        hi, lo = _step((zero, zero), inc)
        lo_sum = lo + init[1]
        hi = hi + init[0] + (lo_sum < lo).astype(np.uint64)
        hi, lo = _step((hi, lo_sum), inc)
        state_hi[rows], state_lo[rows], inc_hi[rows], inc_lo[rows] = hi, lo, inc[0], inc[1]
    return (state_hi, state_lo), (inc_hi, inc_lo)


def doubles(state: tuple, inc: tuple, count: int) -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """``count`` ``random()`` doubles per generator, in draw order, and the advanced states."""
    out = []
    for _ in range(count):
        state = _step(state, inc)
        hi, lo = state
        x, r = hi ^ lo, hi >> np.uint64(58)
        value = (x >> r) | (x << ((np.uint64(64) - r) & np.uint64(63)))
        out.append((value >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0))
    return out, state


def to_limbs(value: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """``[N, 4]`` int64 uint32 limbs of 128-bit values, least significant first, as the State stores them."""
    hi, lo = value
    return np.stack([lo & _U32, lo >> np.uint64(32), hi & _U32, hi >> np.uint64(32)], 1).astype(np.int64)


def from_limbs(limbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of ``to_limbs``."""
    parts = np.asarray(limbs, dtype=np.int64).astype(np.uint64)
    return parts[:, 2] | (parts[:, 3] << np.uint64(32)), parts[:, 0] | (parts[:, 1] << np.uint64(32))
