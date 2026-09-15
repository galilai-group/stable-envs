"""A torch reimplementation of JAX's threefry2x32 counter PRNG and ``jax.random.uniform``.

Phase-0 prototype. Not yet a stable_envs module.

Two facts decide the shape of this code and both were read out of the installed jax, not assumed:

* ``jax_threefry_partitionable`` defaults to **True** in jax 0.11.1
  (``jax/_src/config.py:1411``). Under that flag ``threefry_random_bits`` is
  ``_threefry_random_bits_partitionable``: the counter for output element ``i`` is the 64-bit
  value ``i`` split into ``(hi, lo)`` uint32 halves, both halves go through one threefry2x32
  block, and the 32 output bits are ``bits1 ^ bits2``. That is **elementwise** — element ``i``
  depends only on ``i`` and the key. The pre-0.4.36 ``_original`` path instead pairs element
  ``i`` with element ``i + n/2`` in a single block and returns the halves concatenated, which
  gives DIFFERENT bits for the same key. Both are implemented here so the difference is
  measurable rather than argued.
* ``uniform`` randomises only the mantissa: ``bitcast_f32((bits >> 9) | 0x3f800000) - 1.0``,
  then ``max(minval, f*(maxval-minval) + minval)``.

Everything is done in ``int64`` holding 32-bit values, masked after every add and shift. torch's
``uint32`` exists but its arithmetic/bitwise coverage is partial and device-dependent; int64 is
exact here because every intermediate is < 2**32 and torch's ``<<``/``>>``/``^``/``&`` on int64
are exact.
"""
from __future__ import annotations

import torch

M32 = 0xFFFFFFFF
_ROT = ((13, 15, 26, 6), (17, 29, 16, 24))


def _rotl(x: torch.Tensor, d: int) -> torch.Tensor:
    return ((x << d) | (x >> (32 - d))) & M32


def threefry2x32(k1, k2, x1, x2):
    """The 20-round hash. ``k1,k2`` scalars or broadcastable; ``x1,x2`` the counter halves.

    Follows ``jax/_src/random/threefry2x32.py:_threefry2x32_lowering`` with the rolled loop
    (five groups of four rounds, key schedule rotating by one each group).
    """
    ks = [k1 & M32, k2 & M32, (k1 ^ k2 ^ 0x1BD11BDA) & M32]
    x = [(x1 + ks[0]) & M32, (x2 + ks[1]) & M32]
    ks = ks[1:] + ks[:1]                      # rotate_list(ks)
    rots = [_ROT[0], _ROT[1]]
    for i in range(5):
        for r in rots[0]:
            x[0] = (x[0] + x[1]) & M32
            x[1] = _rotl(x[1], r)
            x[1] = x[0] ^ x[1]
        x = [(x[0] + ks[0]) & M32, (x[1] + ks[1] + (i + 1)) & M32]
        ks = ks[1:] + ks[:1]
        rots = rots[1:] + rots[:1]
    return x[0], x[1]


def random_bits(key: torch.Tensor, n: int, partitionable: bool = True) -> torch.Tensor:
    """``n`` uint32 values (as int64) for a raw 2-element uint32 ``key``."""
    k1 = int(key[0]) & M32
    k2 = int(key[1]) & M32
    dev = key.device if isinstance(key, torch.Tensor) else "cpu"
    if partitionable:
        i = torch.arange(n, dtype=torch.int64, device=dev)
        hi = (i >> 32) & M32
        lo = i & M32
        b1, b2 = threefry2x32(k1, k2, hi, lo)
        return b1 ^ b2
    # the pre-0.4.36 path: one iota split into two halves, results concatenated
    odd = n % 2
    m = n + odd
    i = torch.arange(m, dtype=torch.int64, device=dev)
    if odd:
        i = i.clone()
        i[-1] = 0
    h = m // 2
    b1, b2 = threefry2x32(k1, k2, i[:h], i[h:])
    out = torch.cat([b1, b2])
    return out[:n]


def uniform(key: torch.Tensor, shape, minval=0.0, maxval=1.0,
            partitionable: bool = True) -> torch.Tensor:
    """``jax.random.uniform(key, shape, float32, minval, maxval)``, bit for bit."""
    n = 1
    for d in shape:
        n *= int(d)
    bits = random_bits(key, n, partitionable)
    float_bits = ((bits >> 9) | 0x3F800000).to(torch.int64)
    f = float_bits.to(torch.int32).view(torch.float32) - 1.0
    f = f * (maxval - minval) + minval
    return torch.clamp_min(f, minval).reshape(tuple(int(d) for d in shape))


def uniform_batch(keys: torch.Tensor, shape, minval=0.0, maxval=1.0,
                  partitionable: bool = True) -> torch.Tensor:
    """``(B,2)`` uint32 keys -> ``(B, *shape)``. One fused pass, no Python loop over the batch."""
    keys = keys.to(torch.int64)
    B = keys.shape[0]
    n = 1
    for d in shape:
        n *= int(d)
    k1 = (keys[:, 0] & M32).view(B, 1)
    k2 = (keys[:, 1] & M32).view(B, 1)
    i = torch.arange(n, dtype=torch.int64, device=keys.device).view(1, n)
    if partitionable:
        hi = (i >> 32) & M32
        lo = i & M32
        b1, b2 = threefry2x32(k1, k2, hi.expand(B, n), lo.expand(B, n))
        bits = b1 ^ b2
    else:
        raise NotImplementedError("batched original path not needed")
    float_bits = (bits >> 9) | 0x3F800000
    f = float_bits.to(torch.int32).view(torch.float32) - 1.0
    f = f * (maxval - minval) + minval
    return torch.clamp_min(f, minval).reshape(B, *[int(d) for d in shape])


# ---------------------------------------------------------------- int32 variant

def _rotl32(x: torch.Tensor, d: int) -> torch.Tensor:
    # torch's >> on a signed dtype is arithmetic, so the high bits must be masked back off;
    # + and << wrap in two's complement, which IS mod 2**32, so they need no mask at all.
    # a logical shift right by (32-d) keeps d bits, so the mask is (1<<d)-1, not (1<<(32-d))-1
    return (x << d) | ((x >> (32 - d)) & ((1 << d) - 1))


def threefry2x32_i32(k1, k2, x1, x2):
    """The same hash in int32. Two's-complement wraparound is exactly mod 2**32."""
    k1 = torch.as_tensor(k1, dtype=torch.int32) if not torch.is_tensor(k1) else k1.to(torch.int32)
    k2 = torch.as_tensor(k2, dtype=torch.int32) if not torch.is_tensor(k2) else k2.to(torch.int32)
    ks = [k1, k2, k1 ^ k2 ^ torch.tensor(0x1BD11BDA, dtype=torch.int32)]
    x = [x1 + ks[0], x2 + ks[1]]
    ks = ks[1:] + ks[:1]
    rots = [_ROT[0], _ROT[1]]
    for i in range(5):
        for r in rots[0]:
            x[0] = x[0] + x[1]
            x[1] = _rotl32(x[1], r)
            x[1] = x[0] ^ x[1]
        x = [x[0] + ks[0], x[1] + ks[1] + torch.tensor(i + 1, dtype=torch.int32)]
        ks = ks[1:] + ks[:1]
        rots = rots[1:] + rots[:1]
    return x[0], x[1]


def uniform_batch_i32(keys: torch.Tensor, shape, minval=0.0, maxval=1.0) -> torch.Tensor:
    """``(B,2)`` keys -> ``(B, *shape)`` float32, int32 arithmetic throughout."""
    B = keys.shape[0]
    n = 1
    for d in shape:
        n *= int(d)
    k = keys.to(torch.int64).to(torch.int32)
    k1, k2 = k[:, 0].view(B, 1), k[:, 1].view(B, 1)
    lo = torch.arange(n, dtype=torch.int32, device=keys.device).view(1, n).expand(B, n)
    hi = torch.zeros((), dtype=torch.int32, device=keys.device).expand(B, n)
    b1, b2 = threefry2x32_i32(k1, k2, hi, lo)
    bits = b1 ^ b2
    float_bits = ((bits >> 9) & 0x7FFFFF) | 0x3F800000
    f = float_bits.view(torch.float32) - 1.0
    f = f * (maxval - minval) + minval
    return torch.clamp_min(f, minval).reshape(B, *[int(d) for d in shape])
