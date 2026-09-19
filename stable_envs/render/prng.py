"""JAX's threefry2x32 counter PRNG, in torch, bit for bit.

A renderer that reproduces a JAX environment's frames has to reproduce its *noise* too, and JAX's
noise is not a stream — it is a **counter** PRNG. ``jax.random.uniform(key, shape)`` is a pure
function of ``(key, shape)`` with no hidden cursor, so element ``i`` of the output can be computed
from ``i`` alone. That is the property this module needs and the reason a torch port is possible at
all: there is nothing to seed, nothing to advance, and no order dependence.

Craftax Classic's night frames mix in ``jax.random.uniform(state.state_rng, (H, W)) * 95 + 32``
(``craftax.craftax_classic.renderer``), and full Craftax draws the same field at its own window
size. Both are served from here. Nothing in this module knows about either environment.

**Which of JAX's two code paths this is, and why it matters.** ``threefry_random_bits`` has two
implementations and they produce *different bits for the same key*:

* ``_threefry_random_bits_partitionable`` — the counter for output element ``i`` is the 64-bit
  value ``i`` split into ``(hi, lo)`` uint32 halves, both halves go through one threefry2x32 block,
  and the 32 output bits are ``bits1 ^ bits2``. Elementwise, and therefore one fused pass here.
* ``_threefry_random_bits_original`` — pairs element ``i`` with element ``i + n/2`` in a single
  block and concatenates the halves.

Which one runs is decided by the ``jax_threefry_partitionable`` config flag, whose default flipped
to ``True`` in jax 0.4.36. **This module implements the partitionable path only**, because that is
what every supported jax draws. Measured: the partitionable path reproduces
``jax.random.uniform`` on 448/448 draws across seven shapes and 64 keys, and the ``original`` path
matches 0/448 (``examples/results/step7/craftax-classic-full/phase0/prng_exactness.json``). A
caller whose reference runs with the flag off is comparing against a different picture, and the
environment that depends on this must say so in its ``reference_versions`` declaration rather than
leave it to be discovered.

**Why int32 and not int64.** Two's-complement wraparound *is* arithmetic mod 2**32, so ``+`` and
``<<`` need no masking at all in int32; only the logical right shift does, because torch's ``>>``
on a signed dtype is arithmetic. A shift right by ``32 - d`` keeps ``d`` bits, so the mask is
``(1 << d) - 1`` — **not** ``(1 << (32 - d)) - 1``, which is the plausible-looking version and
produces a hash that is wrong everywhere while looking like noise. Measured on one Apple arm64 core
at ``B=256``, ``(112,144)``: int32 is bit-identical to an int64-with-masking implementation and to
jax, and **2.3x faster** (190.7 ms -> 82.8 ms).
"""

from __future__ import annotations

import torch

#: The five groups of four rotation distances, alternating (``_threefry2x32_lowering``).
_ROTATIONS = ((13, 15, 26, 6), (17, 29, 16, 24))
#: Threefry's key-schedule parity constant.
_PARITY = 0x1BD11BDA
#: float32 1.0 as an integer bit pattern, and the mantissa mask.
_ONE_BITS = 0x3F800000
_MANTISSA = 0x7FFFFF


def _rotl32(x: torch.Tensor, d: int) -> torch.Tensor:
    """Rotate left by ``d`` on int32 holding a uint32 bit pattern."""
    return (x << d) | ((x >> (32 - d)) & ((1 << d) - 1))


def threefry2x32(
    k1: torch.Tensor, k2: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The 20-round Threefry-2x32 hash. All arguments int32 and mutually broadcastable.

    ``k1, k2`` are the two halves of the key; ``x1, x2`` the two halves of the counter. Returns the
    two 32-bit output words. This is ``jax/_src/random/threefry2x32.py:_threefry2x32_lowering`` with
    the rolled loop, transcribed: five groups of four rounds, with the three-entry key schedule and
    the two rotation sets each rotating by one position per group.
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


def _counter(n: int, offset: int, device, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """``(hi, lo)`` int32 halves of the 64-bit counters ``offset .. offset+n-1``.

    Built in int64 as a length-``n`` vector — one cheap allocation, not a ``(B, n)`` one — and
    narrowed, so a counter past 2**31 still splits correctly instead of wrapping into the sign bit.
    """
    i = torch.arange(int(offset), int(offset) + int(n), dtype=torch.int64, device=device)
    # narrowing a value >= 2**31 to int32 wraps into the sign bit, which is the bit pattern wanted
    return (i >> 32).to(dtype), (i & 0xFFFFFFFF).to(dtype)


def random_bits(keys: torch.Tensor, n: int, offset: int = 0) -> torch.Tensor:
    """``(B, n)`` int32 holding the uint32 bit patterns ``jax.random.bits`` would produce.

    ``keys`` is ``(B, 2)`` int32 — the raw two-word threefry key, the thing
    ``jax.random.PRNGKey`` returns. ``offset`` is the counter index of the first element, ``0`` for
    a plain ``jax.random.uniform(key, shape)`` call; it exists so a caller can draw one very large
    field in chunks and still get the field jax would have drawn in one call.
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


def uniform(keys: torch.Tensor, shape: int | tuple[int, ...], offset: int = 0) -> torch.Tensor:
    """``jax.random.uniform(key, shape)`` on ``[0, 1)`` for a batch of keys.

    ``keys`` ``(B, 2)`` int32 -> ``(B, *shape)`` float32. The construction is JAX's: randomise the
    mantissa of an exponent-1 float and subtract one, which is why the result is exactly
    representable and why matching it is a bit-level claim rather than a statistical one.

    **There are deliberately no ``minval``/``maxval`` arguments, and that is a correctness
    decision.** ``jax.random.uniform(key, shape, dtype, minval, maxval)`` computes
    ``floats * (maxval - minval) + minval``, and under ``jax.jit`` XLA contracts that into a **fused
    multiply-add**, which rounds once where a separate multiply and add round twice. Measured: for
    the span ``[32, 127)`` at shape ``(32, 127)``, **508 of 4096 values differ** between jax's jitted
    range call and ``uniform() * 95 + 32``; a power-of-two span happens to agree, which is exactly
    the kind of coincidence that makes such a bug survive a test. A range argument here could
    therefore be bit-exact against one jax call and not another, depending on a fusion decision the
    caller cannot see.

    Both craftax renderers write the scaling themselves and un-jitted
    (``jax.random.uniform(state.state_rng, shape) * 95 + 32``), so a caller reproduces them by
    writing the same expression. That is one fewer place for the two to silently diverge.
    ``tests/unit/test_craftax_prng.py:test_the_scaled_night_field_matches`` pins exactly that
    expression against jax-generated values.
    """
    n = 1
    for d in shape:
        n *= int(d)
    bits = random_bits(keys, n, offset)
    floats = (((bits >> 9) & _MANTISSA) | _ONE_BITS).view(torch.float32) - 1.0
    return floats.reshape(keys.shape[0], *[int(d) for d in shape])


def key_from_halves(halves: torch.Tensor) -> torch.Tensor:
    """``(B, 4)`` of 16-bit halves -> ``(B, 2)`` int32 keys.

    A key does **not** survive a float32 state row: only 2.008% of random uint32 round-trip through
    float32, because 2**32 > 2**24. Split into 16-bit halves, 100% of 100,000 draws round-trip
    (``phase0/purity_and_state_window.json``), which is why an environment that carries a key in its
    state carries four numbers rather than two. Halves are ``(hi0, lo0, hi1, lo1)``.
    """
    h = halves.to(torch.int64) & 0xFFFF
    w0 = (h[:, 0] << 16) | h[:, 1]
    w1 = (h[:, 2] << 16) | h[:, 3]
    return torch.stack([w0, w1], dim=1).to(torch.int32)


def key_to_halves(keys: torch.Tensor) -> torch.Tensor:
    """``(B, 2)`` int32 keys -> ``(B, 4)`` int64 of 16-bit halves. The inverse of the above."""
    k = keys.to(torch.int64) & 0xFFFFFFFF
    return torch.stack([(k[:, 0] >> 16) & 0xFFFF, k[:, 0] & 0xFFFF, (k[:, 1] >> 16) & 0xFFFF, k[:, 1] & 0xFFFF], dim=1)
