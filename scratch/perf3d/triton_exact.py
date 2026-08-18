"""Go/no-go for a fused 3-D kernel: is Triton fp32 BIT-IDENTICAL to a chain of torch elementwise ops?

The parity tier's whole claim is that the fast path and the reference path produce the same bytes, so
a fused kernel is only admissible if its arithmetic agrees with the torch expression it replaces to
the last bit. Two ways it can fail and both are silent:

  * **FMA contraction.** ``a*b + c`` is two rounded operations in torch (one kernel each) and can be
    one ``fma.rn.f32`` in Triton, which rounds once. That is a better answer and the wrong one here.
  * **Division / sqrt lowering.** ``div.approx`` and ``sqrt.approx`` are not correctly rounded; the
    ``.rn`` forms are.

Each probe below is the exact expression shape ``raycast3d`` uses, run on adversarial inputs (random
mantissas, near-cancelling sums, silhouette-grazing discriminants), and scored by counting rows whose
32-bit patterns differ rather than by a tolerance.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

N = 1 << 22
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


def bits_differ(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


@triton.jit
def _k_dot(ax, ay, az, bx, by, bz, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = (tl.load(ax + i, mask=m) * tl.load(bx + i, mask=m)
         + tl.load(ay + i, mask=m) * tl.load(by + i, mask=m)
         + tl.load(az + i, mask=m) * tl.load(bz + i, mask=m))
    tl.store(out + i, r, mask=m)


@triton.jit
def _k_quad(b, c, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    bb = tl.load(b + i, mask=m)
    cc = tl.load(c + i, mask=m)
    disc = bb * bb - cc
    sq = tl.sqrt(tl.maximum(disc, 0.0))
    tl.store(out + i, -bb - sq, mask=m)


@triton.jit
def _k_div(a, b, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(a + i, mask=m) / tl.load(b + i, mask=m), mask=m)


@triton.jit
def _k_madd(a, b, c, out, n, BLOCK: tl.constexpr):
    """The single most dangerous shape: one multiply feeding one add."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(a + i, mask=m) * tl.load(b + i, mask=m) + tl.load(c + i, mask=m),
             mask=m)


def rnd(scale=1.0):
    return (torch.rand(N, generator=g, device=dev, dtype=torch.float32) - 0.5) * 2 * scale


grid = (triton.cdiv(N, 1024),)
fails = 0

# --- 1. a*b + c ------------------------------------------------------------------------------
a, b, c = rnd(), rnd(), rnd()
ref = a * b + c
got = torch.empty_like(ref)
_k_madd[grid](a, b, c, got, N, BLOCK=1024)
n = bits_differ(ref, got)
fails += n > 0
print(f"a*b + c                      : {n:>10,} / {N:,} bits differ   "
      f"maxabs {(ref - got).abs().max().item():.3e}")

# --- 2. three-term dot, the _dot / _mv shape --------------------------------------------------
ax, ay, az, bx, by, bz = (rnd() for _ in range(6))
ref = ax * bx + ay * by + az * bz
got = torch.empty_like(ref)
_k_dot[grid](ax, ay, az, bx, by, bz, got, N, BLOCK=1024)
n = bits_differ(ref, got)
fails += n > 0
print(f"ax*bx + ay*by + az*bz (_dot) : {n:>10,} / {N:,} bits differ   "
      f"maxabs {(ref - got).abs().max().item():.3e}")

# --- 3. the ray quadratic, on near-cancelling inputs ------------------------------------------
bq = rnd(4.0)
cq = bq * bq * (1.0 + rnd(1e-4))            # disc = b*b - c is a catastrophic cancellation
ref = -bq - torch.sqrt(torch.clamp(bq * bq - cq, min=0))
got = torch.empty_like(ref)
_k_quad[grid](bq, cq, got, N, BLOCK=1024)
n = bits_differ(ref, got)
fails += n > 0
print(f"-b - sqrt(max(b*b - c, 0))   : {n:>10,} / {N:,} bits differ   "
      f"maxabs {(ref - got).abs().max().item():.3e}")

# --- 4. division ------------------------------------------------------------------------------
da, db = rnd(), rnd()
db = torch.where(db.abs() < 1e-6, torch.full_like(db, 1e-6), db)
ref = da / db
got = torch.empty_like(ref)
_k_div[grid](da, db, got, N, BLOCK=1024)
n = bits_differ(ref, got)
fails += n > 0
print(f"a / b                        : {n:>10,} / {N:,} bits differ   "
      f"maxabs {(ref - got).abs().max().item():.3e}")

print("\nVERDICT:", "bit-exact on every probe" if fails == 0 else
      f"{fails} probe(s) disagree -- a fused kernel cannot be dropped in without a fix")
