"""Go/no-go for fusing the SHADER: is every remaining torch op reproducible in Triton, bit for bit?

The cast needed three knobs (``enable_fp_fusion=False``, ``div_rn``, ``sqrt_rn``) and they took four
shapes from ~1 M differing patterns each to exactly zero. The shader introduces three more shapes
that the cast never used, and each is a separate question:

  * ``x ** y`` for fp32 tensors — the specular lobe, ``ndh ** (128 * shininess)`` and the spotlight
    falloff ``se ** exponent``. torch lowers this to ``powf``; Triton has ``tl.math.pow``. If the two
    are not the same libdevice routine this is where it shows.
  * ``(v*v).sum(-1)`` — ``_norm``'s length. This is a REDUCTION, not an unrolled dot: torch chooses
    the accumulation order and Triton cannot be told what it chose, so the two have to be checked
    rather than assumed. (``_dot`` in the same module is written out precisely because of this, so
    the question is only about ``_norm``.)
  * ``torch.clamp(x, 0, 1)`` and ``torch.clamp(x, min=0)`` against ``tl.minimum``/``tl.maximum``.

Anything that does not come back at zero is a shape the fused shader must not contain.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

N = 1 << 22
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)


def bits(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def rnd(lo=-1.0, hi=1.0):
    return torch.rand(N, generator=g, device=dev, dtype=torch.float32) * (hi - lo) + lo


@triton.jit
def _k_pow(x, y, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.math.pow(tl.load(x + i, mask=m), tl.load(y + i, mask=m)), mask=m)


@triton.jit
def _k_len(vx, vy, vz, out, n, BLOCK: tl.constexpr):
    """``sum(v*v, -1)`` written out left-to-right, which is what a kernel can do."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(vx + i, mask=m)
    b = tl.load(vy + i, mask=m)
    c = tl.load(vz + i, mask=m)
    tl.store(out + i, a * a + b * b + c * c, mask=m)


@triton.jit
def _k_clamp(x, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(x + i, mask=m)
    tl.store(out + i, tl.minimum(tl.maximum(v, 0.0), 1.0), mask=m)


@triton.jit
def _k_relu(x, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.maximum(tl.load(x + i, mask=m), 0.0), mask=m)


grid = (triton.cdiv(N, 1024),)
rows = []

# --- 1. the specular lobe. Two regimes: a per-pixel exponent (128*shininess) and a per-light one.
base = rnd(0.0, 1.0)
for lbl, expo in (("exponent 64 (shininess .5)", torch.full((N,), 64.0, device=dev)),
                  ("exponent 10 (spot falloff)", torch.full((N,), 10.0, device=dev)),
                  ("exponent random 0-128", rnd(0.0, 128.0))):
    ref = torch.pow(base, expo)
    got = torch.empty_like(ref)
    _k_pow[grid](base, expo, got, N, BLOCK=1024, enable_fp_fusion=False)
    rows.append((f"x ** y   {lbl}", bits(ref, got)))

# --- 2. _norm's length: a 3-element reduction against an unrolled sum
v = torch.stack([rnd(), rnd(), rnd()], -1).contiguous()
ref = (v * v).sum(-1)
got = torch.empty_like(ref)
_k_len[grid](v[:, 0].contiguous(), v[:, 1].contiguous(), v[:, 2].contiguous(), got, N,
             BLOCK=1024, enable_fp_fusion=False)
rows.append(("(v*v).sum(-1) vs unrolled", bits(ref, got)))

# ...and against the unrolled form written in TORCH, which is what the kernel really has to match if
# _norm is rewritten to use it
ref2 = v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]
rows.append(("(v*v).sum(-1) vs torch unrolled", bits(ref, ref2)))

# --- 3. the clamps
x = rnd(-2.0, 2.0)
ref = torch.clamp(x, 0, 1)
got = torch.empty_like(ref)
_k_clamp[grid](x, got, N, BLOCK=1024, enable_fp_fusion=False)
rows.append(("clamp(x, 0, 1)", bits(ref, got)))
ref = torch.clamp(x, min=0)
got = torch.empty_like(ref)
_k_relu[grid](x, got, N, BLOCK=1024, enable_fp_fusion=False)
rows.append(("clamp(x, min=0)", bits(ref, got)))

print(f"triton {triton.__version__}  torch {torch.__version__}\n")
for lbl, n in rows:
    flag = "OK" if n == 0 else "DIFFERS"
    print(f"  {lbl:38s} {n:>12,} / {N:,}   {flag}")
bad = [lbl for lbl, n in rows if n]
print("\nVERDICT:", "every shader shape is bit-reproducible in Triton" if not bad else
      f"NOT reproducible: {bad}")
