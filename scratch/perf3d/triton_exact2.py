"""Can Triton's fp32 be made bit-identical to torch's, or is the fused 3-D kernel structurally out?

``triton_exact.py`` says the default lowering disagrees on all four shapes ``raycast3d`` is built from.
This asks whether the two known knobs close it:

  * ``enable_fp_fusion=False`` at launch, which is Triton's ``-ffp-contract=off`` -- if the ``a*b + c``
    disagreement is FMA contraction, this removes it and nothing else has to change.
  * ``tl.math.div_rn`` / an IEEE sqrt, in case the division and the root are lowering to the
    ``.approx`` forms rather than the correctly-rounded ones.

If both close, the fused kernel is back on the table. If either does not, it is out at the parity
tier, because a renderer that disagrees with the reference by one ulp on a silhouette ray flips whole
pixels (module docstring: 1.2e-7 in ``dl`` -> 2.4e-4 in ``disc``, and ``disc > 0`` is a hit test).
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


def rnd(scale=1.0):
    return (torch.rand(N, generator=g, device=dev, dtype=torch.float32) - 0.5) * 2 * scale


@triton.jit
def _k_madd(a, b, c, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(a + i, mask=m) * tl.load(b + i, mask=m) + tl.load(c + i, mask=m),
             mask=m)


@triton.jit
def _k_dot(ax, ay, az, bx, by, bz, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    r = (tl.load(ax + i, mask=m) * tl.load(bx + i, mask=m)
         + tl.load(ay + i, mask=m) * tl.load(by + i, mask=m)
         + tl.load(az + i, mask=m) * tl.load(bz + i, mask=m))
    tl.store(out + i, r, mask=m)


@triton.jit
def _k_div(a, b, out, n, IEEE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(a + i, mask=m)
    y = tl.load(b + i, mask=m)
    if IEEE:
        r = tl.math.div_rn(x, y)
    else:
        r = x / y
    tl.store(out + i, r, mask=m)


@triton.jit
def _k_sqrt(a, out, n, RN: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(a + i, mask=m)
    r = tl.sqrt_rn(x) if RN else tl.sqrt(x)
    tl.store(out + i, r, mask=m)


grid = (triton.cdiv(N, 1024),)
report = []

a, b, c = rnd(), rnd(), rnd()
ref = a * b + c
for fuse in (True, False):
    got = torch.empty_like(ref)
    _k_madd[grid](a, b, c, got, N, BLOCK=1024, enable_fp_fusion=fuse)
    report.append((f"a*b + c   enable_fp_fusion={fuse}", bits(ref, got)))

ax, ay, az, bx, by, bz = (rnd() for _ in range(6))
ref = ax * bx + ay * by + az * bz
for fuse in (True, False):
    got = torch.empty_like(ref)
    _k_dot[grid](ax, ay, az, bx, by, bz, got, N, BLOCK=1024, enable_fp_fusion=fuse)
    report.append((f"_dot      enable_fp_fusion={fuse}", bits(ref, got)))

da, db = rnd(), rnd()
db = torch.where(db.abs() < 1e-6, torch.full_like(db, 1e-6), db)
ref = da / db
for ieee in (False, True):
    got = torch.empty_like(ref)
    try:
        _k_div[grid](da, db, got, N, IEEE=ieee, BLOCK=1024, enable_fp_fusion=False)
        report.append((f"a / b     div_rn={ieee}", bits(ref, got)))
    except Exception as e:
        report.append((f"a / b     div_rn={ieee}", f"UNAVAILABLE {type(e).__name__}: {e}"))

sa = rnd().abs() + 1e-6
ref = torch.sqrt(sa)
for rn in (False, True):
    got = torch.empty_like(ref)
    try:
        _k_sqrt[grid](sa, got, N, RN=rn, BLOCK=1024, enable_fp_fusion=False)
        report.append((f"sqrt      sqrt_rn={rn}", bits(ref, got)))
    except Exception as e:
        report.append((f"sqrt      sqrt_rn={rn}", f"UNAVAILABLE {type(e).__name__}: {e}"))

print(f"triton {triton.__version__}   torch {torch.__version__}\n")
for lbl, n in report:
    print(f"  {lbl:36s} {n if isinstance(n, str) else f'{n:>12,} / {N:,} bits differ'}")
ok = all(isinstance(n, int) and n == 0 for lbl, n in report
         if "enable_fp_fusion=False" in lbl or "div_rn=True" in lbl or "sqrt_rn=True" in lbl
         or ("sqrt_rn=False" in lbl))
print("\nVERDICT:", "the knobs close it -- a fused kernel can be bit-exact" if ok else
      "at least one shape still disagrees; see the rows above")
