"""Which Triton spelling of ``pow`` exists here, and which one is bit-identical to ``torch.pow``?

``tl.math.pow`` does not exist in Triton 3.6 (``CompilationError`` at the call site), so before the
shader can be fused the question "can the specular lobe be reproduced" has to be answered against
whatever the installed Triton actually offers. Every candidate is compiled and scored; a name that
does not exist is reported as missing rather than skipped silently, because "no name works" is the
answer that stops the fusion.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

print(f"triton {triton.__version__}\n")
print("tl.math members with 'pow'/'exp'/'log':",
      sorted(n for n in dir(tl.math) if any(k in n for k in ("pow", "exp", "log"))))
print("tl members with 'pow':", sorted(n for n in dir(tl) if "pow" in n))
try:
    from triton.language.extra import libdevice
    print("libdevice members with 'pow':", sorted(n for n in dir(libdevice) if "pow" in n))
except Exception as e:                                                       # noqa: BLE001
    print("triton.language.extra.libdevice unavailable:", e)
try:
    from triton.language.extra.cuda import libdevice as cudev
    print("extra.cuda.libdevice with 'pow':", sorted(n for n in dir(cudev) if "pow" in n))
except Exception as e:                                                       # noqa: BLE001
    print("triton.language.extra.cuda.libdevice unavailable:", e)

N = 1 << 22
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
grid = (triton.cdiv(N, 1024),)


def bits(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


@triton.jit
def _pow_libdev(x, y, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, _ld.pow(tl.load(x + i, mask=m), tl.load(y + i, mask=m)), mask=m)


@triton.jit
def _pow_operator(x, y, out, n, BLOCK: tl.constexpr):
    """``**`` in Triton, whatever it lowers to."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, tl.load(x + i, mask=m) ** tl.load(y + i, mask=m), mask=m)


@triton.jit
def _pow_exp2(x, y, out, n, BLOCK: tl.constexpr):
    """``exp2(y * log2(x))``, the identity a hand-rolled lobe would use."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(x + i, mask=m)
    b = tl.load(y + i, mask=m)
    tl.store(out + i, tl.exp2(b * tl.log2(a)), mask=m)


cases = {"libdevice.pow": _pow_libdev, "exp2(y*log2(x))": _pow_exp2}

# The two regimes the shader actually uses, plus a hostile one. Base in [0,1] because it is a clamped
# cosine; exponent 64 is every DM Control material (shininess 0.5) and 10 is the spotlight falloff.
regimes = {
    "base[0,1] exp=64": (torch.rand(N, generator=g, device=dev), torch.full((N,), 64.0, device=dev)),
    "base[0,1] exp=10": (torch.rand(N, generator=g, device=dev), torch.full((N,), 10.0, device=dev)),
    "base[0,1] exp rand": (torch.rand(N, generator=g, device=dev),
                           torch.rand(N, generator=g, device=dev) * 128),
}

print()
for cname, kern in cases.items():
    for rname, (x, y) in regimes.items():
        ref = torch.pow(x, y)
        got = torch.empty_like(ref)
        try:
            kern[grid](x, y, got, N, BLOCK=1024, enable_fp_fusion=False)
            n = bits(ref, got)
            worst = (ref - got).abs().max().item()
            print(f"  {cname:18s} {rname:20s} {n:>12,} / {N:,} differ   max |d| {worst:.3e}"
                  f"   {'OK' if n == 0 else ''}")
        except Exception as e:                                               # noqa: BLE001
            print(f"  {cname:18s} {rname:20s} UNAVAILABLE: {type(e).__name__}: "
                  f"{str(e).splitlines()[-1][:80]}")
