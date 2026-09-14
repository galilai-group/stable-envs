"""Which Triton spelling of ``exp`` is torch's cuda float64 ``exp``?

The SWMNAV-2 feasibility probe found ``dist2`` bit-identical once fp fusion is off, and the float64
``exp`` NOT: 628,757 of 4,194,304 differ by up to 5.96e-08, which is one ulp of float32 near 1.0 --
the signature of a transcendental evaluated in the wrong precision rather than of a different
algorithm. ``_gaussian_dot`` widens to float64 for exactly this reason ("CUDA's float32 expf and the
CPU's differ by up to one ulp, the alpha then multiplies a background of 255, and .to(uint8)
truncates"), so a kernel that quietly computes it in float32 reintroduces the bug the port was
written to avoid.

This tries every spelling this Triton exposes, against the torch expression, and says which one is
the function. A row of zeros is the answer; if no row is zero, the kernel cannot carry this
expression and SWMNAV-2 has to keep the dot in torch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@triton.jit
def _k_tl_exp(a_p, o_p, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(a_p + o, mask=m, other=0.0)
    tl.store(o_p + o, tl.exp(x).to(tl.float32), mask=m)


@triton.jit
def _k_math_exp(a_p, o_p, N, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(a_p + o, mask=m, other=0.0)
    tl.store(o_p + o, tl.math.exp(x).to(tl.float32), mask=m)


@triton.jit
def _k_libdevice_exp(a_p, o_p, N, BLOCK: tl.constexpr):
    from triton.language.extra import libdevice
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(a_p + o, mask=m, other=0.0)
    tl.store(o_p + o, libdevice.exp(x).to(tl.float32), mask=m)


@triton.jit
def _k_exp_f32(a_p, o_p, N, BLOCK: tl.constexpr):
    """The wrong one on purpose: the whole thing in float32, to size what the widening buys."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(a_p + o, mask=m, other=0.0).to(tl.float32)
    tl.store(o_p + o, tl.exp(x), mask=m)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 22)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    N = a.n
    # the argument -dist2/denom actually reaches: 0 down to well past float32 underflow
    arg = -(torch.rand(N, generator=g, device=dev, dtype=torch.float64) ** 2) * 120.0
    want = torch.exp(arg).to(torch.float32)
    rows = []
    print(f"n = {N:,}   torch {torch.__version__}   triton {triton.__version__}")
    for name, k in (("tl.exp(float64)", _k_tl_exp), ("tl.math.exp(float64)", _k_math_exp),
                    ("libdevice.exp(float64)", _k_libdevice_exp),
                    ("tl.exp(float32) [control]", _k_exp_f32)):
        out = torch.empty(N, device=dev, dtype=torch.float32)
        BLOCK = 1024
        try:
            k[(triton.cdiv(N, BLOCK),)](arg, out, N, BLOCK=BLOCK, num_warps=4,
                                        enable_fp_fusion=False)
        except Exception as e:                                          # noqa: BLE001
            print(f"  {name:28s} unavailable: {type(e).__name__}: {str(e)[:70]}")
            rows.append({"spelling": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        diff = int((want != out).sum())
        worst = float((want.double() - out.double()).abs().max()) if diff else 0.0
        print(f"  {name:28s} {diff:>12,} / {N:,} differ   max |diff| {worst:.3e}"
              + ("   <-- this is the function" if not diff else ""))
        rows.append({"spelling": name, "differing": diff, "n": N, "max_abs_diff": worst})
    ok = [r for r in rows if r.get("differing") == 0]
    print(("EXACT: " + ok[0]["spelling"]) if ok else "NO SPELLING MATCHES torch cuda exp(float64)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"n": N, "torch": torch.__version__,
                                           "triton": triton.__version__,
                                           "card": torch.cuda.get_device_name(0),
                                           "rows": rows,
                                           "exact": ok[0]["spelling"] if ok else None}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
