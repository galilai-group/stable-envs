"""Can a Triton kernel reproduce swm_nav's arithmetic BIT-FOR-BIT? Asked before it is written.

SWMNAV-2's hard requirement is bit-identity with the torch path, and the whole ticket rests on two
expressions this module does not share with any kernel in the repo:

    _gaussian_dot:  dist2 = dx*dx + dy*dy            (float32, two roundings)
                    dot   = exp((-dist2/denom).double()).to(float32)
                            -- the transcendental in FLOAT64, rounded once
                    alpha = dot / dot.amax()
    _alpha_blend:   out   = img.to(f32) * (1 - a) + trunc(rgb) * a  ->  .to(uint8)

``render/raycast3d.py`` already established what makes a Triton float32 expression match torch:
``enable_fp_fusion=False`` so ``a*b + c`` is not contracted into one ``fma``, ``tl.math.div_rn`` for
every division, ``tl.sqrt_rn`` for every root. What it did NOT have to establish is whether a
float64 ``exp`` in Triton is the same function torch's cuda ``exp`` is -- both should reach
libdevice, and "should" is not a measurement.

So this probe is the feasibility gate: the same expressions, both ways, over random inputs drawn to
cover the ranges the envs actually reach and the ones they do not. A nonzero count in any row means
the kernel cannot be bit-identical as written and the ticket changes shape rather than the contract.

    python scratch/step6/raster2d/swmnav_exact_probe.py --n 4194304
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import triton                                                    # noqa: E402
import triton.language as tl                                     # noqa: E402


@triton.jit
def _probe_kernel(gx_p, gy_p, px_p, py_p, den_p, m_p, img_p, col_p,
                  o_dist2, o_dot, o_alpha, o_blend, N, BLOCK: tl.constexpr):
    """Every expression the kernel would need, one element per lane, written the torch way."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < N
    gx = tl.load(gx_p + o, mask=m, other=0.0)
    gy = tl.load(gy_p + o, mask=m, other=0.0)
    px = tl.load(px_p + o, mask=m, other=0.0)
    py = tl.load(py_p + o, mask=m, other=0.0)
    den = tl.load(den_p + o, mask=m, other=1.0)
    mx = tl.load(m_p + o, mask=m, other=1.0)
    img = tl.load(img_p + o, mask=m, other=0.0)
    col = tl.load(col_p + o, mask=m, other=0.0)

    dx = gx - px
    dy = gy - py
    dist2 = dx * dx + dy * dy                       # two roundings; fp fusion must be OFF
    tl.store(o_dist2 + o, dist2, mask=m)

    # exp in float64, rounded once back to float32 -- the reference's own shape, and the DIVISION
    # is float32. torch writes ``(-dist2 / denom).double()``, so it divides and THEN widens; dividing
    # in float64 instead is a different number and cost this probe its first run (628,757 of
    # 4,194,304 differing, which looked like Triton's exp and was this line).
    arg = tl.math.div_rn(-dist2, den)
    dot = tl.exp(arg.to(tl.float64)).to(tl.float32)
    tl.store(o_dot + o, dot, mask=m)

    alpha = tl.where(mx > 0.0, tl.math.div_rn(dot, mx), dot)
    tl.store(o_alpha + o, alpha, mask=m)

    a = tl.minimum(tl.maximum(alpha, 0.0), 1.0)
    out = img * (1.0 - a) + col * a
    tl.store(o_blend + o, out.to(tl.uint8), mask=m)


def torch_side(gx, gy, px, py, den, mx, img, col):
    dx, dy = gx - px, gy - py
    dist2 = dx * dx + dy * dy
    dot = torch.exp((-dist2 / den).double()).to(torch.float32)
    alpha = torch.where(mx > 0, dot / mx, dot)
    a = alpha.clamp(0.0, 1.0)
    blend = (img * (1.0 - a) + col * a).to(torch.uint8)
    return dist2, dot, alpha, blend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion", action="store_true", help="leave fp fusion ON, to show it matters")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(a.seed)
    N = a.n
    r = lambda *s: torch.rand(*s, generator=g, device=dev, dtype=torch.float32)   # noqa: E731

    # the ranges the envs reach: a 224 px lattice, centres on and off the frame, sigma 7..14,
    # backgrounds and colours over the whole uint8 range, and frame maxima from 0 to 1
    gx = torch.arange(N, device=dev, dtype=torch.float32) % 224.0
    gy = (torch.arange(N, device=dev, dtype=torch.float32) // 224.0) % 224.0
    px = r(N) * 640.0 - 208.0
    py = r(N) * 640.0 - 208.0
    sig = (r(N) * 7.0 + 7.0).double()
    den = (2.0 * sig * sig).to(torch.float32)
    mx = torch.where(r(N) < 0.02, torch.zeros(N, device=dev), r(N))
    img = (r(N) * 255.0).trunc()
    col = (r(N) * 255.0).trunc()

    outs = [torch.empty(N, device=dev, dtype=t) for t in
            (torch.float32, torch.float32, torch.float32, torch.uint8)]
    BLOCK = 1024
    _probe_kernel[(triton.cdiv(N, BLOCK),)](gx, gy, px, py, den, mx, img, col, *outs, N,
                                            BLOCK=BLOCK, num_warps=4,
                                            enable_fp_fusion=bool(a.fusion))
    want = torch_side(gx, gy, px, py, den, mx, img, col)
    names = ("dist2", "dot (float64 exp)", "alpha = dot/max", "blend -> uint8")
    rows, bad = [], 0
    print(f"n = {N:,}   enable_fp_fusion = {bool(a.fusion)}")
    for name, w, got in zip(names, want, outs):
        diff = int((w != got).sum())
        bad += diff
        worst = 0.0
        if diff:
            d = (w.double() - got.double()).abs()
            worst = float(d.max())
        print(f"  {name:22s} {diff:>12,} / {N:,} differ   max |diff| {worst:.3e}")
        rows.append({"expr": name, "differing": diff, "n": N, "max_abs_diff": worst})
    print("FEASIBLE: every expression is bit-identical" if not bad else
          "NOT bit-identical as written")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"n": N, "enable_fp_fusion": bool(a.fusion), "triton": triton.__version__,
             "torch": torch.__version__, "card": torch.cuda.get_device_name(0),
             "rows": rows, "bit_identical": not bad}, indent=2))
        print("wrote", a.out)
    raise SystemExit(1 if bad and not a.fusion else 0)


if __name__ == "__main__":
    main()
