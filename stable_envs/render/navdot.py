"""One fused kernel for the swm_nav family: a base image, a few Gaussian dots, and the blend.

TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones are the four rows in this repo that render with
no kernel at all. Their picture is a flat base image -- background, wall stripe, doors, zone tints,
border -- with one or two NORMALISED Gaussian dots alpha-composited over it, and the reference
composites them one at a time through uint8. Written as torch ops that is a dozen full-frame
temporaries per dot per row, and measured on a 3090 (job 6365727, gpu2109) it is what the family
costs:

    config           B     render     base     dot   blend   launches   float temporaries
    TwoRoom        4096    103.6 ms   2.2      2.4     2.0      2,662        14,112 MiB
    TwoRoomTarget  4096    172.0 ms   2.2    2x2.4   2x2.0      3,526        28,224 MiB
    Piecewise      4096     74.7 ms   0.4      2.4     2.0      1,092        14,112 MiB
    PiecewiseZones 4096     87.1 ms   1.1      2.4     2.0      1,605        14,112 MiB

(the per-stage figures are per 256-frame chunk, which is what ``render`` iterates). The output of a
256-frame TwoRoomTarget chunk is 36 MiB; getting there costs 1,764 MiB of float32 and float64
intermediates that no pixel ever reads twice. That is the whole mechanism -- not launch count, which
at ~170 per chunk is about 13% of the time -- and it is what a fused kernel removes: the dot is
never materialised, the blend never widens a frame, and the only traffic is the base in and the
uint8 out.

WHAT MAKES THIS ADMISSIBLE IS BIT-IDENTITY, NOT SPEED. ``_gaussian_dot``'s docstring records that
CUDA's float32 ``expf`` and the CPU's differ by up to one ulp, that the alpha then multiplies a
background of 255, and that ``.to(uint8)`` truncates -- so the port evaluates the transcendental in
float64 and rounds once. A kernel that quietly did it in float32 would reintroduce exactly the bug
the port was written to avoid: measured over 4.2 M arguments, a float32 ``exp`` differs from the
float64-then-round answer in 3,162,766 of them.

Four things make every expression here match the torch path bit for bit, and all four were measured
rather than assumed (``scratch/step6/raster2d/swmnav_exact_probe.py``, ``swmnav_exp_probe.py``):

  * ``enable_fp_fusion=False`` at launch. LLVM contracts ``a*b + c*d`` into an ``fma``, which rounds
    once where torch -- separate kernels -- rounds twice. With fusion on, ``dist2`` differs in
    687,501 of 4,194,304 cases; with it off, in zero.
  * ``tl.math.div_rn`` for every division. Plain ``/`` lowers to the non-IEEE ``div.full.f32``.
  * **the division is float32 and the exp is float64.** ``torch.exp((-dist2 / denom).double())``
    divides and THEN widens. Dividing in float64 instead is a different number, and it is the one
    thing in this file that looked like a Triton defect and was not.
  * **quantise between blends.** ``_alpha_blend`` ends in ``.to(uint8)`` and the reference calls it
    once per dot, so the target dot is truncated to uint8 before the agent dot composites over it.
    Chaining in float32 and quantising once at the end is the obvious implementation and measured
    35,957 disagreeing pixels against 172 on TwoRoomTarget.

The frame maximum the dot is normalised by is a REDUCTION over the whole frame, which a per-pixel
kernel cannot see. It is computed by ``_dot_max``, a first pass that evaluates the same expression
and keeps only the maximum -- so nothing is materialised there either. A maximum is exact whatever
order it is taken in, which is what makes a two-pass split safe where a sum would not be.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                    # reported by the renderer, never silent
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _dot_max_kernel(pos_ptr, den_ptr, out_ptr, H: tl.constexpr, W: tl.constexpr,
                        ND: tl.constexpr, NBLK: tl.constexpr, BLOCK: tl.constexpr):
        """Partial frame maxima of the RAW dot: ``(B, ND, NBLK)``, reduced by the caller.

        The dot is evaluated and thrown away; only the maximum leaves. Two stages rather than an
        atomic because ``amax`` over the partials is one more exact operation and needs no
        assumption about the bit pattern of a float in an atomic.
        """
        b = tl.program_id(0)
        d = tl.program_id(1)
        j = tl.program_id(2)
        px = tl.load(pos_ptr + (b * ND + d) * 2 + 0)
        py = tl.load(pos_ptr + (b * ND + d) * 2 + 1)
        den = tl.load(den_ptr + b * ND + d)
        o = j * BLOCK + tl.arange(0, BLOCK)
        m = o < H * W
        gy = (o // W).to(tl.float32)
        gx = (o % W).to(tl.float32)
        dx = gx - px
        dy = gy - py
        dist2 = dx * dx + dy * dy
        arg = tl.math.div_rn(-dist2, den)
        dot = tl.exp(arg.to(tl.float64)).to(tl.float32)
        tl.store(out_ptr + (b * ND + d) * NBLK + j,
                 tl.max(tl.where(m, dot, float("-inf"))))

    @triton.jit
    def _blend_kernel(base_ptr, pos_ptr, den_ptr, max_ptr, col_ptr, out_ptr,
                      H: tl.constexpr, W: tl.constexpr, ND: tl.constexpr, BLOCK: tl.constexpr):
        """``base`` + ND normalised Gaussian dots -> uint8, one pass, nothing materialised.

        The dot loop is ``tl.static_range`` because ND is 1 or 2 and each iteration quantises the
        running image to uint8, which is a different value to carry rather than a different address.
        """
        b = tl.program_id(0)
        o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = o < H * W
        gy = (o // W).to(tl.float32)
        gx = (o % W).to(tl.float32)
        base = base_ptr + b * H * W * 3 + o * 3
        r = tl.load(base + 0, mask=m, other=0).to(tl.float32)
        g = tl.load(base + 1, mask=m, other=0).to(tl.float32)
        bl = tl.load(base + 2, mask=m, other=0).to(tl.float32)
        for d in tl.static_range(ND):
            px = tl.load(pos_ptr + (b * ND + d) * 2 + 0)
            py = tl.load(pos_ptr + (b * ND + d) * 2 + 1)
            den = tl.load(den_ptr + b * ND + d)
            mx = tl.load(max_ptr + b * ND + d)
            cr = tl.load(col_ptr + (b * ND + d) * 3 + 0)
            cg = tl.load(col_ptr + (b * ND + d) * 3 + 1)
            cb = tl.load(col_ptr + (b * ND + d) * 3 + 2)
            dx = gx - px
            dy = gy - py
            dist2 = dx * dx + dy * dy
            arg = tl.math.div_rn(-dist2, den)
            dot = tl.exp(arg.to(tl.float64)).to(tl.float32)
            alpha = tl.where(mx > 0.0, tl.math.div_rn(dot, mx), dot)
            a = tl.minimum(tl.maximum(alpha, 0.0), 1.0)
            ia = 1.0 - a
            # .to(uint8) then back: _alpha_blend returns uint8 and the reference calls it once per
            # dot, so the image between two dots IS quantised. See the module docstring.
            r = (r * ia + cr * a).to(tl.uint8).to(tl.float32)
            g = (g * ia + cg * a).to(tl.uint8).to(tl.float32)
            bl = (bl * ia + cb * a).to(tl.uint8).to(tl.float32)
        out = out_ptr + b * H * W * 3 + o * 3
        tl.store(out + 0, r.to(tl.uint8), mask=m)
        tl.store(out + 1, g.to(tl.uint8), mask=m)
        tl.store(out + 2, bl.to(tl.uint8), mask=m)


#: Pixels per program in the blend pass. 1,024 lanes over a 224x224 frame is 49 programs a row,
#: which keeps a 3090 saturated from batch 16 up without making the launch grid the cost at batch 1.
_BLOCK = 1024


def available(device) -> bool:
    """Whether the fused path can run here: Triton importable AND the tensors on a CUDA device."""
    return HAVE_TRITON and torch.device(device).type == "cuda"


@torch.no_grad()
def blend_dots(base: torch.Tensor, pos: torch.Tensor, sigma: torch.Tensor,
               colour: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``(B,H,W,3)`` uint8 base + ``(B,ND,...)`` dots -> ``(B,H,W,3)`` uint8, composited in order.

    ``sigma`` is the radius as the env declares it; the float64 ``2*s*s`` and its single rounding to
    float32 happen here, in torch, exactly as ``_gaussian_dot`` does them, because they are per row
    rather than per pixel and there is nothing to fuse. ``colour`` is truncated here for the same
    reason. Everything that is per PIXEL is in the kernel.
    """
    if not HAVE_TRITON:
        raise ImportError("triton unavailable; use the torch path")
    B, H, W, _ = base.shape
    ND = int(pos.shape[1])
    dev = base.device
    pos = pos.contiguous().to(torch.float32)
    std = sigma.reshape(B, ND).double().clamp_min(1e-6)
    den = (2.0 * std * std).to(torch.float32).contiguous()
    col = colour.reshape(B, ND, 3).trunc().to(torch.float32).contiguous()
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=dev)
    with torch.cuda.device(dev):
        nblk = triton.cdiv(H * W, _BLOCK)
        part = torch.empty((B, ND, nblk), dtype=torch.float32, device=dev)
        _dot_max_kernel[(B, ND, nblk)](pos, den, part, H=H, W=W, ND=ND, NBLK=nblk,
                                       BLOCK=_BLOCK, num_warps=4, enable_fp_fusion=False)
        mx = part.amax(dim=2).contiguous()            # exact: a maximum has no rounding order
        _blend_kernel[(B, nblk)](base, pos, den, mx, col, out,
                                 H=H, W=W, ND=ND, BLOCK=_BLOCK, num_warps=4,
                                 enable_fp_fusion=False)
    return out
