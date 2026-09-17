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

import math

import torch

from .raster2d import tile_shape

#: The smallest alpha ``swm_nav._alpha_blend`` can act on, MEASURED against that function over every
#: uint8 contrast (``scratch/step6/raster2d/alpha_bound.py``): 2.98e-8 = 2^-25, reached at img=1,
#: colour=0 because ``img * (1 - a)`` goes through ``1 - a``, which rounds to exactly 1.0 in float32
#: below the spacing at 1.0. Below this alpha the blend is the identity.
ALPHA_FLOOR = 2.0 ** -25

#: ``d/sigma`` at which a normalised Gaussian drops under ``ALPHA_FLOOR`` for a peak on a sample.
DOT_SIGMAS = math.sqrt(2.0 * math.log(1.0 / ALPHA_FLOOR))


def dot_window(pos: torch.Tensor, sigma: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """``(..., 4)`` int64 ``r0, r1, c0, c1`` outside which a normalised dot cannot move a pixel.

    ONE definition, used by the CPU path (``swm_nav._blend_dot``) and by the kernel below, because
    the two must agree about what "outside" means or the fast path is not the slow one.

    ``_gaussian_dot`` divides by the frame's own maximum, so the profile that reaches
    ``_alpha_blend`` is ``exp(-(d^2 - m^2) / 2 s^2)`` where ``m`` is the distance from the centre to
    the largest sample IN THE FRAME. For a centre inside the frame that is at most half a pixel in
    each axis. For a centre outside it -- which ``render(states)`` permits, since only ``step``
    clamps -- the normalisation rescales the whole dot until its largest in-frame sample is 1.0, so
    the window grows by that distance. Two consequences the kernel depends on:

      * every pixel outside the window has alpha below ``ALPHA_FLOOR``, so the blend there is the
        identity and the pixel may be skipped;
      * the window ALWAYS contains the in-frame argmax, because that sample is within ``m <= d`` of
        the centre. So a maximum taken over the window is the maximum over the frame, which is what
        lets the normalisation be computed without touching the rest of the picture.

    Plus a pixel of margin. The radius is exact for exact arithmetic, but every term in it -- the
    sqrt, the log, the squares -- is a rounding away from the pixel it decides, and the bound has no
    slack of its own.
    """
    x, y = pos[..., 0], pos[..., 1]
    mx = x.round().clamp(0, W - 1)
    my = y.round().clamp(0, H - 1)
    m2 = (x - mx) ** 2 + (y - my) ** 2
    d = torch.sqrt((sigma.reshape(x.shape) * DOT_SIGMAS) ** 2 + m2) + 1.0
    c0 = torch.ceil(x - d).clamp(0, W)
    c1 = torch.floor(x + d).clamp(0, W - 1) + 1
    r0 = torch.ceil(y - d).clamp(0, H)
    r1 = torch.floor(y + d).clamp(0, H - 1) + 1
    return torch.stack([r0, r1, c0, c1], dim=-1).to(torch.int64)

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                    # reported by the renderer, never silent
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _dot_max_kernel(pos_ptr, den_ptr, win_ptr, out_ptr, H: tl.constexpr, W: tl.constexpr,
                        ND: tl.constexpr, NTX: tl.constexpr, NT: tl.constexpr,
                        BH: tl.constexpr, BW: tl.constexpr):
        """Partial frame maxima of the RAW dot: ``(B, ND, NT)``, reduced by the caller.

        The dot is evaluated and thrown away; only the maximum leaves. Two stages rather than an
        atomic because ``amax`` over the partials is one more exact operation and needs no
        assumption about the bit pattern of a float in an atomic.

        A TILE whose bounds miss the dot's window is skipped outright, and that is exact rather than
        approximate: ``dot_window`` contains the in-frame argmax by construction, so the maximum over
        the window IS the maximum over the frame. The skip is a scalar branch, so what it saves is
        the whole per-pixel body -- and the body is a float64 ``exp``, which a 3090 issues at 1/64
        the float32 rate.
        """
        b = tl.program_id(0)
        d = tl.program_id(1)
        t = tl.program_id(2)
        r0 = (t // NTX) * BH
        c0 = (t % NTX) * BW
        wb = win_ptr + (b * ND + d) * 4
        if (r0 < tl.load(wb + 1)) & (r0 + BH > tl.load(wb + 0)) & \
           (c0 < tl.load(wb + 3)) & (c0 + BW > tl.load(wb + 2)):
            px = tl.load(pos_ptr + (b * ND + d) * 2 + 0)
            py = tl.load(pos_ptr + (b * ND + d) * 2 + 1)
            den = tl.load(den_ptr + b * ND + d)
            rr = r0 + tl.arange(0, BH)[:, None]
            cc = c0 + tl.arange(0, BW)[None, :]
            m = (rr < H) & (cc < W)
            dx = cc.to(tl.float32) - px
            dy = rr.to(tl.float32) - py
            dist2 = dx * dx + dy * dy
            arg = tl.math.div_rn(-dist2, den)
            dot = tl.exp(arg.to(tl.float64)).to(tl.float32)
            tl.store(out_ptr + (b * ND + d) * NT + t,
                     tl.max(tl.where(m, dot, float("-inf"))))
        else:
            tl.store(out_ptr + (b * ND + d) * NT + t, float("-inf"))

    @triton.jit
    def _blend_kernel(base_ptr, pos_ptr, den_ptr, max_ptr, col_ptr, win_ptr, out_ptr,
                      H: tl.constexpr, W: tl.constexpr, ND: tl.constexpr,
                      NTX: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
        """``base`` + ND normalised Gaussian dots -> uint8, one pass, nothing materialised.

        The dot loop is ``tl.static_range`` because ND is 1 or 2 and each iteration quantises the
        running image to uint8, which is a different value to carry rather than a different address.

        A dot whose window misses this tile is skipped, and skipping it is the identity rather than
        an approximation: outside the window the alpha is below ``ALPHA_FLOOR``, where ``1 - a``
        rounds to exactly 1.0 and the truncation to uint8 absorbs ``colour * a``. The tile is still
        WRITTEN -- the output has to carry the base image whether a dot reaches it or not -- so what
        the branch saves is the float64 ``exp``, not the store.
        """
        b = tl.program_id(0)
        t = tl.program_id(1)
        r0 = (t // NTX) * BH
        c0 = (t % NTX) * BW
        rr = r0 + tl.arange(0, BH)[:, None]
        cc = c0 + tl.arange(0, BW)[None, :]
        m = (rr < H) & (cc < W)
        o = rr * W + cc
        base = base_ptr + b * H * W * 3 + o * 3
        r = tl.load(base + 0, mask=m, other=0).to(tl.float32)
        g = tl.load(base + 1, mask=m, other=0).to(tl.float32)
        bl = tl.load(base + 2, mask=m, other=0).to(tl.float32)
        gx = cc.to(tl.float32)
        gy = rr.to(tl.float32)
        for d in tl.static_range(ND):
            wb = win_ptr + (b * ND + d) * 4
            if (r0 < tl.load(wb + 1)) & (r0 + BH > tl.load(wb + 0)) & \
               (c0 < tl.load(wb + 3)) & (c0 + BW > tl.load(wb + 2)):
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
                # .to(uint8) then back: _alpha_blend returns uint8 and the reference calls it once
                # per dot, so the image between two dots IS quantised. See the module docstring.
                r = (r * ia + cr * a).to(tl.uint8).to(tl.float32)
                g = (g * ia + cg * a).to(tl.uint8).to(tl.float32)
                bl = (bl * ia + cb * a).to(tl.uint8).to(tl.float32)
        out = out_ptr + b * H * W * 3 + o * 3
        tl.store(out + 0, r.to(tl.uint8), mask=m)
        tl.store(out + 1, g.to(tl.uint8), mask=m)
        tl.store(out + 2, bl.to(tl.uint8), mask=m)


#: Lanes per program. A 2-D TILE rather than a run of consecutive pixels, for the reason
#: ``raster2d`` records: both cover the same number of lanes and only the rectangle has a compact
#: footprint in the plane, which is the whole point when the loop is skipping dots that miss it. A
#: 1x1024 strip of a 224-wide frame spans five rows and every column, and is missed by almost
#: nothing.
_BLOCK = 1024


def _tile(H: int, W: int) -> tuple[int, int, int, int]:
    """``(BH, BW, tiles across, tiles total)``: the widest power-of-two tile that wastes few lanes.

    Wide at equal area, because the output is ``(B,H,W,3)`` uint8 and the three stores are strided
    by 3, so a narrow tile scatters its writes. At 224 px this is 32x32 and divides exactly.

    The shape itself is ``raster2d.tile_shape``: this kernel and the scene rasterizer pick their
    tile by the same law (widest power of two that wastes at most ``TILE_WASTE_MAX`` of a row of
    lanes, then whatever height the lanes allow), and it was written out twice. Only the width cap
    differs -- unbounded here, 128 there -- so only the width cap is an argument.
    """
    bh, bw = tile_shape(H, W, lanes=_BLOCK, max_width=_BLOCK)
    ntx = -(-W // bw)
    return bh, bw, ntx, ntx * (-(-H // bh))


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
    win = dot_window(pos, sigma.reshape(B, ND), H, W).reshape(B, ND, 4).contiguous()
    bh, bw, ntx, nt = _tile(H, W)
    with torch.cuda.device(dev):
        part = torch.empty((B, ND, nt), dtype=torch.float32, device=dev)
        _dot_max_kernel[(B, ND, nt)](pos, den, win, part, H=H, W=W, ND=ND, NTX=ntx, NT=nt,
                                     BH=bh, BW=bw, num_warps=4, enable_fp_fusion=False)
        mx = part.amax(dim=2).contiguous()            # exact: a maximum has no rounding order
        _blend_kernel[(B, nt)](base, pos, den, mx, col, win, out,
                               H=H, W=W, ND=ND, NTX=ntx, BH=bh, BW=bw, num_warps=4,
                               enable_fp_fusion=False)
    return out
