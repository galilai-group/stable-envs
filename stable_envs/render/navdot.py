from __future__ import annotations

import math

import torch

from .raster2d import tile_shape

# Below this alpha, float32 subtraction and uint8 truncation leave every contrast unchanged.
ALPHA_FLOOR = 2.0**-25

# Gaussian cutoff in sigma units for a peak exactly on a pixel sample.
DOT_SIGMAS = math.sqrt(2.0 * math.log(1.0 / ALPHA_FLOOR))


def dot_window(pos: torch.Tensor, sigma: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Return int64 (..., 4) windows (r0, r1, c0, c1) for normalized Gaussian dots.

    Include the in-frame maximum even for off-frame centers. Pixels outside
    have alpha below ALPHA_FLOOR; add one pixel of slack for bound rounding.
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
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _dot_max_kernel(
        pos_ptr,
        den_ptr,
        win_ptr,
        out_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        ND: tl.constexpr,
        NTX: tl.constexpr,
        NT: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
    ):
        """Write per-tile raw Gaussian maxima; window culling must retain the frame maximum."""
        b = tl.program_id(0)
        d = tl.program_id(1)
        t = tl.program_id(2)
        r0 = (t // NTX) * BH
        c0 = (t % NTX) * BW
        wb = win_ptr + (b * ND + d) * 4
        if (r0 < tl.load(wb + 1)) & (r0 + BH > tl.load(wb + 0)) & (c0 < tl.load(wb + 3)) & (c0 + BW > tl.load(wb + 2)):
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
            tl.store(out_ptr + (b * ND + d) * NT + t, tl.max(tl.where(m, dot, float("-inf"))))
        else:
            tl.store(out_ptr + (b * ND + d) * NT + t, float("-inf"))

    @triton.jit
    def _blend_kernel(
        base_ptr,
        pos_ptr,
        den_ptr,
        max_ptr,
        col_ptr,
        win_ptr,
        out_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        ND: tl.constexpr,
        NTX: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
    ):
        """Blend ordered normalized dots into uint8 output, quantizing after each dot.

        Tiles outside a dot window skip its exponential but still write the base image.
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
            if (
                (r0 < tl.load(wb + 1))
                & (r0 + BH > tl.load(wb + 0))
                & (c0 < tl.load(wb + 3))
                & (c0 + BW > tl.load(wb + 2))
            ):
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
                # Quantize between dots, matching a separate uint8 alpha-blend call for each.
                r = (r * ia + cr * a).to(tl.uint8).to(tl.float32)
                g = (g * ia + cg * a).to(tl.uint8).to(tl.float32)
                bl = (bl * ia + cb * a).to(tl.uint8).to(tl.float32)
        out = out_ptr + b * H * W * 3 + o * 3
        tl.store(out + 0, r.to(tl.uint8), mask=m)
        tl.store(out + 1, g.to(tl.uint8), mask=m)
        tl.store(out + 2, bl.to(tl.uint8), mask=m)


# Rectangular tiles permit spatial culling that long flattened strips cannot.
_BLOCK = 1024


def _tile(H: int, W: int) -> tuple[int, int, int, int]:
    """Choose a compact power-of-two tile and return height, width, tiles across, and total."""
    bh, bw = tile_shape(H, W, lanes=_BLOCK, max_width=_BLOCK)
    ntx = -(-W // bw)
    return bh, bw, ntx, ntx * (-(-H // bh))


def available(device) -> bool:
    """Return whether Triton is importable and the requested device is CUDA."""
    return HAVE_TRITON and torch.device(device).type == "cuda"


@torch.no_grad()
def blend_dots(
    base: torch.Tensor, pos: torch.Tensor, sigma: torch.Tensor, colour: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Composite dots in order onto a CUDA uint8 base (B, H, W, 3).

    Positions are (B, ND, 2), sigmas (B, ND), and colors (B, ND, 3). Form the
    denominator in float64 before rounding to float32 and truncate colors to
    match the torch path. Write into out when supplied; otherwise allocate it.
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
        _dot_max_kernel[(B, ND, nt)](
            pos, den, win, part, H=H, W=W, ND=ND, NTX=ntx, NT=nt, BH=bh, BW=bw, num_warps=4, enable_fp_fusion=False
        )
        mx = part.amax(dim=2).contiguous()  # exact: a maximum has no rounding order
        _blend_kernel[(B, nt)](
            base,
            pos,
            den,
            mx,
            col,
            win,
            out,
            H=H,
            W=W,
            ND=ND,
            NTX=ntx,
            BH=bh,
            BW=bw,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
