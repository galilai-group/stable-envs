"""Single-pass tile blit: one gather per output word, no intermediate tiled buffer.

The torch reference does ``atlas[idx]`` (a coalesced gather into a (B,H,W,T,T,3)
buffer) and then a permute+reshape, which is a second full-size transpose copy.
This kernel writes the final (B, H*T, W*T, 3) layout directly.

Both source and destination are addressed in words of the widest size that divides a
tile row (T*3 bytes) -- 4 for the usual T=32, but rjepa renders MiniGrid at T=14, where
T*3 = 42 and the widest safe word is 2. Every offset the kernel forms is a multiple of
a tile row (atlas: T*T*3; output rows: W*T*3), so word alignment holds on both sides.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _blit_kernel(
    atlas_ptr, idx_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, T: tl.constexpr,
    WORDS_PER_TILE_ROW: tl.constexpr,   # T*3//4
    WORDS_PER_ROW: tl.constexpr,        # W*T*3//4
    WORDS_PER_TILE: tl.constexpr,       # T*T*3//4
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)              # flattened (b, y)
    blk = tl.program_id(1)

    b = row // (H * T)
    y = row % (H * T)
    tj = y // T
    py = y % T

    w = blk * BLOCK + tl.arange(0, BLOCK)
    m = w < WORDS_PER_ROW

    ti = w // WORDS_PER_TILE_ROW
    wi = w % WORDS_PER_TILE_ROW

    tile = tl.load(idx_ptr + b * (H * W) + tj * W + ti, mask=m, other=0)
    src = tile.to(tl.int64) * WORDS_PER_TILE + py * WORDS_PER_TILE_ROW + wi
    val = tl.load(atlas_ptr + src, mask=m, other=0)
    tl.store(out_ptr + row.to(tl.int64) * WORDS_PER_ROW + w, val, mask=m)


def word_size(T: int) -> int:
    """Widest word (bytes) that evenly divides a tile row of T*3 bytes."""
    row = T * 3
    return 4 if row % 4 == 0 else (2 if row % 2 == 0 else 1)


_WORD_DTYPE = {4: torch.int32, 2: torch.int16, 1: torch.int8}


def blit(atlas_w: torch.Tensor, idx: torch.Tensor, T: int, out=None):
    """atlas_w: (N, T*T*3//w) word view of the atlas; idx: (B,H,W) int32
    -> (B, H*T, W*T, 3) uint8."""
    B, H, W = idx.shape
    w = word_size(T)
    wpr = W * T * 3 // w
    if out is None:
        out = torch.empty(B, H * T, W * T, 3, dtype=torch.uint8, device=idx.device)
    out_w = out.view(-1).view(_WORD_DTYPE[w])
    idx = idx.contiguous()

    BLOCK = 256 if wpr >= 256 else triton.next_power_of_2(wpr)
    grid = (B * H * T, triton.cdiv(wpr, BLOCK))
    _blit_kernel[grid](
        atlas_w, idx, out_w,
        H=H, W=W, T=T,
        WORDS_PER_TILE_ROW=T * 3 // w,
        WORDS_PER_ROW=wpr,
        WORDS_PER_TILE=T * T * 3 // w,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def _vis_kernel(t_ptr, m_ptr, n, V: tl.constexpr, BLOCK: tl.constexpr):
    """Grid.process_vis as a per-environment 64-bit bitmask scan.

    The visibility sweep is genuinely sequential -- each cell reads what the same
    sweep just wrote -- so the torch transcription costs ~670 elementwise kernels for
    V=7. Here the whole V*V mask lives in one int64 register, bit k = (i*V + j), and
    the 2*V*(V-1) steps unroll inside a single launch.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = off < n
    t = tl.load(t_ptr + off, mask=ok, other=0)
    m = tl.zeros((BLOCK,), tl.int64) + (1 << ((V // 2) * V + (V - 1)))
    for j in range(V - 1, -1, -1):
        for i in range(0, V - 1):
            k = i * V + j
            c = (m >> k) & (t >> k) & 1
            m = m | (c << ((i + 1) * V + j))
            if j > 0:
                m = m | (c << ((i + 1) * V + j - 1))
                m = m | (c << (i * V + j - 1))
        for i in range(V - 1, 0, -1):
            k = i * V + j
            c = (m >> k) & (t >> k) & 1
            m = m | (c << ((i - 1) * V + j))
            if j > 0:
                m = m | (c << ((i - 1) * V + j - 1))
                m = m | (c << (i * V + j - 1))
    tl.store(m_ptr + off, m, mask=ok)


def process_vis(transp_bits: torch.Tensor, V: int) -> torch.Tensor:
    """transp_bits: (B,) int64 packed see_behind() mask -> (B,) int64 visibility mask."""
    n = transp_bits.numel()
    out = torch.empty_like(transp_bits)
    BLOCK = 128
    _vis_kernel[(triton.cdiv(n, BLOCK),)](transp_bits, out, n, V=V, BLOCK=BLOCK,
                                          num_warps=4)
    return out
