from __future__ import annotations

import numpy as np
import torch

from . import gfxdraw as G
from .gfxdraw_fused import (
    HAVE_TRITON,
    edge_table,
)

if HAVE_TRITON:
    import triton
    import triton.language as tl

OP_POLY, OP_DISC, OP_HLINE, OP_VLINE, OP_BLIT = range(5)

(CM_OP, CM_R, CM_G, CM_B, CM_X0, CM_Y0, CM_X1, CM_Y1, CM_A, CM_B2, CM_C, CM_D) = range(12)
CM_W = 12

SOLID = G.SOLID


def disc_table(r: int) -> np.ndarray:
    """Return integer disc spans and two ordered fringe-weight maps for radius r.

    Layout: r+1 row half-widths, followed by two flattened (2r+1)² weight maps.
    Duplicate writes within a pass or unsupported fringe weights are rejected."""
    r = int(r)
    d = 2 * r + 1
    hw = np.asarray(G.circle_hw(r), np.int32)
    w = np.zeros((2, d, d), np.int32)
    # Multiple passes may revisit a pixel, but each pass needs distinct offsets.
    seen: dict[tuple[int, int, int], int] = {}
    for dy, dx, wt, pas in G.AACIRCLE_FRINGE[r]:
        key = (pas, dy, dx)
        assert key not in seen, (
            f"aacircle fringe for r={r} has two pass-{pas} writes at (dy={dy}, dx={dx}) "
            f"(weights {seen[key]} and {wt}); a dense per-pass map can only hold one, so the "
            f"fused path would drop the second while the per-primitive path makes both"
        )
        seen[key] = wt
    for dy, dx, wt, pas in G.AACIRCLE_FRINGE[r]:
        assert 0 < wt < SOLID, f"aacircle fringe weight {wt} at r={r} is outside (0, {SOLID})"
        w[pas, dy + r, dx + r] = wt
    return np.concatenate([hw, w[0].reshape(-1), w[1].reshape(-1)])


class DiscTables:
    """Concatenated disc lookup tables and the offset for each radius."""

    def __init__(self, radii, device):
        self.offset: dict[int, int] = {}
        blocks, at = [], 0
        for r in sorted({int(x) for x in radii}):
            self.offset[r] = at
            b = disc_table(r)
            blocks.append(b)
            at += len(b)
        self.flat = torch.as_tensor(
            np.concatenate(blocks) if blocks else np.zeros(1, np.int32), dtype=torch.int32, device=device
        )


class Ops:
    """Batched classic-control draw operations in painter order.

    Geometry and colors vary per environment; canvas dimensions and disc radii
    are fixed. Call finish() before rendering."""

    def __init__(self, B, H, W, device):
        self.B, self.H, self.W, self.device = B, H, W, device
        self.meta: list[torch.Tensor] = []
        self.verts: list[torch.Tensor] = []
        self.bg = None
        self.plate = None
        self.sprite = None
        self.aa = None
        self.discs = None

    def _row(self, op, color, x0, y0, x1, y1, a=0, b=0, c=0, d=0):
        B, one = self.B, torch.ones(self.B, dtype=torch.int32, device=self.device)
        col = G.to_u8(color).to(torch.int32)

        def f(v):
            return (v.to(torch.int32) if torch.is_tensor(v) else one * int(v)).reshape(B)

        yo0 = (self.H - 1) - f(y1)
        yo1 = (self.H - 1) - f(y0)
        self.meta.append(
            torch.stack(
                [
                    one * int(op),
                    col[:, 0],
                    col[:, 1],
                    col[:, 2],
                    f(x0).clamp(0, self.W - 1),
                    yo0.clamp(0, self.H - 1),
                    f(x1).clamp(0, self.W - 1),
                    yo1.clamp(0, self.H - 1),
                    f(a),
                    f(b),
                    f(c),
                    f(d),
                ],
                -1,
            ).to(torch.int32)
        )

    def polygon(self, verts, color):
        """Add a triangle or quad: antialiased fringe followed by opaque scanline fill."""
        vi = torch.trunc(verts).to(torch.int64)  # (B, n, 2)
        n = int(vi.shape[1])
        if n not in (3, 4):
            raise ValueError("classic gfxdraw polygons require 3 or 4 vertices")
        pad = vi if n == 4 else torch.cat([vi, vi[:, :1]], 1)  # [v0..v_{n-1}, v0]
        slot = len(self.verts)
        self.verts.append(pad[:, :4])
        vx, vy = pad[..., 0], pad[..., 1]
        self._row(
            OP_POLY,
            color,
            vx[:, :n].amin(1) - 1,
            vy[:, :n].amin(1) - 1,
            vx[:, :n].amax(1) + 1,
            vy[:, :n].amax(1) + 1,
            a=n,
            b=slot,
        )

    def disc(self, cx, cy, r, color):
        """Add an antialiased disc with a fixed integer radius."""
        r = int(r)
        self._row(
            OP_DISC, color, cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1, a=cx, b=cy, c=self.discs.offset[r], d=r
        )

    def hline(self, y, color):
        self._row(OP_HLINE, color, 0, y, self.W - 1, y, a=y)

    def vline(self, x, y0, y1, color):
        self._row(OP_VLINE, color, x, y0, x, y1, a=x, b=y0, c=y1)

    def blit(self, origin, win, flip, n, color):
        """Add a pre-scaled RGBA sprite using SDL surface-blit arithmetic."""
        self._row(OP_BLIT, color, origin, origin, origin + win - 1, origin + win - 1, a=flip, b=n, c=origin, d=win)

    def finish(self):
        self.meta_t = torch.stack(self.meta, 1).contiguous()  # (B,P,CM_W)
        self.vert_t = (
            torch.stack(self.verts, 1).to(torch.int32).contiguous()
            if self.verts
            else torch.zeros((self.B, 1, 4, 2), dtype=torch.int32, device=self.device)
        )  # (B,PG,4,2)
        PG = self.vert_t.shape[1]
        ex = self.vert_t[..., 0].reshape(-1, 4).to(torch.int64)
        ey = self.vert_t[..., 1].reshape(-1, 4).to(torch.int64)
        self.edge_t = edge_table(ex, ey, self.H, self.W).reshape(self.B, PG, 4, -1).contiguous()
        return self


if HAVE_TRITON:
    from .gfxdraw_fused import (
        _blend,
        _polygon_fill,
        _wu_edge,
        cEG_DRAW,
        cEG_W,
    )

    _c = tl.constexpr
    cCM_W, cCM_OP, cCM_R, cCM_G, cCM_B = _c(CM_W), _c(CM_OP), _c(CM_R), _c(CM_G), _c(CM_B)
    cCM_X0, cCM_Y0, cCM_X1, cCM_Y1 = _c(CM_X0), _c(CM_Y0), _c(CM_X1), _c(CM_Y1)
    cCM_A, cCM_B2, cCM_C, cCM_D = _c(CM_A), _c(CM_B2), _c(CM_C), _c(CM_D)
    cOP_POLY, cOP_DISC, cOP_HLINE, cOP_VLINE, cOP_BLIT = (
        _c(OP_POLY),
        _c(OP_DISC),
        _c(OP_HLINE),
        _c(OP_VLINE),
        _c(OP_BLIT),
    )

    @triton.jit
    def _classic_kernel(
        meta_ptr,
        vert_ptr,
        edge_ptr,
        disc_ptr,
        bg_ptr,
        plate_ptr,
        sprite_ptr,
        aa_ptr,
        out_ptr,
        P,
        PG,
        H: tl.constexpr,
        W: tl.constexpr,
        PLATE_B: tl.constexpr,
        HAS_PLATE: tl.constexpr,
        HAS_SPRITE: tl.constexpr,
        SPR_N: tl.constexpr,
        SPR_S: tl.constexpr,
        NTX: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
    ):
        b = tl.program_id(0)
        tid = tl.program_id(1)
        tiy = tid // NTX
        tix = tid % NTX
        oy = tiy * BH + tl.arange(0, BH)[:, None]
        ox = tix * BW + tl.arange(0, BW)[None, :]
        m = (oy < H) & (ox < W)
        b64 = b.to(tl.int64)

        zi = tl.zeros([BH, BW], dtype=tl.int32)
        X = zi + ox
        Yout = zi + oy
        Y = (H - 1) - Yout

        aa = tl.load(aa_ptr + b) != 0
        tx0 = tix * BW
        tx1 = tx0 + BW - 1
        ty0 = tiy * BH
        ty1 = ty0 + BH - 1

        if HAS_PLATE:
            pb = b64 % PLATE_B
            po = (pb * H + Y.to(tl.int64)) * W + X.to(tl.int64)
            pl = H * PLATE_B * W
            cr = tl.load(plate_ptr + po, mask=m, other=0).to(tl.int32)
            cg = tl.load(plate_ptr + pl + po, mask=m, other=0).to(tl.int32)
            cb = tl.load(plate_ptr + 2 * pl + po, mask=m, other=0).to(tl.int32)
        else:
            cr = zi + tl.load(bg_ptr + b * 3 + 0)
            cg = zi + tl.load(bg_ptr + b * 3 + 1)
            cb = zi + tl.load(bg_ptr + b * 3 + 2)

        for i in range(0, P):
            pm = meta_ptr + (b64 * P + i) * cCM_W
            bx0 = tl.load(pm + cCM_X0)
            by0 = tl.load(pm + cCM_Y0)
            bx1 = tl.load(pm + cCM_X1)
            by1 = tl.load(pm + cCM_Y1)
            if (bx1 >= tx0) & (bx0 <= tx1) & (by1 >= ty0) & (by0 <= ty1):
                op = tl.load(pm + cCM_OP)
                fr = tl.load(pm + cCM_R)
                fg = tl.load(pm + cCM_G)
                fb = tl.load(pm + cCM_B)
                pa = tl.load(pm + cCM_A)
                pb2 = tl.load(pm + cCM_B2)
                pc = tl.load(pm + cCM_C)
                pd = tl.load(pm + cCM_D)

                if op == cOP_HLINE:
                    hit = Y == pa
                    cr = tl.where(hit, fr, cr)
                    cg = tl.where(hit, fg, cg)
                    cb = tl.where(hit, fb, cb)

                if op == cOP_VLINE:
                    hit = (X == pa) & (Y >= pb2) & (Y <= pc)
                    cr = tl.where(hit, fr, cr)
                    cg = tl.where(hit, fg, cg)
                    cb = tl.where(hit, fb, cb)

                if op == cOP_BLIT:
                    if HAS_SPRITE:
                        off = (pd // 2) - (pb2 // 2)
                        ly = (Y - pc) - off
                        lx = (X - pc) - off
                        ok = (ly >= 0) & (ly < pb2) & (lx >= 0) & (lx < pb2)
                        lyc = tl.minimum(tl.maximum(ly, 0), SPR_S - 1).to(tl.int64)
                        lxc = tl.minimum(tl.maximum(lx, 0), SPR_S - 1).to(tl.int64)
                        so = ((((pa.to(tl.int64) * SPR_N + pb2.to(tl.int64)) * SPR_S + lyc) * SPR_S) + lxc) * 4
                        sa = tl.load(sprite_ptr + so + 3, mask=m, other=0).to(tl.int32)
                        sa = tl.where(ok, sa, 0)
                        sr = tl.load(sprite_ptr + so + 0, mask=m, other=0).to(tl.int32)
                        sg = tl.load(sprite_ptr + so + 1, mask=m, other=0).to(tl.int32)
                        sb = tl.load(sprite_ptr + so + 2, mask=m, other=0).to(tl.int32)
                        # SDL surface blits use a signed difference, unlike packed SDL_gfx blending.
                        cr = cr + (((sr - cr) * sa) >> 8)
                        cg = cg + (((sg - cg) * sa) >> 8)
                        cb = cb + (((sb - cb) * sa) >> 8)

                if op == cOP_DISC:
                    r = pd
                    d = 2 * r + 1
                    dy = Y - pb2
                    dx = X - pa
                    ady = tl.where(dy < 0, -dy, dy)
                    adx = tl.where(dx < 0, -dx, dx)
                    near = (ady <= r) & (adx <= r)
                    iy = tl.minimum(tl.maximum(dy + r, 0), d - 1)
                    ix = tl.minimum(tl.maximum(dx + r, 0), d - 1)
                    wo = (pc + r + 1 + iy * d + ix).to(tl.int64)
                    if aa:
                        # Preserve both fringe writes in order before the opaque disc fill.
                        w0 = tl.where(near, tl.load(disc_ptr + wo, mask=m, other=0), 0)
                        l0 = w0 > 0
                        a0 = (255 * w0) >> 8
                        cr = tl.where(l0, _blend(fr, cr, a0, 16), cr)
                        cg = tl.where(l0, _blend(fg, cg, a0, 8), cg)
                        cb = tl.where(l0, _blend(fb, cb, a0, 0), cb)
                        w1 = tl.where(near, tl.load(disc_ptr + wo + d * d, mask=m, other=0), 0)
                        l1 = w1 > 0
                        a1 = (255 * w1) >> 8
                        cr = tl.where(l1, _blend(fr, cr, a1, 16), cr)
                        cg = tl.where(l1, _blend(fg, cg, a1, 8), cg)
                        cb = tl.where(l1, _blend(fb, cb, a1, 0), cb)
                    hw = tl.load(disc_ptr + (pc + tl.minimum(ady, r)).to(tl.int64), mask=m, other=0)
                    hit = (ady <= r) & (adx <= hw)
                    cr = tl.where(hit, fr, cr)
                    cg = tl.where(hit, fg, cg)
                    cb = tl.where(hit, fb, cb)

                if op == cOP_POLY:
                    nv = pa
                    vb = vert_ptr + (b64 * PG + pb2.to(tl.int64)) * 8
                    if aa:
                        eb = edge_ptr + (b64 * PG + pb2.to(tl.int64)) * (4 * cEG_W)
                        for e in tl.static_range(4):
                            if e < nv:
                                ep = eb + e * cEG_W
                                if tl.load(ep + cEG_DRAW) != 0:
                                    cr, cg, cb = _wu_edge(ep, X, Y, fr, fg, fb, cr, cg, cb)

                    cr, cg, cb = _polygon_fill(vb, nv, X, Y, fr, fg, fb, cr, cg, cb, W, BH, BW)

        o = out_ptr + (b64 * (H * W) + oy.to(tl.int64) * W + ox.to(tl.int64)) * 3
        tl.store(o + 0, cr.to(tl.uint8), mask=m)
        tl.store(o + 1, cg.to(tl.uint8), mask=m)
        tl.store(o + 2, cb.to(tl.uint8), mask=m)


TILE = (1, 128)
WARPS = 1


@torch.no_grad()
def render_classic(ops: Ops, out: torch.Tensor | None = None) -> torch.Tensor:
    """Render operations to (B, H, W, 3) uint8 RGB, with the final vertical flip.

    Requires CUDA and Triton. A supplied output buffer is overwritten and returned."""
    if torch.device(ops.device).type != "cuda":
        raise ValueError("render_classic requires CUDA; use the environment's Torch renderer on CPU")
    if not HAVE_TRITON:
        raise ImportError("triton unavailable")
    B, H, W, dev = ops.B, ops.H, ops.W, ops.device
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=dev)
    P = int(ops.meta_t.shape[1])
    PG = int(ops.vert_t.shape[1])
    plate = ops.plate
    spr = ops.sprite
    bh, bw = TILE
    ntx = -(-W // bw)
    empty8 = torch.empty(1, dtype=torch.uint8, device=dev)
    with torch.cuda.device(dev):
        _classic_kernel[(B, ntx * -(-H // bh))](
            ops.meta_t,
            ops.vert_t,
            ops.edge_t,
            ops.discs.flat,
            ops.bg if ops.bg is not None else torch.zeros((B, 3), dtype=torch.int32, device=dev),
            plate.contiguous() if plate is not None else empty8,
            spr.contiguous() if spr is not None else empty8,
            ops.aa,
            out,
            P,
            PG,
            H=H,
            W=W,
            PLATE_B=int(plate.shape[1]) if plate is not None else 1,
            HAS_PLATE=plate is not None,
            HAS_SPRITE=spr is not None,
            SPR_N=int(spr.shape[1]) if spr is not None else 1,
            SPR_S=int(spr.shape[2]) if spr is not None else 1,
            NTX=ntx,
            BH=bh,
            BW=bw,
            num_warps=WARPS,
        )
    return out
