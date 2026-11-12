"""Triton kernels for gfxdraw polygon surfaces and integer resampling, with their host-side tables."""

from __future__ import annotations

import torch

from ... import tuning
from .. import gfxdraw as G

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

PM_NV, PM_R, PM_G, PM_B, PM_X0, PM_Y0, PM_X1, PM_Y1 = range(8)
PM_W = 8
(
    EG_DRAW,
    EG_SX,
    EG_SY,
    EG_X0,
    EG_Y0,
    EG_DX,
    EG_DY,
    EG_XDIR,
    EG_KIND,
    EG_MAJOR,
    EG_E,
    EG_LO,
    EG_BX0,
    EG_BY0,
    EG_BX1,
    EG_BY1,
) = range(16)
EG_W = 16
K_GEN, K_VERT, K_HORIZ = 0, 1, 2

M32 = G.M32


@torch.no_grad()
def edge_table(vx: torch.Tensor, vy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Clip edges and prepare Wu-line constants for integer quad vertices (M, 4).

    Return (M, 4, EG_W) int32 entries for the four directed edges.
    """
    M = vx.shape[0]
    if vx.device.type == "cpu":
        from .. import cpu_raster

        if cpu_raster.available():
            from ..cpu_raster._common import edge_table_rows

            out = torch.empty((M, 4, EG_W), dtype=torch.int32)
            edge_table_rows(
                vx.to(torch.int64).contiguous().numpy(), vy.to(torch.int64).contiguous().numpy(), H, W, out.numpy()
            )
            return out
    nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
    a, b, c, d = (t.reshape(-1) for t in (vx, vy, nx, ny))
    (draw, x1, y1, x2, y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo) = G.edge_geometry(a, b, c, d, W, H)
    kind = torch.where(vert, K_VERT, torch.where(horiz, K_HORIZ, K_GEN))
    kind = torch.where((kind == K_GEN) & ymaj, K_GEN + 4, kind)

    ex0 = torch.minimum(x1, x2) - 1
    ex1 = torch.maximum(x1, x2) + 1
    ey0 = torch.minimum(y1, y2) - 1
    ey1 = torch.maximum(y1, y2) + 1
    cols = [draw.to(torch.int64), x1, y1, xx0, yy0, dx, dy, xdir, kind, major, E, lo, ex0, ey0, ex1, ey1]
    return torch.stack(cols, -1).to(torch.int32).reshape(M, 4, EG_W)


@torch.no_grad()
def bin_primitives(
    box: torch.Tensor, H: int, W: int, bin_h: int, bin_w: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return primitive lists (B, NB, K), counts (B, NB), and horizontal bin count.

    Input boxes (B, P, 4) use output coordinates (x0, y0, x1, y1).
    A stable prefix sum preserves painter order where primitives overlap.
    """
    B, P, _ = box.shape
    nby, nbx = -(-H // bin_h), -(-W // bin_w)
    dev = box.device
    by = torch.arange(nby, device=dev, dtype=torch.int32) * bin_h
    bx = torch.arange(nbx, device=dev, dtype=torch.int32) * bin_w
    oy = (box[..., 1:2] <= (by + bin_h - 1)) & (box[..., 3:4] >= by)  # (B,P,nby)
    ox = (box[..., 0:1] <= (bx + bin_w - 1)) & (box[..., 2:3] >= bx)  # (B,P,nbx)
    ov = (oy.unsqueeze(-1) & ox.unsqueeze(-2)).permute(0, 2, 3, 1).reshape(B, nby * nbx, P)
    ai = ov.to(torch.int32)
    cnt = ai.sum(-1)
    K = max(int(cnt.max().item()), 1)
    pos = ai.cumsum(-1) - ai
    slot = torch.where(ov, pos, torch.full_like(pos, K)).to(torch.int64)
    lst = torch.zeros((B, nby * nbx, K + 1), dtype=torch.int32, device=dev)
    src = torch.arange(P, dtype=torch.int32, device=dev).view(1, 1, P).expand_as(slot)
    lst.scatter_(2, slot, src)
    return lst[..., :K].contiguous(), cnt.contiguous(), nbx


if HAVE_TRITON:
    # Host layout indices stay integers; kernels read constexpr aliases of the same values.
    _c = tl.constexpr
    cPM_W, cPM_NV, cPM_R, cPM_G, cPM_B = _c(PM_W), _c(PM_NV), _c(PM_R), _c(PM_G), _c(PM_B)
    cPM_X0, cPM_Y0, cPM_X1, cPM_Y1 = _c(PM_X0), _c(PM_Y0), _c(PM_X1), _c(PM_Y1)
    cEG_W, cEG_DRAW, cEG_SX, cEG_SY = _c(EG_W), _c(EG_DRAW), _c(EG_SX), _c(EG_SY)
    cEG_X0, cEG_Y0, cEG_DX, cEG_DY = _c(EG_X0), _c(EG_Y0), _c(EG_DX), _c(EG_DY)
    cEG_XDIR, cEG_KIND, cEG_MAJOR, cEG_E, cEG_LO = (_c(EG_XDIR), _c(EG_KIND), _c(EG_MAJOR), _c(EG_E), _c(EG_LO))
    cEG_BX0, cEG_BY0, cEG_BX1, cEG_BY1 = _c(EG_BX0), _c(EG_BY0), _c(EG_BX1), _c(EG_BY1)
    cK_VERT, cK_HORIZ, cK_GEN, cK_YMAJ = _c(K_VERT), _c(K_HORIZ), _c(K_GEN), _c(K_GEN + 4)
    cM32 = _c(M32)

    @triton.jit
    def _blend(s, d, a, sh: tl.constexpr):
        s6 = s.to(tl.int64)
        d6 = d.to(tl.int64)
        diff = ((s6 - d6) << sh) & cM32
        v = ((diff * a.to(tl.int64)) & cM32) >> 8
        return ((((d6 << sh) + v) & (255 << sh)) >> sh).to(tl.int32)

    @triton.jit
    def _round16(v):
        return (v >> 16) + ((v & 32768) >> 15)

    # Invert the Wu walk per pixel; process edges in order because fringes can overlap.
    @triton.jit
    def _wu_edge(ep, X, Y, fr, fg, fb, cr, cg, cb):
        sx = tl.load(ep + cEG_SX)
        sy = tl.load(ep + cEG_SY)
        xx0 = tl.load(ep + cEG_X0)
        yy0 = tl.load(ep + cEG_Y0)
        dxe = tl.load(ep + cEG_DX)
        dye = tl.load(ep + cEG_DY)
        xdir = tl.load(ep + cEG_XDIR)
        kind = tl.load(ep + cEG_KIND)
        major = tl.load(ep + cEG_MAJOR)
        Ec = tl.load(ep + cEG_E)
        lo = tl.load(ep + cEG_LO)

        hit = (Y == sy) & (X == sx)
        if kind == cK_VERT:
            hit = hit | ((dye > 0) & (X == xx0) & (Y >= yy0) & (Y <= yy0 + dye))
        if kind == cK_HORIZ:
            hit = hit | ((Y == yy0) & (X >= lo) & (X <= lo + dxe))
        cr = tl.where(hit, fr, cr)
        cg = tl.where(hit, fg, cg)
        cb = tl.where(hit, fb, cb)

        if kind == cK_YMAJ:
            kk = Y - yy0
            acc = kk * Ec
            wraps = acc >> 16
            wgt = (acc & 0xFFFF) >> 8
            xa = xx0 + xdir * wraps
            ok = (kk >= 1) & (kk < major)
            ha = ok & (X == xa)
            hb = ok & (X == xa + xdir)
            w = tl.where(ha, 255 - wgt, wgt)
        else:
            kk = (X - xx0) * xdir
            acc = kk * Ec
            wraps = acc >> 16
            wgt = (acc & 0xFFFF) >> 8
            ya = yy0 + wraps
            ok = (kk >= 1) & (kk < major) & (kind == cK_GEN)
            ha = ok & (Y == ya)
            hb = ok & (Y == ya + 1)
            w = tl.where(ha, 255 - wgt, wgt)
        live = ha | hb
        # Full fringe weight still blends: (255 * 255) >> 8 is 254, not opaque.
        al = (255 * w) >> 8
        cr = tl.where(live, _blend(fr, cr, al, 16), cr)
        cg = tl.where(live, _blend(fg, cg, al, 8), cg)
        cb = tl.where(live, _blend(fb, cb, al, 0), cb)
        return cr, cg, cb

    @triton.jit
    def _polygon_fill(vb, nv, X, Y, fr, fg, fb, cr, cg, cb, W: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
        maxy = tl.load(vb + 1)
        for e in tl.static_range(1, 4):
            if e < nv:
                maxy = tl.maximum(maxy, tl.load(vb + 2 * e + 1))
        # Widen before 16.16 scanline products; preserve the inclusive maximum-row rule.
        Y6 = Y.to(tl.int64)
        BIG = 1 << 40
        xlo_a = tl.full([BH, BW], BIG, tl.int64)
        xhi_a = tl.full([BH, BW], -BIG, tl.int64)
        nact = tl.zeros([BH, BW], dtype=tl.int32)
        for e in tl.static_range(4):
            if e < nv:
                j = (e + nv - 1) % nv
                ax = tl.load(vb + 2 * j + 0).to(tl.int64)
                ay = tl.load(vb + 2 * j + 1).to(tl.int64)
                bxv = tl.load(vb + 2 * e + 0).to(tl.int64)
                byv = tl.load(vb + 2 * e + 1).to(tl.int64)
                sw = ay > byv
                ylo = tl.minimum(ay, byv)
                yhi = tl.maximum(ay, byv)
                xl = tl.where(sw, bxv, ax)
                xh = tl.where(sw, ax, bxv)
                act = (ay != byv) & (((Y6 >= ylo) & (Y6 < yhi)) | ((Y6 == maxy) & (Y6 > ylo) & (Y6 <= yhi)))
                den = tl.maximum(yhi - ylo, 1)
                Xv = ((65536 * (Y6 - ylo)) // den) * (xh - xl) + 65536 * xl
                xlo_a = tl.minimum(xlo_a, tl.where(act, Xv, BIG))
                xhi_a = tl.maximum(xhi_a, tl.where(act, Xv, -BIG))
                nact += act.to(tl.int32)
        xa2 = tl.maximum(_round16(xlo_a + 1), 0).to(tl.int32)
        xb2 = tl.minimum(_round16(xhi_a - 1), W - 1).to(tl.int32)
        hit = (nact >= 2) & (X >= xa2) & (X <= xb2)
        cr = tl.where(hit, fr, cr)
        cg = tl.where(hit, fg, cg)
        cb = tl.where(hit, fb, cb)
        return cr, cg, cb

    @triton.jit
    def _surface_kernel(
        pv_ptr,
        pm_ptr,
        pe_ptr,
        lst_ptr,
        cnt_ptr,
        aa_ptr,
        out_ptr,
        P,
        PG,
        PPOST,
        K,
        NB,
        H: tl.constexpr,
        W: tl.constexpr,
        VMAX: tl.constexpr,
        BINH: tl.constexpr,
        BINW: tl.constexpr,
        NBX: tl.constexpr,
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
        # Both coordinates need full tile shape for Triton conditional-value merging.
        zi = tl.zeros([BH, BW], dtype=tl.int32)
        X = zi + ox
        Yout = zi + oy
        Ypre = (H - 1) - Yout

        cr = tl.zeros([BH, BW], dtype=tl.int32)
        cg = tl.zeros([BH, BW], dtype=tl.int32)
        cb = tl.zeros([BH, BW], dtype=tl.int32)
        aa = tl.load(aa_ptr + b) != 0

        tx0 = tix * BW
        tx1 = tx0 + BW - 1
        ty0 = tiy * BH
        ty1 = ty0 + BH - 1
        bin_id = (ty0 // BINH) * NBX + (tx0 // BINW)
        n = tl.load(cnt_ptr + b * NB + bin_id)
        lbase = lst_ptr + (b.to(tl.int64) * NB + bin_id) * K

        for ii in range(0, n):
            i = tl.load(lbase + ii).to(tl.int64)
            pm = pm_ptr + (b.to(tl.int64) * P + i) * cPM_W
            bx0 = tl.load(pm + cPM_X0)
            by0 = tl.load(pm + cPM_Y0)
            bx1 = tl.load(pm + cPM_X1)
            by1 = tl.load(pm + cPM_Y1)
            if (bx1 >= tx0) & (bx0 <= tx1) & (by1 >= ty0) & (by0 <= ty1):
                nv = tl.load(pm + cPM_NV)
                fr = tl.load(pm + cPM_R)
                fg = tl.load(pm + cPM_G)
                fb = tl.load(pm + cPM_B)
                vb = pv_ptr + (b.to(tl.int64) * P + i) * (VMAX * 2)
                if i >= PPOST:
                    Y = Yout
                else:
                    Y = Ypre

                if i < PG:
                    if aa:
                        eb = pe_ptr + (b.to(tl.int64) * PG + i) * (4 * cEG_W)
                        for e in tl.static_range(4):
                            ep = eb + e * cEG_W
                            if tl.load(ep + cEG_DRAW) != 0:
                                ebx0 = tl.load(ep + cEG_BX0)
                                ebx1 = tl.load(ep + cEG_BX1)
                                eby0 = H - 1 - tl.load(ep + cEG_BY1)
                                eby1 = H - 1 - tl.load(ep + cEG_BY0)
                                if (ebx1 >= tx0) & (ebx0 <= tx1) & (eby1 >= ty0) & (eby0 <= ty1):
                                    cr, cg, cb = _wu_edge(ep, X, Y, fr, fg, fb, cr, cg, cb)

                    cr, cg, cb = _polygon_fill(vb, 4, X, Y, fr, fg, fb, cr, cg, cb, W, BH, BW)

                else:
                    miny = tl.load(vb + 1)
                    maxy = miny
                    for e in range(1, nv):
                        vyy = tl.load(vb + 2 * e + 1)
                        miny = tl.minimum(miny, vyy)
                        maxy = tl.maximum(maxy, vyy)
                    my = maxy
                    lt = tl.zeros([BH, BW], dtype=tl.int32)
                    le = tl.zeros([BH, BW], dtype=tl.int32)
                    nact = tl.zeros([BH, BW], dtype=tl.int32)
                    extra = X < 0
                    for e in range(0, nv):
                        j = (e + nv - 1) % nv
                        ax = tl.load(vb + 2 * j + 0)
                        ay = tl.load(vb + 2 * j + 1)
                        bxv = tl.load(vb + 2 * e + 0)
                        byv = tl.load(vb + 2 * e + 1)
                        up = ay < byv
                        y1 = tl.minimum(ay, byv)
                        y2 = tl.maximum(ay, byv)
                        x1 = tl.where(up, ax, bxv)
                        x2 = tl.where(up, bxv, ax)
                        act = (ay != byv) & (((y1 <= Y) & (Y < y2)) | ((Y == my) & (y2 == my)))
                        # Correctly rounded division keeps exact scanline crossings on the same pixel.
                        q = tl.math.div_rn(
                            ((Y - y1).to(tl.int64) * (x2 - x1).to(tl.int64)).to(tl.float32),
                            tl.maximum(y2 - y1, 1).to(tl.float32),
                        )
                        xc = tl.where(nact % 2 == 0, tl.floor(q), tl.ceil(q)).to(tl.int32) + x1
                        lt += (act & (xc < X)).to(tl.int32)
                        le += (act & (xc <= X)).to(tl.int32)
                        nact += act.to(tl.int32)
                        if ay == byv:
                            inside = (ay > miny) & (ay < maxy)
                            extra = extra | (
                                inside & (Y == ay) & (X >= tl.minimum(ax, bxv)) & (X <= tl.maximum(ax, bxv))
                            )
                    ka = tl.maximum(lt - 1, 0)
                    kb = tl.minimum(le - 1, nact - 2)
                    hit = (ka <= kb) & ((ka % 2 == 0) | (ka + 1 <= kb))
                    if miny == maxy:  # a quad one row high: min x to max x
                        lox = tl.load(vb + 0)
                        hix = lox
                        for e in range(1, nv):
                            vxx = tl.load(vb + 2 * e + 0)
                            lox = tl.minimum(lox, vxx)
                            hix = tl.maximum(hix, vxx)
                        hit = hit | ((Y == miny) & (X >= lox) & (X <= hix))
                    hit = hit | extra
                    cr = tl.where(hit, fr, cr)
                    cg = tl.where(hit, fg, cg)
                    cb = tl.where(hit, fb, cb)

        o = out_ptr + (b.to(tl.int64) * (H * W) + oy.to(tl.int64) * W + ox.to(tl.int64)) * 3
        tl.store(o + 0, cr.to(tl.uint8), mask=m)
        tl.store(o + 1, cg.to(tl.uint8), mask=m)
        tl.store(o + 2, cb.to(tl.uint8), mask=m)

    @triton.jit
    def _shrink_kernel(in_ptr, out_ptr, idx_ptr, w_ptr, N, MOUT, C, recip, K: tl.constexpr, BLOCK: tl.constexpr):
        a = tl.program_id(0).to(tl.int64)
        off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = off < MOUT * C
        j = off // C
        c = off % C
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        for k in tl.static_range(K):
            i = tl.load(idx_ptr + j * K + k, mask=m, other=0)
            w = tl.load(w_ptr + j * K + k, mask=m, other=0)
            v = tl.load(in_ptr + (a * N + i) * C + c, mask=m, other=0).to(tl.int32)
            acc += tl.where(w == (1 << 14), v, (v * w) >> 14)
        out = ((acc & 0xFFFF) * recip) >> 16
        out = tl.minimum(tl.maximum(out, 0), 255)
        tl.store(out_ptr + a * (MOUT * C) + off, out.to(tl.uint8), mask=m)


TILE = (1, 128)
BINS = (64, 128)


@torch.no_grad()
def render_surface(
    vert: torch.Tensor,
    meta: torch.Tensor,
    edge: torch.Tensor,
    aa: torch.Tensor,
    H: int,
    W: int,
    PG: int,
    PPOST: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render polygon lists to (B, H, W, 3) uint8 RGB, with the final vertical flip.

    Vertices are (B, P, VMAX, 2) int32; metadata is (B, P, PM_W).
    The first PG polygons are gfxdraw quads with edge tables (B, PG, 4, EG_W).
    The remainder use pygame.draw fill rules. Polygons from PPOST onward
    already use flipped coordinates. Requires CUDA and Triton.
    """
    if not vert.is_cuda:
        raise ValueError("render_surface requires CUDA; CarRacing has no CPU renderer")
    if not HAVE_TRITON:
        raise ImportError("triton unavailable")
    B, P, VMAX, _ = vert.shape
    dev = vert.device
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=dev)
    lst, cnt, nbx = bin_primitives(meta[..., cPM_X0 : cPM_Y1 + 1], H, W, *BINS)
    bh, bw = TILE
    ntx = -(-W // bw)
    with torch.cuda.device(dev):
        _surface_kernel[(B, ntx * -(-H // bh))](
            vert,
            meta,
            edge,
            lst,
            cnt,
            aa,
            out,
            P,
            PG,
            PPOST,
            lst.shape[2],
            lst.shape[1],
            H=H,
            W=W,
            VMAX=VMAX,
            BINH=BINS[0],
            BINW=BINS[1],
            NBX=nbx,
            NTX=ntx,
            BH=bh,
            BW=bw,
            num_warps=tuning.get("gfxdraw_warps"),
        )
    return out


@torch.no_grad()
def shrink(x: torch.Tensor, plan, out: torch.Tensor | None = None) -> torch.Tensor:
    """Resample one axis from (A, N, C) to (A, plan.m, C) uint8; requires CUDA and Triton."""
    if not x.is_cuda:
        raise ValueError("gfxdraw shrink requires CUDA")
    if not HAVE_TRITON:
        raise ImportError("triton unavailable")
    A, N, C = x.shape
    if out is None:
        out = torch.empty((A, plan.m, C), dtype=torch.uint8, device=x.device)
    blk = 512
    with torch.cuda.device(x.device):
        _shrink_kernel[(A, -(-(plan.m * C) // blk))](
            x, out, plan.idx32, plan.w32, N, plan.m, C, plan.recip, K=plan.K, BLOCK=blk
        )
    return out
