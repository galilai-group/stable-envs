"""``gfxdraw.py``'s rules, per PIXEL and in one kernel, for scenes that are a long list of polygons.

Why this exists next to ``gfxdraw.py``. That module is one batched torch op per primitive, which is
the right shape when a frame is a dozen primitives: classic control draws 5-30 and the dispatch is
lost in the noise. CarRacing draws ~200 on a 1000x800 surface, and there the shape is fatal in two
independent ways, both measured:

  * **dispatch.** A frame took 3,575 ms at batch 64 and 3,814 ms at batch 256 -- the SAME time for
    four times the work, i.e. 95% of it was python issuing ~110,000 small CUDA launches. 97.8% of
    that sat in ``polygon``, at 34.6 ms per polygon, most of it ``clip_line``'s six Cohen-Sutherland
    iterations and the Wu walk's fixed ``KMAX = 1024`` step bound, both per EDGE.
  * **memory.** ``Canvas.paint`` rewrites the whole ``(B,H,W,3)`` buffer per primitive because a
    data-dependent window would need a host sync, and at batch 1024 that is 10.7 GB of traffic per
    polygon, 1.1 TB per frame.

Neither is a property of the RULES, which are integer arithmetic on integer geometry and evaluate
per pixel in closed form. So the rules move into one kernel that walks the primitives back to front
in registers, and both costs go away together: one launch, and the only traffic is the surface
written once.

THREE THINGS MAKE THE CLOSED FORM WORK, and each is the reason a step that looks sequential is not:

1. **The Wu walk inverts.** ``aalineColor`` accumulates ``E`` over ``k`` steps along the major axis,
   which reads as a loop, but ``k`` is recoverable from the pixel: on a y-major edge ``k = y - y0``,
   so the two writes that land on a given row are ``x0 + xdir*(k*E >> 16)`` at weight ``255 - w`` and
   the next column over at ``w``. A pixel therefore asks the edge one question instead of the edge
   visiting every pixel.
2. **An edge writes each pixel at most once.** The solid initial pixel is at ``k = 0`` (or ``k =
   major`` when ``clip_line`` swapped the endpoints), and the Wu pairs run ``1 <= k < major``, so
   they cannot collide; and within a pair the two writes differ in the major coordinate. That is what
   lets a pixel take ONE weight per edge rather than replaying an ordered list of writes.
3. **Order is the loop.** Painter's order across primitives, and within a primitive the fringe blends
   before the fill overwrites it, are both just the order of the register updates.

Everything else is a bound. The primitive list is binned into screen tiles once per frame so a tile
walks ~7 primitives instead of ~200; a tile that a primitive's bounding box misses cannot show it,
and an edge whose bounding box misses the tile cannot blend into it. Both are exact.

``tests/gpu/test_gfxdraw_fused.py`` diffs this against ``gfxdraw.py``'s per-primitive path on random
polygons rather than reasoning about the equivalence, and CarRacing's own parity test then diffs the
whole pipeline against pygame.
"""
from __future__ import annotations

import torch

from . import gfxdraw as G

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

#: primitive metadata columns
PM_NV, PM_R, PM_G, PM_B, PM_X0, PM_Y0, PM_X1, PM_Y1 = range(8)
PM_W = 8
#: per-edge columns of the Wu/clip pre-pass
(EG_DRAW, EG_SX, EG_SY, EG_X0, EG_Y0, EG_DX, EG_DY, EG_XDIR, EG_KIND, EG_MAJOR, EG_E, EG_LO,
 EG_BX0, EG_BY0, EG_BX1, EG_BY1) = range(16)
EG_W = 16
#: ``kind``: which of ``aalineColor``'s three cases this edge takes
K_GEN, K_VERT, K_HORIZ = 0, 1, 2

#: mask for the unsigned-32 arithmetic SDL_gfx's blend relies on. ``gfxdraw``'s, not a second copy:
#: ``_blend`` below is that module's ``blend`` in Triton and the two must mask at the same width.
M32 = G.M32


# ------------------------------------------------------------------ host-side pre-pass
@torch.no_grad()
def edge_table(vx: torch.Tensor, vy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """``clip_line`` plus the Wu walk's constants, for every edge of every quad at once.

    ``vx``/``vy`` are ``(M, 4)`` int64 truncated vertices; the result is ``(M, 4, EG_W)`` int32.
    This is the whole of ``aaline`` except the writes: six Cohen-Sutherland iterations and one
    integer divide, done once for ~800 edges instead of once per edge, which is where the 34.6 ms
    per polygon went.
    """
    M = vx.shape[0]
    nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
    a, b, c, d = (t.reshape(-1) for t in (vx, vy, nx, ny))
    # The clip and the walk's constants come from ``gfxdraw`` itself rather than from a second
    # transcription of them here: the per-primitive path this kernel is diffed against computes the
    # same sixteen quantities, and two copies of the same integer arithmetic in two modules that
    # must agree pixel for pixel is one edit away from a fringe that moves in one path only.
    (draw, x1, y1, x2, y2, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major,
     E, lo) = G.edge_geometry(a, b, c, d, W, H)
    kind = torch.where(vert, K_VERT, torch.where(horiz, K_HORIZ, K_GEN))
    # ``kind`` already says which axis is major for the two degenerate cases; for a general edge the
    # kernel needs it, and it is exactly ``dy > dx``. Folded into ``kind`` as a third bit rather than
    # a column of its own.
    kind = torch.where((kind == K_GEN) & ymaj, K_GEN + 4, kind)

    # Every write of this edge lies in the segment's own box grown by one pixel: the Wu pair steps
    # one off the line and the solid initial pixel is an endpoint. A tile the box misses cannot see
    # the edge, which is what lets a large quad skip its fringe entirely.
    ex0 = torch.minimum(x1, x2) - 1
    ex1 = torch.maximum(x1, x2) + 1
    ey0 = torch.minimum(y1, y2) - 1
    ey1 = torch.maximum(y1, y2) + 1
    cols = [draw.to(torch.int64), x1, y1, xx0, yy0, dx, dy, xdir, kind, major, E, lo,
            ex0, ey0, ex1, ey1]
    return torch.stack(cols, -1).to(torch.int32).reshape(M, 4, EG_W)


@torch.no_grad()
def bin_primitives(box: torch.Tensor, H: int, W: int, bin_h: int, bin_w: int):
    """Per screen-bin lists of the primitives whose box reaches them, in DRAW ORDER.

    ``box`` is ``(B,P,4)`` int32 ``x0,y0,x1,y1`` in output coordinates, with an empty box for a
    primitive that is gated off. Returns ``(lists (B,NB,K) int32, counts (B,NB) int32, nbx)``.

    Order is preserved because the list is built by a stable prefix sum over the primitive axis, not
    by a sort: painter's order IS the z-buffer here, so a bin that walked its primitives in another
    order would draw a different picture wherever two of them overlap.
    """
    B, P, _ = box.shape
    nby, nbx = -(-H // bin_h), -(-W // bin_w)
    dev = box.device
    by = torch.arange(nby, device=dev, dtype=torch.int32) * bin_h
    bx = torch.arange(nbx, device=dev, dtype=torch.int32) * bin_w
    oy = (box[..., 1:2] <= (by + bin_h - 1)) & (box[..., 3:4] >= by)            # (B,P,nby)
    ox = (box[..., 0:1] <= (bx + bin_w - 1)) & (box[..., 2:3] >= bx)            # (B,P,nbx)
    ov = (oy.unsqueeze(-1) & ox.unsqueeze(-2)).permute(0, 2, 3, 1).reshape(B, nby * nbx, P)
    ai = ov.to(torch.int32)
    cnt = ai.sum(-1)
    K = max(int(cnt.max().item()), 1)                        # the one synchronisation per frame
    pos = ai.cumsum(-1) - ai
    slot = torch.where(ov, pos, torch.full_like(pos, K)).to(torch.int64)
    lst = torch.zeros((B, nby * nbx, K + 1), dtype=torch.int32, device=dev)
    src = torch.arange(P, dtype=torch.int32, device=dev).view(1, 1, P).expand_as(slot)
    lst.scatter_(2, slot, src)
    return lst[..., :K].contiguous(), cnt.contiguous(), nbx


if HAVE_TRITON:

    # A kernel may only read a global that IS a ``tl.constexpr`` object, and the column names above
    # have to stay plain ints because the host side indexes tensors with them. So each one gets a
    # constexpr twin rather than being duplicated as a literal in the kernel, where it would drift
    # from the layout it names.
    _c = tl.constexpr
    cPM_W, cPM_NV, cPM_R, cPM_G, cPM_B = _c(PM_W), _c(PM_NV), _c(PM_R), _c(PM_G), _c(PM_B)
    cPM_X0, cPM_Y0, cPM_X1, cPM_Y1 = _c(PM_X0), _c(PM_Y0), _c(PM_X1), _c(PM_Y1)
    cEG_W, cEG_DRAW, cEG_SX, cEG_SY = _c(EG_W), _c(EG_DRAW), _c(EG_SX), _c(EG_SY)
    cEG_X0, cEG_Y0, cEG_DX, cEG_DY = _c(EG_X0), _c(EG_Y0), _c(EG_DX), _c(EG_DY)
    cEG_XDIR, cEG_KIND, cEG_MAJOR, cEG_E, cEG_LO = (_c(EG_XDIR), _c(EG_KIND), _c(EG_MAJOR),
                                                    _c(EG_E), _c(EG_LO))
    cEG_BX0, cEG_BY0, cEG_BX1, cEG_BY1 = _c(EG_BX0), _c(EG_BY0), _c(EG_BX1), _c(EG_BY1)
    cK_VERT, cK_HORIZ, cK_GEN, cK_YMAJ = _c(K_VERT), _c(K_HORIZ), _c(K_GEN), _c(K_GEN + 4)
    cM32 = _c(M32)

    @triton.jit
    def _blend(s, d, a, sh: tl.constexpr):
        """``_putPixelAlpha`` on SDL's masked channel word, in unsigned 32-bit. ``gfxdraw`` rule 4.

        The truncating ``>>8`` lands at a different bit position per channel and the unsigned wrap
        turns it into a ceiling where the source is darker than the destination, so the shift is part
        of the arithmetic and not a packing detail.

        Widened to 64 bits HERE rather than by keeping the canvas in 64: ``diff * a`` reaches 1.1e12
        and needs the room, but it is one short-lived value, where an int64 canvas is three live tile
        registers for the whole kernel. Measured on a 8x128 tile: 27.7 ms against 12.4.
        """
        s6 = s.to(tl.int64)
        d6 = d.to(tl.int64)
        diff = ((s6 - d6) << sh) & cM32
        v = ((diff * a.to(tl.int64)) & cM32) >> 8
        return (((((d6 << sh) + v) & (255 << sh)) >> sh)).to(tl.int32)

    @triton.jit
    def _round16(v):
        return (v >> 16) + ((v & 32768) >> 15)

    @triton.jit
    def _surface_kernel(pv_ptr, pm_ptr, pe_ptr, lst_ptr, cnt_ptr, aa_ptr, out_ptr,
                        P, PG, PPOST, K, NB,
                        H: tl.constexpr, W: tl.constexpr, VMAX: tl.constexpr,
                        BINH: tl.constexpr, BINW: tl.constexpr, NBX: tl.constexpr,
                        NTX: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr):
        b = tl.program_id(0)
        tid = tl.program_id(1)
        tiy = tid // NTX
        tix = tid % NTX
        oy = tiy * BH + tl.arange(0, BH)[:, None]
        ox = tix * BW + tl.arange(0, BW)[None, :]
        m = (oy < H) & (ox < W)
        # The reference's final ``transform.flip`` is applied by reading the pre-flip row here rather
        # than by moving the surface afterwards; the indicators and the score are drawn in POST-flip
        # coordinates, which is why a primitive carries which frame it is in.
        #
        # Both axes are broadcast to the full tile up front rather than left as a (BH,1) column and a
        # (1,BW) row. Triton needs a variable's shape to agree across a conditional, and the y-major
        # and x-major halves of the Wu inversion index different axes, so the lazy shapes make the
        # two branches disagree at compile time.
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
                    # ---- gfxdraw: aapolygon, then filled_polygon over the same vertices ----
                    if aa:
                        eb = pe_ptr + (b.to(tl.int64) * PG + i) * (4 * cEG_W)
                        for e in tl.static_range(4):
                            ep = eb + e * cEG_W
                            if tl.load(ep + cEG_DRAW) != 0:
                                # the edge's own box, in the surface's PRE-flip rows mapped to output
                                ebx0 = tl.load(ep + cEG_BX0)
                                ebx1 = tl.load(ep + cEG_BX1)
                                eby0 = H - 1 - tl.load(ep + cEG_BY1)
                                eby1 = H - 1 - tl.load(ep + cEG_BY0)
                                if (ebx1 >= tx0) & (ebx0 <= tx1) & (eby1 >= ty0) & (eby0 <= ty1):
                                    sx = tl.load(ep + cEG_SX)
                                    sy = tl.load(ep + cEG_SY)
                                    xx0 = tl.load(ep + cEG_X0)
                                    yy0 = tl.load(ep + cEG_Y0)
                                    dx = tl.load(ep + cEG_DX)
                                    dy = tl.load(ep + cEG_DY)
                                    xdir = tl.load(ep + cEG_XDIR)
                                    kind = tl.load(ep + cEG_KIND)
                                    major = tl.load(ep + cEG_MAJOR)
                                    # k*E is at most 999 * 65536, so the Wu walk needs no widening
                                    Ec = tl.load(ep + cEG_E)
                                    lo = tl.load(ep + cEG_LO)

                                    # 1. the solid initial pixel, plus the two axis-aligned runs SDL
                                    #    short-circuits to a solid hline/vline with no AA at all
                                    hit = (Y == sy) & (X == sx)
                                    if kind == cK_VERT:
                                        hit = hit | ((dy > 0) & (X == xx0) & (Y >= yy0)
                                                     & (Y <= yy0 + dy))
                                    if kind == cK_HORIZ:
                                        hit = hit | ((Y == yy0) & (X >= lo) & (X <= lo + dx))
                                    cr = tl.where(hit, fr, cr)
                                    cg = tl.where(hit, fg, cg)
                                    cb = tl.where(hit, fb, cb)

                                    # 2. the Wu pair that lands on this pixel, if any
                                    if kind == cK_YMAJ:                     # y-major
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
                                    # alpha = (255*w) >> 8 never reaches 255, so even a full-weight
                                    # fringe pixel blends rather than being written outright
                                    al = (255 * w) >> 8
                                    cr = tl.where(live, _blend(fr, cr, al, 16), cr)
                                    cg = tl.where(live, _blend(fg, cg, al, 8), cg)
                                    cb = tl.where(live, _blend(fb, cb, al, 0), cb)

                    # filled_polygon: a 16.16 fixed-point scanline fill. Convex, so the sort over
                    # crossings collapses to a min and a max.
                    maxy = tl.load(vb + 1)
                    for e in tl.static_range(1, 4):
                        maxy = tl.maximum(maxy, tl.load(vb + 2 * e + 1))
                    my = maxy
                    Y6 = Y.to(tl.int64)
                    BIG = 1 << 40
                    xlo_a = tl.full([BH, BW], BIG, tl.int64)
                    xhi_a = tl.full([BH, BW], -BIG, tl.int64)
                    nact = tl.zeros([BH, BW], dtype=tl.int32)
                    for e in tl.static_range(4):
                        j = (e + 3) % 4
                        ax = tl.load(vb + 2 * j + 0).to(tl.int64)
                        ay = tl.load(vb + 2 * j + 1).to(tl.int64)
                        bxv = tl.load(vb + 2 * e + 0).to(tl.int64)
                        byv = tl.load(vb + 2 * e + 1).to(tl.int64)
                        sw = ay > byv
                        ylo = tl.minimum(ay, byv)
                        yhi = tl.maximum(ay, byv)
                        xl = tl.where(sw, bxv, ax)
                        xh = tl.where(sw, ax, bxv)
                        act = (ay != byv) & (((Y6 >= ylo) & (Y6 < yhi))
                                             | ((Y6 == my) & (Y6 > ylo) & (Y6 <= yhi)))
                        den = tl.maximum(yhi - ylo, 1)
                        Xv = ((65536 * (Y6 - ylo)) // den) * (xh - xl) + 65536 * xl
                        xlo_a = tl.minimum(xlo_a, tl.where(act, Xv, BIG))
                        xhi_a = tl.maximum(xhi_a, tl.where(act, Xv, -BIG))
                        nact += act.to(tl.int32)
                    xa = tl.maximum(_round16(xlo_a + 1), 0).to(tl.int32)
                    xb = tl.minimum(_round16(xhi_a - 1), W - 1).to(tl.int32)
                    hit = (nact >= 2) & (X >= xa) & (X <= xb)
                    cr = tl.where(hit, fr, cr)
                    cg = tl.where(hit, fg, cg)
                    cb = tl.where(hit, fb, cb)

                else:
                    # ---- pygame.draw.polygon: a DIFFERENT rasterizer (see carracing.py) ----
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
                    extra = X < 0                              # an all-false tile-shaped mask
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
                        # the load-bearing line: even CROSSINGS are floored and odd ones ceiled,
                        # counting edges in vertex order per row, so a given edge changes rounding
                        # when the number of lower-indexed active edges does
                        #
                        # ``div_rn`` and not ``/``: the reference divides in float32 and then FLOORS,
                        # so a quotient that is exactly an integer must land exactly on it. Triton's
                        # ``/`` is free to use an approximate reciprocal, and -42/14 coming back as
                        # -3.0000002 floors to -4 and moves the crossing a whole pixel. Measured: one
                        # pixel in 19.2 M on 24 CarRacing frames, which is what an edge that happens
                        # to cross exactly on a pixel centre is worth.
                        q = tl.math.div_rn(
                            ((Y - y1).to(tl.int64) * (x2 - x1).to(tl.int64)).to(tl.float32),
                            tl.maximum(y2 - y1, 1).to(tl.float32))
                        xc = tl.where(nact % 2 == 0, tl.floor(q), tl.ceil(q)).to(tl.int32) + x1
                        lt += (act & (xc < X)).to(tl.int32)
                        le += (act & (xc <= X)).to(tl.int32)
                        nact += act.to(tl.int32)
                        # a horizontal edge strictly inside the polygon gets one span of its own
                        if ay == byv:
                            inside = (ay > miny) & (ay < maxy)
                            extra = extra | (inside & (Y == ay)
                                             & (X >= tl.minimum(ax, bxv))
                                             & (X <= tl.maximum(ax, bxv)))
                    # x is covered iff some EVEN k has srt[k] <= x <= srt[k+1]; with lt and le that
                    # is a range test on k and needs no sort
                    ka = tl.maximum(lt - 1, 0)
                    kb = tl.minimum(le - 1, nact - 2)
                    hit = (ka <= kb) & ((ka % 2 == 0) | (ka + 1 <= kb))
                    if miny == maxy:                        # a quad one row high: min x to max x
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
    def _shrink_kernel(in_ptr, out_ptr, idx_ptr, w_ptr, N, MOUT, C, recip,
                       K: tl.constexpr, BLOCK: tl.constexpr):
        """One axis of ``pygame.transform.smoothscale``'s shrink: a fixed gather, not a filter.

        ``(A, N, C)`` -> ``(A, MOUT, C)`` uint8. The unit is 2^14 and a WHOLE tap takes the sample
        outright while a partial one takes ``(v*w) >> 14``; the two are not the same number, which is
        why the weight is branched on rather than always multiplied. int32 throughout: a tap is at
        most 255*16384 and at most twelve of them accumulate, so nothing here reaches 2^31 -- where
        the torch form widens the whole surface to int64 and needs 19 GB to shrink 27 KB.
        """
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


#: ``(BH, BW)`` pixel tile, the ``(bin_h, bin_w)`` bins the primitive lists are built on, and the
#: warp count. The bins are coarser than the tiles on purpose: the list is ``(B, bins, P)`` in memory
#: and 4-row bins would be 200 rows of them per frame, while a 64-row bin costs each tile only the
#: few extra road quads that reach the rest of the bin -- and the measurement says the list length is
#: not what this kernel is bound by. Whole-render milliseconds at batch 1024:
#:
#:     bins      8x128/4  4x128/4  4x128/2  4x128/1  2x128/1  1x128/1  4x64/4  2x64/2
#:     16x128     126.0        --       73.9      --       --       --      76.8    86.1
#:     32x128     122.9        --       71.1      --       --       --      73.6    83.5
#:     64x128     121.4     105.1       69.3     65.4     46.6     46.1     72.6    82.1
#:    128x128     123.1     105.7       69.5     65.6     46.5     46.3      --      --
#:
#: i.e. the tile SHAPE is worth 2.7x and the bin size 4%. The gradient is entirely pixels per THREAD:
#: 8x128 with four warps holds eight and spills its three colour accumulators, 1x128 with one warp
#: holds four and does not. A one-row tile also costs nothing in coalescing here, because the row it
#: writes is 128 contiguous pixels.
TILE = (1, 128)
BINS = (64, 128)
WARPS = 1


@torch.no_grad()
def render_surface(vert, meta, edge, aa, H: int, W: int, PG: int, PPOST: int, out=None):
    """Rasterize a batch of polygon lists to ``(B,H,W,3)`` uint8, already flipped.

    ``vert`` ``(B,P,VMAX,2)`` int32 truncated screen vertices, ``meta`` ``(B,P,PM_W)`` int32,
    ``edge`` ``(B,PG,4,EG_W)`` int32 from ``edge_table``, ``aa`` ``(B,)`` int32. Primitives
    ``[0,PG)`` are gfxdraw quads, ``[PG,PPOST)`` are ``pygame.draw`` polygons in pre-flip
    coordinates, and ``[PPOST,P)`` are ``pygame.draw`` polygons already in flipped ones.
    """
    if not HAVE_TRITON:
        raise ImportError("triton unavailable")
    B, P, VMAX, _ = vert.shape
    dev = vert.device
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=dev)
    lst, cnt, nbx = bin_primitives(meta[..., cPM_X0:cPM_Y1 + 1], H, W, *BINS)
    bh, bw = TILE
    ntx = -(-W // bw)
    with torch.cuda.device(dev):
        _surface_kernel[(B, ntx * -(-H // bh))](
            vert, meta, edge, lst, cnt, aa, out,
            P, PG, PPOST, lst.shape[2], lst.shape[1],
            H=H, W=W, VMAX=VMAX, BINH=BINS[0], BINW=BINS[1], NBX=nbx,
            NTX=ntx, BH=bh, BW=bw, num_warps=WARPS)
    return out


@torch.no_grad()
def shrink(x: torch.Tensor, plan, out=None) -> torch.Tensor:
    """``(A,N,C)`` uint8 -> ``(A,MOUT,C)`` uint8 through one axis of the reference's resample."""
    A, N, C = x.shape
    if out is None:
        out = torch.empty((A, plan.m, C), dtype=torch.uint8, device=x.device)
    blk = 512
    with torch.cuda.device(x.device):
        _shrink_kernel[(A, -(-(plan.m * C) // blk))](
            x, out, plan.idx32, plan.w32, N, plan.m, C, plan.recip, K=plan.K, BLOCK=blk)
    return out
