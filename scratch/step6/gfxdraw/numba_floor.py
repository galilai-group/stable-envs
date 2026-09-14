"""The same per-pixel walk as ``cost_floor.cpp``, as a numba ``@njit`` loop.

The third candidate mechanism for GFXDRAW-2-CPU, and the one whose packaging cost is different in
kind: numba is a pip dependency and an LLVM JIT at import, where a ``cpp_extension`` is a compiler
in the build and a binary in the wheel. Measured beside the compiled floor so the choice is made on
two numbers rather than on a preference.

Transcribed from ``cost_floor.cpp`` rather than from ``gfxdraw_classic.py``: the C++ is already the
scalar form, so a line-for-line port keeps the comparison about the MECHANISM. Integer throughout,
which is why this is a transcription and not a re-derivation — the six SDL_gfx rules have no
floating point in them, so there is no association order or contraction to preserve.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                                       # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):                                                    # type: ignore[misc]
        def wrap(f):
            return f
        return wrap if not a else a[0]

OP_POLY, OP_DISC, OP_HLINE, OP_VLINE, OP_BLIT = 0, 1, 2, 3, 4
CM_OP, CM_R, CM_G, CM_B, CM_X0, CM_Y0, CM_X1, CM_Y1, CM_A, CM_B2, CM_C, CM_D = range(12)
EG_DRAW, EG_SX, EG_SY, EG_X0, EG_Y0, EG_DX, EG_DY, EG_XDIR = range(8)
EG_KIND, EG_MAJOR, EG_E, EG_LO = 8, 9, 10, 11
K_GEN, K_VERT, K_HORIZ, K_YMAJ = 0, 1, 2, 4
M32 = (1 << 32) - 1


@njit(cache=True, fastmath=False, nogil=True)
def _blend1(s, d, a, sh):
    diff = ((np.int64(s) - np.int64(d)) << sh) & M32
    v = ((diff * np.int64(a)) & M32) >> 8
    return np.int32(((((np.int64(d) << sh) + v) & (np.int64(255) << sh)) >> sh))


@njit(cache=True, fastmath=False, nogil=True)
def _round16(v):
    return (v >> 16) + ((v & 32768) >> 15)


@njit(cache=True, fastmath=False, nogil=True)
def render_one(meta, vert, edge, disc, bg, aa, plate, plate_B, sprite, spr_n, spr_s,
               env, H, W, out):
    P = meta.shape[1]
    has_plate = plate.size > 0
    has_spr = sprite.size > 0
    use_aa = aa[env] != 0
    for oy in range(H):
        Y = (H - 1) - oy
        for ox in range(W):
            X = ox
            if has_plate:
                pb = env if plate_B > 1 else 0
                pl = plate_B * H * W
                po = (pb * H + Y) * W + X
                cr = np.int32(plate[po]); cg = np.int32(plate[pl + po])
                cb = np.int32(plate[2 * pl + po])
            else:
                cr = bg[env, 0]; cg = bg[env, 1]; cb = bg[env, 2]
            for i in range(P):
                if ox < meta[env, i, CM_X0] or ox > meta[env, i, CM_X1]:
                    continue
                if oy < meta[env, i, CM_Y0] or oy > meta[env, i, CM_Y1]:
                    continue
                op = meta[env, i, CM_OP]
                fr = meta[env, i, CM_R]; fg = meta[env, i, CM_G]; fb = meta[env, i, CM_B]
                pa = meta[env, i, CM_A]; pb2 = meta[env, i, CM_B2]
                pc = meta[env, i, CM_C]; pd = meta[env, i, CM_D]
                if op == OP_HLINE:
                    if Y == pa:
                        cr = fr; cg = fg; cb = fb
                elif op == OP_VLINE:
                    if X == pa and pb2 <= Y <= pc:
                        cr = fr; cg = fg; cb = fb
                elif op == OP_BLIT:
                    if has_spr:
                        off = (pd // 2) - (pb2 // 2)
                        ly = (Y - pc) - off
                        lx = (X - pc) - off
                        if 0 <= ly < pb2 and 0 <= lx < pb2:
                            so = ((((pa * spr_n + pb2) * spr_s + ly) * spr_s) + lx) * 4
                            al = np.int32(sprite[so + 3])
                            cr += ((np.int32(sprite[so + 0]) - cr) * al) >> 8
                            cg += ((np.int32(sprite[so + 1]) - cg) * al) >> 8
                            cb += ((np.int32(sprite[so + 2]) - cb) * al) >> 8
                elif op == OP_DISC:
                    r = pd
                    d = 2 * r + 1
                    dy = Y - pb2
                    dx = X - pa
                    ady = -dy if dy < 0 else dy
                    adx = -dx if dx < 0 else dx
                    if ady <= r and adx <= r:
                        wbase = pc + r + 1 + (dy + r) * d + (dx + r)
                        if use_aa:
                            for p in range(2):
                                w = disc[wbase + p * d * d]
                                if w > 0:
                                    al = (255 * w) >> 8
                                    cr = _blend1(fr, cr, al, 16)
                                    cg = _blend1(fg, cg, al, 8)
                                    cb = _blend1(fb, cb, al, 0)
                        if adx <= disc[pc + ady]:
                            cr = fr; cg = fg; cb = fb
                else:
                    nv = pa
                    if use_aa:
                        for e in range(nv):
                            if edge[env, pb2, e, EG_DRAW] == 0:
                                continue
                            kind = edge[env, pb2, e, EG_KIND]
                            xx0 = edge[env, pb2, e, EG_X0]; yy0 = edge[env, pb2, e, EG_Y0]
                            dxe = edge[env, pb2, e, EG_DX]; dye = edge[env, pb2, e, EG_DY]
                            xdir = edge[env, pb2, e, EG_XDIR]
                            major = edge[env, pb2, e, EG_MAJOR]; Ec = edge[env, pb2, e, EG_E]
                            lo = edge[env, pb2, e, EG_LO]
                            hit = (Y == edge[env, pb2, e, EG_SY]
                                   and X == edge[env, pb2, e, EG_SX])
                            if kind == K_VERT:
                                hit = hit or (dye > 0 and X == xx0 and yy0 <= Y <= yy0 + dye)
                            if kind == K_HORIZ:
                                hit = hit or (Y == yy0 and lo <= X <= lo + dxe)
                            if hit:
                                cr = fr; cg = fg; cb = fb
                            if kind == K_YMAJ:
                                kk = Y - yy0
                                acc = kk * Ec
                                wraps = acc >> 16
                                wgt = (acc & 0xFFFF) >> 8
                                xa = xx0 + xdir * wraps
                                ok = 1 <= kk < major
                                ha = ok and X == xa
                                hb = ok and X == xa + xdir
                            else:
                                kk = (X - xx0) * xdir
                                acc = kk * Ec
                                wraps = acc >> 16
                                wgt = (acc & 0xFFFF) >> 8
                                ya = yy0 + wraps
                                ok = (1 <= kk < major) and kind == K_GEN
                                ha = ok and Y == ya
                                hb = ok and Y == ya + 1
                            if ha or hb:
                                w = (255 - wgt) if ha else wgt
                                al = (255 * w) >> 8
                                cr = _blend1(fr, cr, al, 16)
                                cg = _blend1(fg, cg, al, 8)
                                cb = _blend1(fb, cb, al, 0)
                    maxy = np.int64(vert[env, pb2, 0, 1])
                    for e in range(1, nv):
                        if vert[env, pb2, e, 1] > maxy:
                            maxy = np.int64(vert[env, pb2, e, 1])
                    BIG = np.int64(1) << 40
                    xlo = BIG
                    xhi = -BIG
                    nact = 0
                    Y6 = np.int64(Y)
                    for e in range(nv):
                        j = (e + nv - 1) % nv
                        ax = np.int64(vert[env, pb2, j, 0]); ay = np.int64(vert[env, pb2, j, 1])
                        bx = np.int64(vert[env, pb2, e, 0]); by = np.int64(vert[env, pb2, e, 1])
                        if ay == by:
                            continue
                        ylo = ay if ay < by else by
                        yhi = by if ay < by else ay
                        xl = bx if ay > by else ax
                        xh = ax if ay > by else bx
                        act = (ylo <= Y6 < yhi) or (Y6 == maxy and ylo < Y6 <= yhi)
                        if not act:
                            continue
                        den = yhi - ylo
                        if den < 1:
                            den = np.int64(1)
                        Xv = ((65536 * (Y6 - ylo)) // den) * (xh - xl) + 65536 * xl
                        if Xv < xlo:
                            xlo = Xv
                        if Xv > xhi:
                            xhi = Xv
                        nact += 1
                    if nact >= 2:
                        xa2 = _round16(xlo + 1)
                        if xa2 < 0:
                            xa2 = np.int64(0)
                        xb2 = _round16(xhi - 1)
                        if xb2 > W - 1:
                            xb2 = np.int64(W - 1)
                        if xa2 <= X <= xb2:
                            cr = fr; cg = fg; cb = fb
            o = (oy * W + ox) * 3
            out[o + 0] = np.uint8(cr)
            out[o + 1] = np.uint8(cg)
            out[o + 2] = np.uint8(cb)


def render_batch(ops, out=None):
    """``render_classic_torch``'s frame, through the numba loop. For timing and for the diff."""
    import torch
    meta = ops.meta_t.cpu().numpy().astype(np.int32)
    vert = ops.vert_t.cpu().numpy().astype(np.int32)
    edge = ops.edge_t.cpu().numpy().astype(np.int32)
    disc = ops.discs.flat.cpu().numpy().astype(np.int32)
    bg = ((ops.bg if ops.bg is not None else torch.zeros((ops.B, 3), dtype=torch.int32))
          .cpu().numpy().astype(np.int32))
    aa = ops.aa.cpu().numpy().astype(np.int32)
    plate = (ops.plate.cpu().numpy().astype(np.uint8).reshape(-1) if ops.plate is not None
             else np.zeros(0, np.uint8))
    spr = (ops.sprite.cpu().numpy().astype(np.uint8).reshape(-1) if ops.sprite is not None
           else np.zeros(0, np.uint8))
    plate_B = int(ops.plate.shape[1]) if ops.plate is not None else 1
    spr_n = int(ops.sprite.shape[1]) if ops.sprite is not None else 1
    spr_s = int(ops.sprite.shape[2]) if ops.sprite is not None else 1
    if out is None:
        out = np.empty((ops.B, ops.H, ops.W, 3), np.uint8)
    for e in range(ops.B):
        render_one(meta, vert, edge, disc, bg, aa, plate, plate_B, spr, spr_n, spr_s,
                   e, ops.H, ops.W, out[e].reshape(-1))
    return out
