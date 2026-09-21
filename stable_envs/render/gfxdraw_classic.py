"""``gfxdraw.py``'s rules for a classic-control frame, per PIXEL and in one kernel.

Why this exists next to ``gfxdraw.py`` and ``gfxdraw_fused.py``. Step 6 took the per-primitive torch
path from 60 to 258-520 frames/s on one CPU core and from 16,661 to 37,221 on a 3090 by making each
primitive write only its own box on a planar surface. What is left is the shape itself, and it is
the same shape ``gfxdraw_fused`` was written to escape for CarRacing:

  * **~20,000 operator launches per frame**, whatever the batch. At batch 1 that IS the frame.
  * **the surface is read and written once per primitive**, even bounded: a rotated quad's box is
    203x203 on a 500x500 canvas and the fringe scatters touch the whole frame's index space.
  * **the output interleave**, where no torch or numpy layout measured on the laptop gets under
    0.202 ms per 400x600 frame on one core and the shipped form costs 0.289 -- an indication,
    measured off the certification host, not a proof.

None of the three is a property of the RULES. SDL_gfx's primitives are integer arithmetic on integer
geometry and every one of them evaluates per pixel in closed form, so the frame can be a single pass
that walks the primitive list in registers and stores each pixel once, already flipped.

``gfxdraw_fused`` is that for CarRacing and covers exactly what CarRacing draws: 4-vertex gfxdraw
quads and ``pygame.draw`` polygons. Classic control draws a different five-primitive alphabet --
a polygon (3 or 4 vertices) with its Wu fringe, a filled circle with its ``aacircle`` fringe, an
``hline``, a ``vline``, and an alpha blit of a pre-scaled sprite -- over either a flat background or
a pre-baked plate. This module is that alphabet.

THE THINGS THAT MAKE IT THE SAME PICTURE, not a similar one:

1. **Order is the loop.** Painter's order across primitives, and within a primitive the fringe
   blending before the fill overwrites it, are the order of the register updates. Within a polygon
   the EDGES are also walked in order, which is stricter than the per-primitive path's scatter: two
   edges can land on one pixel at a shared vertex, and a scatter reads every destination before
   writing any of them while a register walk cannot.
2. **The Wu walk inverts** (``gfxdraw_fused``'s docstring derives this): ``k`` is recoverable from
   the pixel, so a pixel asks the edge one question instead of the edge visiting every pixel.
3. **A disc is a table.** ``filled_circle``'s half-width per row and ``aacircle``'s fringe weights
   are constants of the RADIUS -- which is a constant of the environment, never of the state or of
   the style -- so both become a dense lookup at ``(dy, dx)`` rather than a walk. The tables are
   built by ``gfxdraw.circle_hw`` and ``gfxdraw.AACIRCLE_FRINGE``, the same two objects the
   per-primitive path uses, so there is one definition of a disc in this repository.
4. **Everything a style can change flows through as data.** Colours are per (environment,
   primitive); the anti-alias switch is per environment; MountainCar's baked terrain is per
   environment; Pendulum's sprite index is per environment. Nothing is cached on anything but
   static geometry, so a batch holding 4,096 different appearances is one launch.

``render_classic_torch`` is the executable specification: the same closed form, written in torch,
readable and slow. ``tests/gpu/test_gfxdraw_classic.py`` diffs the Triton kernel against it and
diffs IT against ``gfxdraw.py``'s per-primitive path, so the chain from the six measured rules to
the kernel is checked at both joints rather than asserted at either.
"""

from __future__ import annotations

import numpy as np
import torch

from . import gfxdraw as G
from .gfxdraw_fused import (
    EG_DRAW,
    EG_DX,
    EG_DY,
    EG_E,
    EG_KIND,
    EG_LO,
    EG_MAJOR,
    EG_SX,
    EG_SY,
    EG_X0,
    EG_XDIR,
    EG_Y0,
    HAVE_TRITON,
    K_GEN,
    K_HORIZ,
    K_VERT,
    edge_table,
)

if HAVE_TRITON:
    import triton
    import triton.language as tl

#: the five primitives a classic-control frame is made of
OP_POLY, OP_DISC, OP_HLINE, OP_VLINE, OP_BLIT = range(5)

#: per-primitive metadata columns. ``A..D`` are op-specific; the box is in OUTPUT (post-flip) rows.
(CM_OP, CM_R, CM_G, CM_B, CM_X0, CM_Y0, CM_X1, CM_Y1, CM_A, CM_B2, CM_C, CM_D) = range(12)
CM_W = 12

#: ``aacircle``'s fringe weight meaning ``pixelColor`` rather than ``pixelColorWeight``
SOLID = G.SOLID


# ------------------------------------------------------------------ disc tables
def disc_table(r: int) -> np.ndarray:
    """``filled_circle`` half-widths and ``aacircle`` fringe weights for one radius, as one block.

    Layout, int32: ``[hw[0..r]] + [w0 over (2r+1)^2] + [w1 over (2r+1)^2]``, where ``w{p}`` is the
    fringe weight of pass ``p`` at ``(dy + r, dx + r)`` and 0 means "no write". Two passes because
    ``aaellipseColor`` writes a few pixels twice and the two writes must land in order.

    Derived from ``gfxdraw.circle_hw`` and ``gfxdraw.AACIRCLE_FRINGE`` -- the same two objects the
    per-primitive path reads -- so a disc has one definition here.
    """
    r = int(r)
    d = 2 * r + 1
    hw = np.asarray(G.circle_hw(r), np.int32)
    w = np.zeros((2, d, d), np.int32)
    # A dense map holds ONE weight per (pass, dy, dx), so two entries of the same pass at the same
    # offset would silently become one and the kernel would drop a write the per-primitive path
    # makes. Two passes exist precisely because ``aaellipseColor`` writes a few pixels twice, so a
    # duplicate ACROSS passes is the normal case and a duplicate WITHIN one is a table that this
    # representation cannot hold. Checked rather than assumed, because the table is generated
    # and a regenerated one must fail loudly here
    # instead of quietly losing pixels only on the radii that grew one.
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
        # A fringe weight of SOLID would mean pixelColor -- write the colour outright rather than
        # blend -- and neither this table nor the kernel implements that case for a disc. No radius
        # in AACIRCLE_FRINGE uses it; checked here so it cannot appear silently in a regenerated
        # table and diverge from the per-primitive path only on the pixels it touches.
        assert 0 < wt < SOLID, f"aacircle fringe weight {wt} at r={r} is outside (0, {SOLID})"
        w[pas, dy + r, dx + r] = wt
    return np.concatenate([hw, w[0].reshape(-1), w[1].reshape(-1)])


class DiscTables:
    """Every radius a frame uses, concatenated once, with the offset each op needs."""

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


# ------------------------------------------------------------------ the op list
class Ops:
    """One frame's primitive list, as device tensors. Nothing here depends on the batch's VALUES
    except through the per-environment columns, which is what keeps a 4,096-appearance batch to one
    launch."""

    def __init__(self, B, H, W, device):
        self.B, self.H, self.W, self.device = B, H, W, device
        self.meta: list[torch.Tensor] = []
        self.verts: list[torch.Tensor] = []
        self.bg = None
        self.plate = None
        self.sprite = None
        self.aa = None
        self.discs = None

    # --- builders, one per primitive -------------------------------------------------------
    def _row(self, op, color, x0, y0, x1, y1, a=0, b=0, c=0, d=0):
        """One ``(B, CM_W)`` metadata row. ``y0``/``y1`` arrive PRE-flip and are stored post-flip."""
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

    def polygon(self, verts, color, nv=None):
        """``aapolygon`` then ``filled_polygon`` over float64 vertices, as the env supplies them."""
        vi = torch.trunc(verts).to(torch.int64)  # (B, n, 2)
        n = int(vi.shape[1]) if nv is None else int(nv)
        pad = vi if n == 4 else torch.cat([vi, vi[:, :1]], 1)  # [v0..v_{n-1}, v0]
        slot = len(self.verts)
        self.verts.append(pad[:, :4])
        vx, vy = pad[..., 0], pad[..., 1]
        # the box is the vertices grown by one: the fill's span is round16(X +/- 1) and the fringe
        # steps one pixel off the line, so nothing this primitive writes lies outside it
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
        """``aacircle`` then ``filled_circle``. ``r`` is a constant of the environment."""
        r = int(r)
        self._row(
            OP_DISC, color, cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1, a=cx, b=cy, c=self.discs.offset[r], d=r
        )

    def hline(self, y, color):
        self._row(OP_HLINE, color, 0, y, self.W - 1, y, a=y)

    def vline(self, x, y0, y1, color):
        self._row(OP_VLINE, color, x, y0, x, y1, a=x, b=y0, c=y1)

    def blit(self, origin, win, flip, n, color):
        """Pendulum's torque arrow: a pre-scaled RGBA sprite composited with SDL's own blit rule."""
        self._row(OP_BLIT, color, origin, origin, origin + win - 1, origin + win - 1, a=flip, b=n, c=origin, d=win)

    # --- finish ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------ the specification, in torch
@torch.no_grad()
def render_classic_torch(ops: Ops, out: torch.Tensor | None = None) -> torch.Tensor:
    """The fused semantics, in torch, over the whole frame. The readable specification.

    Every branch the Triton kernel takes is taken here in the same order over the same closed form;
    the difference is that this evaluates all pixels of all primitives and keeps ``(B,H,W,3)``
    intermediates, which is exactly the cost the kernel exists to remove. Slow on purpose: this is
    an oracle, and an oracle that shares an optimisation with the thing it checks is not one.
    """
    B, H, W, dev = ops.B, ops.H, ops.W, ops.device
    sh = torch.tensor(G.SHIFT, dtype=torch.int64, device=dev)
    ys = torch.arange(H, device=dev).view(1, -1, 1)
    xs = torch.arange(W, device=dev).view(1, 1, -1)
    ypre = (H - 1) - ys  # (1,H,1) pre-flip row
    aa = (ops.aa != 0).view(-1, 1, 1)

    if ops.plate is not None:
        p = ops.plate  # (3, 1 or B, H, W) pre-flip
        col = p[:, :, :, :].permute(1, 2, 3, 0).to(torch.int64)  # (1 or B, H, W, 3)
        col = col.flip(1).expand(B, H, W, 3).clone()
    else:
        col = ops.bg.to(torch.int64).view(B, 1, 1, 3).expand(B, H, W, 3).clone()

    meta = ops.meta_t.to(torch.int64)
    for i in range(meta.shape[1]):
        m = meta[:, i]  # (B, CM_W)
        op = int(m[0, CM_OP])
        c3 = m[:, CM_R : CM_B + 1].view(B, 1, 1, 3)
        box = (
            (ys >= m[:, CM_Y0].view(-1, 1, 1))
            & (ys <= m[:, CM_Y1].view(-1, 1, 1))
            & (xs >= m[:, CM_X0].view(-1, 1, 1))
            & (xs <= m[:, CM_X1].view(-1, 1, 1))
        )
        if op == OP_HLINE:
            hit = box & (ypre == m[:, CM_A].view(-1, 1, 1))
            col = torch.where(hit.unsqueeze(-1), c3, col)
        elif op == OP_VLINE:
            hit = (
                box
                & (xs == m[:, CM_A].view(-1, 1, 1))
                & (ypre >= m[:, CM_B2].view(-1, 1, 1))
                & (ypre <= m[:, CM_C].view(-1, 1, 1))
            )
            col = torch.where(hit.unsqueeze(-1), c3, col)
        elif op == OP_BLIT:
            col = _blit_torch(ops, m, col, ypre, xs, box)
        elif op == OP_DISC:
            col = _disc_torch(ops, m, col, ypre, xs, box, aa, sh, c3)
        elif op == OP_POLY:
            col = _poly_torch(ops, m, col, ypre, xs, box, aa, sh, c3, int(m[0, CM_B2]))
        else:  # pragma: no cover
            raise ValueError(f"unknown op {op}")

    frame = col.to(torch.uint8)
    if out is None:
        return frame
    out.copy_(frame)
    return out


def _blit_torch(ops, m, col, ypre, xs, box):
    o = m[:, CM_C].view(-1, 1, 1)
    n = m[:, CM_B2].view(-1, 1, 1)
    off = (m[:, CM_D].view(-1, 1, 1) // 2) - torch.div(n, 2, rounding_mode="floor")
    ly = (ypre - o) - off
    lx = (xs - o) - off
    ok = box & (ly >= 0) & (ly < n) & (lx >= 0) & (lx < n)
    S = ops.sprite.shape[-2]
    rgba = ops.sprite[
        m[:, CM_A].view(-1, 1, 1).expand_as(ly), n.expand_as(ly), ly.clamp(0, S - 1), lx.clamp(0, S - 1)
    ].to(torch.int64)
    a = torch.where(ok, rgba[..., 3], torch.zeros_like(rgba[..., 3])).unsqueeze(-1)
    return col + (((rgba[..., :3] - col) * a) >> 8)


def _disc_torch(ops, m, col, ypre, xs, box, aa, sh, c3):
    r = int(m[0, CM_D])
    d = 2 * r + 1
    base = int(m[0, CM_C])
    hw = ops.discs.flat[base : base + r + 1].to(torch.int64)
    w0 = ops.discs.flat[base + r + 1 : base + r + 1 + d * d].to(torch.int64).view(d, d)
    w1 = ops.discs.flat[base + r + 1 + d * d : base + r + 1 + 2 * d * d].to(torch.int64).view(d, d)
    dy = ypre - m[:, CM_B2].view(-1, 1, 1)
    dx = xs - m[:, CM_A].view(-1, 1, 1)
    near = box & (dy.abs() <= r) & (dx.abs() <= r)
    iy = (dy + r).clamp(0, d - 1)
    ix = (dx + r).clamp(0, d - 1)
    for wmap in (w0, w1):
        w = torch.where(near, wmap[iy, ix], torch.zeros_like(iy))
        live = (w > 0) & aa
        col = torch.where(live.unsqueeze(-1), G.blend(c3, col, w.unsqueeze(-1), sh), col)
    h = torch.where(dy.abs() <= r, hw[dy.abs().clamp(0, r)], torch.full_like(dy, -1))
    return torch.where((box & (dx.abs() <= h)).unsqueeze(-1), c3, col)


def _poly_torch(ops, m, col, ypre, xs, box, aa, sh, c3, slot):
    B, W = ops.B, ops.W
    nv = int(m[0, CM_A])
    eg = ops.edge_t[:, slot].to(torch.int64)  # (B,4,EG_W)
    Y, X = ypre, xs
    for e in range(nv):
        E = eg[:, e]

        def g(c):
            return E[:, c].view(-1, 1, 1)

        draw = g(EG_DRAW) != 0
        kind = g(EG_KIND)
        xx0, yy0, dx, dy, xdir = g(EG_X0), g(EG_Y0), g(EG_DX), g(EG_DY), g(EG_XDIR)
        major, Ec, lo = g(EG_MAJOR), g(EG_E), g(EG_LO)
        hit = draw & (Y == g(EG_SY)) & (X == g(EG_SX))
        hit = hit | (draw & (kind == K_VERT) & (dy > 0) & (X == xx0) & (Y >= yy0) & (Y <= yy0 + dy))
        hit = hit | (draw & (kind == K_HORIZ) & (Y == yy0) & (X >= lo) & (X <= lo + dx))
        col = torch.where((hit & aa).unsqueeze(-1), c3, col)

        ymaj = kind == K_GEN + 4
        kk = torch.where(ymaj, Y - yy0, (X - xx0) * xdir)
        acc = kk * Ec
        wraps = acc >> 16
        wgt = (acc & 0xFFFF) >> 8
        ok = draw & (kk >= 1) & (kk < major) & ((kind == K_GEN) | ymaj)
        ha = ok & torch.where(ymaj, X == xx0 + xdir * wraps, Y == yy0 + wraps)
        hb = ok & torch.where(ymaj, X == xx0 + xdir * (wraps + 1), Y == yy0 + wraps + 1)
        w = torch.where(ha, 255 - wgt, wgt)
        live = (ha | hb) & aa
        col = torch.where(live.unsqueeze(-1), G.blend(c3, col, w.unsqueeze(-1), sh), col)

    v = ops.vert_t[:, slot].to(torch.int64)  # (B,4,2)
    maxy = v[:, :nv, 1].amax(1).view(-1, 1, 1)
    BIG = 1 << 40
    xlo = torch.full_like(Y + X * 0, BIG).expand(B, ops.H, W).clone()
    xhi = torch.full_like(xlo, -BIG)
    nact = torch.zeros_like(xlo)
    for e in range(nv):
        j = (e + nv - 1) % nv
        ax, ay = v[:, j, 0].view(-1, 1, 1), v[:, j, 1].view(-1, 1, 1)
        bx, by = v[:, e, 0].view(-1, 1, 1), v[:, e, 1].view(-1, 1, 1)
        sw = ay > by
        ylo, yhi = torch.minimum(ay, by), torch.maximum(ay, by)
        xl = torch.where(sw, bx, ax)
        xh = torch.where(sw, ax, bx)
        act = (ay != by) & (((Y >= ylo) & (Y < yhi)) | ((Y == maxy) & (Y > ylo) & (Y <= yhi)))
        den = (yhi - ylo).clamp_min(1)
        Xv = torch.div(65536 * (Y - ylo), den, rounding_mode="floor") * (xh - xl) + 65536 * xl
        xlo = torch.minimum(xlo, torch.where(act, Xv, torch.full_like(Xv, BIG)))
        xhi = torch.maximum(xhi, torch.where(act, Xv, torch.full_like(Xv, -BIG)))
        nact = nact + act.to(xlo.dtype)
    xa = G._round16(xlo + 1).clamp_min(0)
    xb = G._round16(xhi - 1).clamp_max(W - 1)
    fill = box & (nact >= 2) & (X >= xa) & (X <= xb)
    return torch.where(fill.unsqueeze(-1), c3, col)


# ------------------------------------------------------------------ the kernel
if HAVE_TRITON:
    # The edge layout's constexpr twins, the Wu blend and the 16.16 rounding come from
    # ``gfxdraw_fused``: this kernel reads the table THAT module writes, so a second set of column
    # constexprs here would be a second thing to keep in step with the layout they name.
    from .gfxdraw_fused import (
        _blend,
        _round16,
        cEG_DRAW,
        cEG_DX,
        cEG_DY,
        cEG_E,
        cEG_KIND,
        cEG_LO,
        cEG_MAJOR,
        cEG_SX,
        cEG_SY,
        cEG_W,
        cEG_X0,
        cEG_XDIR,
        cEG_Y0,
        cK_GEN,
        cK_HORIZ,
        cK_VERT,
        cK_YMAJ,
    )

    # A kernel may only read a global that IS a ``tl.constexpr`` object, and the column names have
    # to stay plain ints because the host side indexes tensors with them -- so each gets a
    # constexpr twin rather than being duplicated as a literal where it would drift from the layout
    # it names.
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

        # Both axes are broadcast to the full tile up front rather than left as a column and a row:
        # Triton needs a variable's shape to agree across a conditional, and the y-major and x-major
        # halves of the Wu inversion index different axes.
        zi = tl.zeros([BH, BW], dtype=tl.int32)
        X = zi + ox
        Yout = zi + oy
        # The reference's final ``transform.flip`` is applied by reading the pre-flip row here
        # rather than by moving the surface afterwards, which is the whole of the interleave the
        # per-primitive path pays once per frame.
        Y = (H - 1) - Yout

        aa = tl.load(aa_ptr + b) != 0
        tx0 = tix * BW
        tx1 = tx0 + BW - 1
        ty0 = tiy * BH
        ty1 = ty0 + BH - 1

        if HAS_PLATE:
            # the plate is (3, 1, H, W) when the batch shares one appearance and (3, B, H, W)
            # when it does not; b % PLATE_B is that choice without a branch
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
                        # SDL_BlitSurface, NOT SDL_gfx's pixel writer: d + ((s - d) * a) >> 8
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
                        # pass 0 then pass 1: aaellipseColor writes a few pixels twice and the two
                        # writes must land in order
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
                    # ---- aapolygon: every edge, in order --------------------------------------
                    if aa:
                        eb = edge_ptr + (b64 * PG + pb2.to(tl.int64)) * (4 * cEG_W)
                        for e in tl.static_range(4):
                            if e < nv:
                                ep = eb + e * cEG_W
                                if tl.load(ep + cEG_DRAW) != 0:
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

                                    # the solid initial pixel, plus the two axis-aligned runs
                                    # SDL_gfx short-circuits to a solid hline/vline with no AA
                                    hit = (Y == sy) & (X == sx)
                                    if kind == cK_VERT:
                                        hit = hit | ((dye > 0) & (X == xx0) & (Y >= yy0) & (Y <= yy0 + dye))
                                    if kind == cK_HORIZ:
                                        hit = hit | ((Y == yy0) & (X >= lo) & (X <= lo + dxe))
                                    cr = tl.where(hit, fr, cr)
                                    cg = tl.where(hit, fg, cg)
                                    cb = tl.where(hit, fb, cb)

                                    # the Wu pair that lands on this pixel, if any
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
                                    # alpha = (255*w) >> 8 never reaches 255, so even a full-weight
                                    # fringe pixel blends rather than being written outright
                                    al = (255 * w) >> 8
                                    cr = tl.where(live, _blend(fr, cr, al, 16), cr)
                                    cg = tl.where(live, _blend(fg, cg, al, 8), cg)
                                    cb = tl.where(live, _blend(fb, cb, al, 0), cb)

                    # ---- filled_polygon: a 16.16 fixed-point scanline fill ----------------------
                    maxy = tl.load(vb + 1)
                    for e in tl.static_range(1, 4):
                        if e < nv:
                            maxy = tl.maximum(maxy, tl.load(vb + 2 * e + 1))
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
                            act = (ay != byv) & (
                                ((Y6 >= ylo) & (Y6 < yhi)) | ((Y6 == maxy) & (Y6 > ylo) & (Y6 <= yhi))
                            )
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

        o = out_ptr + (b64 * (H * W) + oy.to(tl.int64) * W + ox.to(tl.int64)) * 3
        tl.store(o + 0, cr.to(tl.uint8), mask=m)
        tl.store(o + 1, cg.to(tl.uint8), mask=m)
        tl.store(o + 2, cb.to(tl.uint8), mask=m)


#: ``(BH, BW)`` pixel tile and warp count, inherited from ``gfxdraw_fused``'s own sweep: the
#: gradient there is entirely pixels per THREAD (a tall tile spills the three colour accumulators),
#: and a one-row tile costs nothing in coalescing because the row it writes is 128 contiguous pixels.
TILE = (1, 128)
WARPS = 1


@torch.no_grad()
def render_classic(ops: Ops, out: torch.Tensor | None = None) -> torch.Tensor:
    """Rasterize a classic-control frame list to ``(B,H,W,3)`` uint8, already flipped."""
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
