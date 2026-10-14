from __future__ import annotations

import torch

from stable_envs.render import gfxdraw as G
from stable_envs.render.gfxdraw import classic as C
from stable_envs.render.gfxdraw import fused as F


@torch.no_grad()
def render_classic_torch(ops: C.Ops, out: torch.Tensor | None = None) -> torch.Tensor:
    """Reference torch implementation of the fused gfxdraw kernel over the whole frame.

    Takes every branch the Triton kernel takes, in the same order and over the same closed form,
    but evaluates all pixels of all primitives with ``(B,H,W,3)`` intermediates. It shares no
    optimisation with the kernel it checks.
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
        m = meta[:, i]  # (B, C.CM_W)
        op = int(m[0, C.CM_OP])
        c3 = m[:, C.CM_R : C.CM_B + 1].view(B, 1, 1, 3)
        box = (
            (ys >= m[:, C.CM_Y0].view(-1, 1, 1))
            & (ys <= m[:, C.CM_Y1].view(-1, 1, 1))
            & (xs >= m[:, C.CM_X0].view(-1, 1, 1))
            & (xs <= m[:, C.CM_X1].view(-1, 1, 1))
        )
        if op == C.OP_HLINE:
            hit = box & (ypre == m[:, C.CM_A].view(-1, 1, 1))
            col = torch.where(hit.unsqueeze(-1), c3, col)
        elif op == C.OP_VLINE:
            hit = (
                box
                & (xs == m[:, C.CM_A].view(-1, 1, 1))
                & (ypre >= m[:, C.CM_B2].view(-1, 1, 1))
                & (ypre <= m[:, C.CM_C].view(-1, 1, 1))
            )
            col = torch.where(hit.unsqueeze(-1), c3, col)
        elif op == C.OP_BLIT:
            col = _blit_torch(ops, m, col, ypre, xs, box)
        elif op == C.OP_DISC:
            col = _disc_torch(ops, m, col, ypre, xs, box, aa, sh, c3)
        elif op == C.OP_POLY:
            col = _poly_torch(ops, m, col, ypre, xs, box, aa, sh, c3, int(m[0, C.CM_B2]))
        else:  # pragma: no cover
            raise ValueError(f"unknown op {op}")

    frame = col.to(torch.uint8)
    if out is None:
        return frame
    out.copy_(frame)
    return out


def _blit_torch(ops, m, col, ypre, xs, box):
    o = m[:, C.CM_C].view(-1, 1, 1)
    n = m[:, C.CM_B2].view(-1, 1, 1)
    off = (m[:, C.CM_D].view(-1, 1, 1) // 2) - torch.div(n, 2, rounding_mode="floor")
    ly = (ypre - o) - off
    lx = (xs - o) - off
    ok = box & (ly >= 0) & (ly < n) & (lx >= 0) & (lx < n)
    S = ops.sprite.shape[-2]
    rgba = ops.sprite[
        m[:, C.CM_A].view(-1, 1, 1).expand_as(ly), n.expand_as(ly), ly.clamp(0, S - 1), lx.clamp(0, S - 1)
    ].to(torch.int64)
    a = torch.where(ok, rgba[..., 3], torch.zeros_like(rgba[..., 3])).unsqueeze(-1)
    return col + (((rgba[..., :3] - col) * a) >> 8)


def _disc_torch(ops, m, col, ypre, xs, box, aa, sh, c3):
    """Each environment's disc at its own radius; environments sharing one are drawn together."""
    for r, base in sorted(set(zip(m[:, C.CM_D].tolist(), m[:, C.CM_C].tolist(), strict=True))):
        rows = ((m[:, C.CM_D] == r) & (m[:, C.CM_C] == base)).view(-1, 1, 1)
        col = _disc_rows(ops, m, col, ypre, xs, box & rows, aa, sh, c3, int(r), int(base))
    return col


def _disc_rows(ops, m, col, ypre, xs, box, aa, sh, c3, r, base):
    d = 2 * r + 1
    hw = ops.discs.flat[base : base + r + 1].to(torch.int64)
    w0 = ops.discs.flat[base + r + 1 : base + r + 1 + d * d].to(torch.int64).view(d, d)
    w1 = ops.discs.flat[base + r + 1 + d * d : base + r + 1 + 2 * d * d].to(torch.int64).view(d, d)
    dy = ypre - m[:, C.CM_B2].view(-1, 1, 1)
    dx = xs - m[:, C.CM_A].view(-1, 1, 1)
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
    nv = int(m[0, C.CM_A])
    eg = ops.edge_t[:, slot].to(torch.int64)  # (B,4,F.EG_W)
    Y, X = ypre, xs
    for e in range(nv):
        E = eg[:, e]

        def g(c):
            return E[:, c].view(-1, 1, 1)

        draw = g(F.EG_DRAW) != 0
        kind = g(F.EG_KIND)
        xx0, yy0, dx, dy, xdir = g(F.EG_X0), g(F.EG_Y0), g(F.EG_DX), g(F.EG_DY), g(F.EG_XDIR)
        major, Ec, lo = g(F.EG_MAJOR), g(F.EG_E), g(F.EG_LO)
        hit = draw & (Y == g(F.EG_SY)) & (X == g(F.EG_SX))
        hit = hit | (draw & (kind == F.K_VERT) & (dy > 0) & (X == xx0) & (Y >= yy0) & (Y <= yy0 + dy))
        hit = hit | (draw & (kind == F.K_HORIZ) & (Y == yy0) & (X >= lo) & (X <= lo + dx))
        col = torch.where((hit & aa).unsqueeze(-1), c3, col)

        ymaj = kind == F.K_GEN + 4
        kk = torch.where(ymaj, Y - yy0, (X - xx0) * xdir)
        acc = kk * Ec
        wraps = acc >> 16
        wgt = (acc & 0xFFFF) >> 8
        ok = draw & (kk >= 1) & (kk < major) & ((kind == F.K_GEN) | ymaj)
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
