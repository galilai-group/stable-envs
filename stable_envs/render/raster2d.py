"""Rasterize ``Scene2D`` discs, convex polygons, and capsules with Torch or a fused Triton kernel."""

from __future__ import annotations

import torch

from . import _resources as _RES
from .scene import KIND_DISC, KIND_POLY, KIND_SEGMENT, Scene2D

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


def _pixel_grid(scene: Scene2D, H: int, W: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (H, W) world-coordinate grids sampled at pixel centers, respecting flip_y."""
    x0, y0, x1, y1 = scene.world
    sx = (x1 - x0) / W
    sy = (y1 - y0) / H
    cx = x0 + (torch.arange(W, device=device, dtype=dtype) + 0.5) * sx
    cy = y0 + (torch.arange(H, device=device, dtype=dtype) + 0.5) * sy
    if scene.flip_y:
        cy = y1 + y0 - cy
    return cx.view(1, W).expand(H, W), cy.view(H, 1).expand(H, W)


def _prim_plan(scene: Scene2D, Rh: int, Rw: int) -> list[tuple]:
    """Plan active primitive families, polygon edges, and bounded pixel windows.

    Each entry contains family/ring/outline flags, (edge, mixed-degeneracy) pairs,
    and row-group windows. Bounds are conservative across the batch; degenerate
    polygons remain unbounded. Plans are recomputed from the supplied scene.
    """
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    k = scene.kind
    g = scene.geom
    G = int(g.shape[2])
    is_disc, is_poly, is_seg = k == KIND_DISC, k == KIND_POLY, k == KIND_SEGMENT

    rows = [
        is_disc.any(0),
        is_poly.any(0),
        is_seg.any(0),
        (is_disc & (g[:, :, 3] > 0)).any(0) if G > 3 else is_disc.new_zeros(N),
        (is_poly & (scene.line_w > 0)).any(0),
    ]
    # Skip an edge only if all rows are degenerate under the same dtype and expression used when drawing.
    for e in range(V):
        j = (e + 1) % V
        ex = g[:, :, 2 * j] - g[:, :, 2 * e]
        ey = g[:, :, 2 * j + 1] - g[:, :, 2 * e + 1]
        degen = (ex * ex + ey * ey) < 1e-12
        rows.append(~degen.all(0))  # some row has this edge
    for e in range(V):
        j = (e + 1) % V
        ex = g[:, :, 2 * j] - g[:, :, 2 * e]
        ey = g[:, :, 2 * j + 1] - g[:, :, 2 * e + 1]
        rows.append(((ex * ex + ey * ey) < 1e-12).any(0))  # ... and some row does not

    f64 = torch.float64
    gg = g.to(f64)
    lw = scene.line_w.abs().to(f64)
    vx, vy = gg[:, :, 0 : 2 * V : 2], gg[:, :, 1 : 2 * V : 2]
    # Treat zero shoelace area in either float64 or the geometry dtype as unbounded to match the kernel.
    area2 = (vx * torch.roll(vy, -1, -1) - torch.roll(vx, -1, -1) * vy).sum(-1)
    v32x, v32y = g[:, :, 0 : 2 * V : 2], g[:, :, 1 : 2 * V : 2]
    a32 = (v32x * torch.roll(v32y, -1, -1) - torch.roll(v32x, -1, -1) * v32y).sum(-1)
    flat = (area2 == 0) | (a32 == 0)
    r = torch.maximum(gg[:, :, 2].abs(), gg[:, :, 3].clamp_min(0)) if G > 3 else gg[:, :, 0] * 0
    hw = gg[:, :, 4].abs() if G > 4 else gg[:, :, 0] * 0
    sx1 = gg[:, :, 2] if G > 2 else gg[:, :, 0]
    sy1 = gg[:, :, 3] if G > 3 else gg[:, :, 1]
    inf = torch.full_like(area2, float("inf"))
    poly0x = torch.where(flat, -inf, vx.min(-1).values - lw)
    poly0y = torch.where(flat, -inf, vy.min(-1).values - lw)
    poly1x = torch.where(flat, inf, vx.max(-1).values + lw)
    poly1y = torch.where(flat, inf, vy.max(-1).values + lw)
    bx0 = torch.where(is_disc, gg[:, :, 0] - r, torch.where(is_poly, poly0x, torch.minimum(gg[:, :, 0], sx1) - hw))
    by0 = torch.where(is_disc, gg[:, :, 1] - r, torch.where(is_poly, poly0y, torch.minimum(gg[:, :, 1], sy1) - hw))
    bx1 = torch.where(is_disc, gg[:, :, 0] + r, torch.where(is_poly, poly1x, torch.maximum(gg[:, :, 0], sx1) + hw))
    by1 = torch.where(is_disc, gg[:, :, 1] + r, torch.where(is_poly, poly1y, torch.maximum(gg[:, :, 1], sy1) + hw))
    off = k == 0
    # NaN geometry cannot supply a safe culling bound; evaluate it across the frame.
    bx0 = torch.where(off, inf, bx0).nan_to_num(nan=float("-inf"))
    by0 = torch.where(off, inf, by0).nan_to_num(nan=float("-inf"))
    bx1 = torch.where(off, -inf, bx1).nan_to_num(nan=float("inf"))
    by1 = torch.where(off, -inf, by1).nan_to_num(nan=float("inf"))

    x0, y0, x1, y1 = scene.world
    sx, sy = (x1 - x0) / Rw, (y1 - y0) / Rh

    # Convert pixel centers to windows with one pixel of slack against rounding at the boundary.
    def _window(lx0, ly0, lx1, ly1):
        cc0 = torch.floor((lx0 - x0) / sx - 0.5) - 1
        cc1 = torch.ceil((lx1 - x0) / sx - 0.5) + 2
        if scene.flip_y:
            rr0 = torch.floor((y1 - ly1) / sy - 0.5) - 1
            rr1 = torch.ceil((y1 - ly0) / sy - 0.5) + 2
        else:
            rr0 = torch.floor((ly0 - y0) / sy - 0.5) - 1
            rr1 = torch.ceil((ly1 - y0) / sy - 0.5) + 2
        return (rr0.clamp(0, Rh), rr1.clamp(0, Rh), cc0.clamp(0, Rw), cc1.clamp(0, Rw))

    ux0, uy0 = bx0.min(0).values, by0.min(0).values
    ux1, uy1 = bx1.max(0).values, by1.max(0).values
    uwin = torch.stack(_window(ux0, uy0, ux1, uy1), dim=1)  # (N,4)

    # Build per-row windows only when the batch union is much larger than any individual bound.
    area = (bx1 - bx0).clamp_min(0) * (by1 - by0).clamp_min(0)  # (B,N)
    spread = (ux1 - ux0).clamp_min(0) * (uy1 - uy0).clamp_min(0) > _GROUP_MIN_SPREAD * area.nan_to_num(0.0).amax(
        0
    )  # (N,)

    # Transfer the complete plan once to avoid a synchronization per decision.
    flags = torch.stack(rows + [spread], dim=1).to(f64)
    plan = torch.cat([flags, uwin], dim=1).cpu().tolist()
    wide = [i for i, row in enumerate(plan) if row[5 + 2 * V] and B > 1]
    per_row = (
        torch.stack(_window(bx0[:, wide], by0[:, wide], bx1[:, wide], by1[:, wide]), dim=2)
        .to(torch.int64)
        .cpu()
        .numpy()
        if wide
        else None
    )  # (B,|wide|,4)
    out = []
    for i, row in enumerate(plan):
        disc, poly, seg, ring, outline = (bool(v) for v in row[:5])
        edges = [(e, bool(row[5 + V + e])) for e in range(V) if row[5 + e]]
        rr0, rr1, cc0, cc1 = (int(v) for v in row[6 + 2 * V :])
        win = [(None, rr0, rr1, cc0, cc1)]
        if i in wide:
            win = row_groups(per_row[:, wide.index(i)], rr0, rr1, cc0, cc1)
        out.append((disc, ring, poly, outline, seg, edges, win))
    return out


#: Required ratio of batch-union area to the largest individual box before considering row grouping.
_GROUP_MIN_SPREAD = 4.0

#: CPU dispatch cost in equivalent window pixels, with a margin before accepting grouping.
_GROUP_DISPATCH_PIXELS = 10_000
_ACCEPT_MARGIN = 1.25


def row_groups(box, r0: int, r1: int, c0: int, c1: int) -> list[tuple]:
    """Partition (B, 4) pixel boxes into row groups with smaller union windows.

    Return (row_indices, r0, r1, c0, c1) tuples. ``row_indices=None`` uses the whole
    batch when estimated savings do not cover grouping overhead. The dispatch cost
    model is calibrated for CPU operations.
    """
    import numpy as np

    B = box.shape[0]
    if B <= 1 or r1 <= r0 or c1 <= c0:
        return [(None, r0, r1, c0, c1)]
    br0, br1, bc0, bc1 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    h, w = np.maximum(br1 - br0, 0), np.maximum(bc1 - bc0, 0)
    live = (h > 0) & (w > 0)
    if not live.any():
        return [(None, r0, r1, c0, c1)]
    th, tw = max(int(h[live].max()), 1), max(int(w[live].max()), 1)
    union = (r1 - r0) * (c1 - c0)

    # Drop empty rows after assigning their boxes to spatial cells.
    cell = (br0 // th) * (int(c1) // tw + 2) + (bc0 // tw)
    cell = np.where(live, cell, -1)
    keys, inv = np.unique(cell, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(keys)))
    ends = np.append(starts[1:], len(order))

    groups, work = [], 0
    for gi, key in enumerate(keys):
        if key < 0:
            continue
        idx = order[starts[gi] : ends[gi]]
        g0, g1 = int(br0[idx].min()), int(br1[idx].max())
        f0, f1 = int(bc0[idx].min()), int(bc1[idx].max())
        work += len(idx) * (g1 - g0) * (f1 - f0)
        groups.append((idx, g0, g1, f0, f1))
    if not groups:
        return [(None, r0, r1, c0, c1)]
    # Charge dispatch overhead before accepting smaller grouped windows.
    if B * union < _ACCEPT_MARGIN * (work + len(groups) * _GROUP_DISPATCH_PIXELS):
        return [(None, r0, r1, c0, c1)]
    return groups


def _output_shape(size: int | tuple[int, int]) -> tuple[int, int]:
    return (int(size), int(size)) if isinstance(size, int) else size


@torch.no_grad()
def render_torch(scene: Scene2D, out_size: int | tuple[int, int], supersample: int = 1) -> torch.Tensor:
    """Rasterize Scene2D into (B, H, W, 3) uint8 frames on the scene's device.

    Evaluate only active primitive families inside conservative windows. A uint8
    canvas suffices without supersampling; supersampling uses a float canvas and
    averages before rounding and clamping.
    """
    H, W = _output_shape(out_size)
    ss = int(supersample)
    Rh, Rw = H * ss, W * ss
    dev, dt = scene.kind.device, scene.geom.dtype
    px, py = _pixel_grid(scene, Rh, Rw, dev, dt)
    B, _, V = scene.batch, scene.n_prims, scene.vertices

    flat_out = ss == 1
    if flat_out:
        u8 = lambda c, m: c.round().clamp(0, 255).to(torch.uint8).view(m, 1, 1, 3)  # noqa: E731
        canvas = u8(scene.background, B).expand(B, Rh, Rw, 3).clone()
    else:
        u8 = lambda c, m: c.view(m, 3, 1, 1)  # noqa: E731
        canvas = scene.background.view(B, 3, 1, 1).expand(B, 3, Rh, Rw).clone()

    for i, (has_disc, has_ring, has_poly, has_outline, has_seg, edges, windows) in enumerate(
        _prim_plan(scene, Rh, Rw)
    ):
        if not (has_disc or has_poly or has_seg):
            continue  # disabled in every environment
        for rows, r0, r1, c0, c1 in windows:
            if r1 <= r0 or c1 <= c0:
                continue  # this window's bound misses the frame
            # Gather all per-environment quantities for the selected rows; None retains a whole-batch view.
            sel = (lambda t: t) if rows is None else (lambda t, _r=rows: t[_r])
            n = B if rows is None else len(rows)
            kind = sel(scene.kind[:, i])  # (n,)
            g = sel(scene.geom[:, i])  # (n,2V)
            fill = u8(sel(scene.fill[:, i]), n)
            line = u8(sel(scene.line[:, i]), n)
            lw = sel(scene.line_w[:, i]).view(n, 1, 1)
            p = px[r0:r1, c0:c1].unsqueeze(0)
            q = py[r0:r1, c0:c1].unsqueeze(0)
            mdim = -1 if flat_out else 1

            def paint(hit, colour, _r=rows, _m=mdim, _w=(r0, r1, c0, c1)):
                """Paint the selected window, writing indexed row copies back to the canvas."""
                a, b, cc, d = _w
                if _r is None:
                    w_ = canvas[:, a:b, cc:d, :] if flat_out else canvas[:, :, a:b, cc:d]
                    w_.copy_(torch.where(hit.unsqueeze(_m), colour, w_))
                elif flat_out:
                    w_ = canvas[_r, a:b, cc:d, :]
                    canvas[_r, a:b, cc:d, :] = torch.where(hit.unsqueeze(_m), colour, w_)
                else:
                    w_ = canvas[_r, :, a:b, cc:d]
                    canvas[_r, :, a:b, cc:d] = torch.where(hit.unsqueeze(_m), colour, w_)

            if has_disc:
                d2 = (p - g[:, 0].view(n, 1, 1)) ** 2 + (q - g[:, 1].view(n, 1, 1)) ** 2
                is_disc = (kind == KIND_DISC).view(n, 1, 1)
                paint(is_disc & (d2 <= g[:, 2].view(n, 1, 1) ** 2), fill)
                if has_ring:
                    paint(is_disc & (d2 <= g[:, 3].view(n, 1, 1) ** 2) & (g[:, 3].view(n, 1, 1) > 0), line)

            # Polygon outlines are unions of edge capsules, giving round corners.
            if has_poly:
                h, w = r1 - r0, c1 - c0
                sdf = torch.full((n, h, w), -1e30, device=dev, dtype=dt)
                edge_d2 = torch.full((n, h, w), 1e30, device=dev, dtype=dt) if has_outline else None
                for e, mixed in edges:
                    ax, ay = g[:, 2 * e].view(n, 1, 1), g[:, 2 * e + 1].view(n, 1, 1)
                    j = (e + 1) % V
                    bx, by = g[:, 2 * j].view(n, 1, 1), g[:, 2 * j + 1].view(n, 1, 1)
                    ex, ey = bx - ax, by - ay
                    L2 = ex * ex + ey * ey
                    degen = L2 < 1e-12
                    inv = torch.rsqrt(L2.clamp_min(1e-20))
                    d = (p - ax) * (ey * inv) + (q - ay) * (-ex * inv)
                    sdf = torch.maximum(sdf, torch.where(degen, torch.full_like(d, -1e30), d) if mixed else d)
                    if has_outline:
                        t = (((p - ax) * ex + (q - ay) * ey) / L2.clamp_min(1e-20)).clamp(0, 1)
                        qx, qy = p - ax - t * ex, q - ay - t * ey
                        e2 = qx * qx + qy * qy
                        edge_d2 = torch.minimum(
                            edge_d2, torch.where(degen, torch.full_like(qx, 1e30), e2) if mixed else e2
                        )
                is_poly = (kind == KIND_POLY).view(n, 1, 1)
                paint(is_poly & (sdf <= 0), fill)
                if has_outline:
                    paint(is_poly & (lw > 0) & (edge_d2 <= lw * lw), line)

            if has_seg:
                ax, ay = g[:, 0].view(n, 1, 1), g[:, 1].view(n, 1, 1)
                ex, ey = g[:, 2].view(n, 1, 1) - ax, g[:, 3].view(n, 1, 1) - ay
                t = (((p - ax) * ex + (q - ay) * ey) / (ex * ex + ey * ey).clamp_min(1e-20)).clamp(0, 1)
                dx, dy = p - ax - t * ex, q - ay - t * ey
                paint((kind == KIND_SEGMENT).view(n, 1, 1) & (dx * dx + dy * dy <= g[:, 4].view(n, 1, 1) ** 2), fill)

    if flat_out:
        return canvas
    if ss > 1:
        canvas = torch.nn.functional.avg_pool2d(canvas, ss)
    return canvas.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


if HAVE_TRITON:

    @triton.jit
    def _bbox_kernel(kind_ptr, geom_ptr, lw_ptr, box_ptr, M, V: tl.constexpr, BLOCK: tl.constexpr):
        """Write (B*N, 4) conservative world bounds: x0, y0, x1, y1.

        Disabled primitives have empty bounds. Zero-area polygons have infinite bounds
        because their halfplane intersection can extend beyond the vertex hull.
        """
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < M
        k = tl.load(kind_ptr + offs, mask=m, other=0)
        lw = tl.abs(tl.load(lw_ptr + offs, mask=m, other=0.0))
        gb = geom_ptr + offs * (2 * V)
        g0 = tl.load(gb + 0, mask=m, other=0.0)
        g1 = tl.load(gb + 1, mask=m, other=0.0)
        g2 = tl.load(gb + 2, mask=m, other=0.0)
        g3 = tl.load(gb + 3, mask=m, other=0.0)
        if 2 * V > 4:  # Capsule half-width.
            g4 = tl.load(gb + 4, mask=m, other=0.0)
        else:
            g4 = tl.zeros([BLOCK], dtype=tl.float32)

        # Include a positive inner radius in the bound even when it exceeds the outer radius.
        r = tl.maximum(tl.abs(g2), tl.maximum(g3, 0.0))
        hw = tl.abs(g4)
        px0 = g0
        py0 = g1
        px1 = g0
        py1 = g1
        area2 = tl.zeros([BLOCK], dtype=tl.float32)
        for e in tl.static_range(V):
            ax = tl.load(gb + 2 * e + 0, mask=m, other=0.0)
            ay = tl.load(gb + 2 * e + 1, mask=m, other=0.0)
            j = (e + 1) % V
            bx = tl.load(gb + 2 * j + 0, mask=m, other=0.0)
            by = tl.load(gb + 2 * j + 1, mask=m, other=0.0)
            px0 = tl.minimum(px0, ax)
            px1 = tl.maximum(px1, ax)
            py0 = tl.minimum(py0, ay)
            py1 = tl.maximum(py1, ay)
            area2 += ax * by - bx * ay
        flat = area2 == 0.0
        INF = float("inf")
        x0 = tl.where(k == 1, g0 - r, tl.where(k == 2, tl.where(flat, -INF, px0 - lw), tl.minimum(g0, g2) - hw))
        y0 = tl.where(k == 1, g1 - r, tl.where(k == 2, tl.where(flat, -INF, py0 - lw), tl.minimum(g1, g3) - hw))
        x1 = tl.where(k == 1, g0 + r, tl.where(k == 2, tl.where(flat, INF, px1 + lw), tl.maximum(g0, g2) + hw))
        y1 = tl.where(k == 1, g1 + r, tl.where(k == 2, tl.where(flat, INF, py1 + lw), tl.maximum(g1, g3) + hw))
        off = k == 0
        ob = box_ptr + offs * 4
        tl.store(ob + 0, tl.where(off, INF, x0), mask=m)
        tl.store(ob + 1, tl.where(off, INF, y0), mask=m)
        tl.store(ob + 2, tl.where(off, -INF, x1), mask=m)
        tl.store(ob + 3, tl.where(off, -INF, y1), mask=m)

    @triton.jit
    def _scene_kernel(
        kind_ptr,
        geom_ptr,
        fill_ptr,
        line_ptr,
        lw_ptr,
        bg_ptr,
        base_ptr,
        box_ptr,
        out_ptr,
        N,
        PSTART,
        H: tl.constexpr,
        W: tl.constexpr,
        SS: tl.constexpr,
        V: tl.constexpr,
        wx0,
        wy0,
        sx,
        sy,
        FLIP: tl.constexpr,
        y_hi,
        HAS_BASE: tl.constexpr,
        CULL: tl.constexpr,
        NTX: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
    ):
        b = tl.program_id(0)
        tid = tl.program_id(1)
        npix = H * W
        if CULL:
            # Compact 2-D tiles allow bounds to reject primitives before evaluating their pixels.
            tiy = tid // NTX
            tix = tid % NTX
            oy = tiy * BH + tl.arange(0, BH)[:, None]
            ox = tix * BW + tl.arange(0, BW)[None, :]
            m = (oy < H) & (ox < W)
            # Use whole-pixel boundaries so the tile includes every supersample.
            bx_lo = wx0 + (tix * BW * SS) * sx
            bx_hi = wx0 + ((tix + 1) * BW * SS) * sx
            by_lo = wy0 + (tiy * BH * SS) * sy
            by_hi = wy0 + ((tiy + 1) * BH * SS) * sy
            if FLIP:
                by_lo, by_hi = y_hi - by_hi, y_hi - by_lo
        else:
            # Without culling, linear runs avoid masked row ends and keep output stores contiguous.
            pos = tid * BW + tl.arange(0, BW)[None, :] + tl.zeros([BH, 1], dtype=tl.int32)
            m = pos < npix
            oy = pos // W
            ox = pos % W
            bx_lo, bx_hi, by_lo, by_hi = wx0, wx0, wy0, wy0  # never read; CULL is False
        offs = oy * W + ox
        # Frame offsets must be int64: large batches exceed the int32 byte-address range.
        obase = b.to(tl.int64) * (npix * 3) + offs.to(tl.int64) * 3

        acc_r = tl.zeros([BH, BW], dtype=tl.float32)
        acc_g = tl.zeros([BH, BW], dtype=tl.float32)
        acc_b = tl.zeros([BH, BW], dtype=tl.float32)

        bg_r = tl.load(bg_ptr + b * 3 + 0)
        bg_g = tl.load(bg_ptr + b * 3 + 1)
        bg_b = tl.load(bg_ptr + b * 3 + 2)

        for sj in tl.static_range(SS):
            for si in tl.static_range(SS):
                ix = ox * SS + si
                iy = oy * SS + sj
                px = wx0 + (ix.to(tl.float32) + 0.5) * sx
                py = wy0 + (iy.to(tl.float32) + 0.5) * sy
                if FLIP:
                    py = y_hi - py

                # Initialize tile-wide values so Triton branch assignments retain the same tensor type.
                zero = tl.zeros([BH, BW], dtype=tl.float32)
                if HAS_BASE:
                    # The baked prefix already has full supersample resolution.
                    bo = (iy.to(tl.int64) * (W * SS) + ix.to(tl.int64)) * 3
                    cr = zero + tl.load(base_ptr + bo + 0, mask=m, other=0).to(tl.float32)
                    cg = zero + tl.load(base_ptr + bo + 1, mask=m, other=0).to(tl.float32)
                    cb = zero + tl.load(base_ptr + bo + 2, mask=m, other=0).to(tl.float32)
                else:
                    cr = zero + bg_r
                    cg = zero + bg_g
                    cb = zero + bg_b

                for i in range(PSTART, N):
                    pid = b.to(tl.int64) * N + i
                    # Empty bounds also reject disabled primitives before their geometry is loaded.
                    if CULL:
                        bb = box_ptr + pid * 4
                        live = (
                            (tl.load(bb + 2) >= bx_lo)
                            & (tl.load(bb + 0) <= bx_hi)
                            & (tl.load(bb + 3) >= by_lo)
                            & (tl.load(bb + 1) <= by_hi)
                        )
                    else:
                        live = tl.load(kind_ptr + pid) != 0
                    if live:
                        k = tl.load(kind_ptr + pid)
                        gb = geom_ptr + pid * (2 * V)
                        fr = tl.load(fill_ptr + pid * 3 + 0)
                        fg = tl.load(fill_ptr + pid * 3 + 1)
                        fb = tl.load(fill_ptr + pid * 3 + 2)
                        lr = tl.load(line_ptr + pid * 3 + 0)
                        lg = tl.load(line_ptr + pid * 3 + 1)
                        lb = tl.load(line_ptr + pid * 3 + 2)
                        lw = tl.load(lw_ptr + pid)

                        g0 = tl.load(gb + 0)
                        g1 = tl.load(gb + 1)
                        g2 = tl.load(gb + 2)
                        g3 = tl.load(gb + 3)

                        if k == 1:  # disc (+ optional inner disc)
                            dx = px - g0
                            dy = py - g1
                            d2 = dx * dx + dy * dy
                            hit = d2 <= g2 * g2
                            cr = tl.where(hit, fr, cr)
                            cg = tl.where(hit, fg, cg)
                            cb = tl.where(hit, fb, cb)
                            hit = (g3 > 0) & (d2 <= g3 * g3)
                            cr = tl.where(hit, lr, cr)
                            cg = tl.where(hit, lg, cg)
                            cb = tl.where(hit, lb, cb)

                        elif k == 2:  # convex polygon
                            sdf = tl.full([BH, BW], -1e30, tl.float32)
                            edge_d2 = tl.full([BH, BW], 1e30, tl.float32)
                            for e in tl.static_range(V):
                                ax = tl.load(gb + 2 * e + 0)
                                ay = tl.load(gb + 2 * e + 1)
                                j = (e + 1) % V
                                bx = tl.load(gb + 2 * j + 0)
                                by = tl.load(gb + 2 * j + 1)
                                ex = bx - ax
                                ey = by - ay
                                L2 = ex * ex + ey * ey
                                inv = 1.0 / tl.sqrt(L2 + 1e-20)
                                d = (px - ax) * (ey * inv) + (py - ay) * (-ex * inv)
                                # Ignore padded zero-length edges; their undefined normals would otherwise pin the SDF at zero.
                                degen = L2 < 1e-12
                                sdf = tl.maximum(sdf, tl.where(degen, -1e30, d))
                                # Edge capsules preserve round outline corners; an absolute-SDF band would produce miters.
                                wxx = px - ax
                                wyy = py - ay
                                t = (wxx * ex + wyy * ey) / (L2 + 1e-20)
                                t = tl.minimum(tl.maximum(t, 0.0), 1.0)
                                qx = wxx - t * ex
                                qy = wyy - t * ey
                                edge_d2 = tl.minimum(edge_d2, tl.where(degen, 1e30, qx * qx + qy * qy))
                            hit = sdf <= 0.0
                            cr = tl.where(hit, fr, cr)
                            cg = tl.where(hit, fg, cg)
                            cb = tl.where(hit, fb, cb)
                            hit = (lw > 0) & (edge_d2 <= lw * lw)
                            cr = tl.where(hit, lr, cr)
                            cg = tl.where(hit, lg, cg)
                            cb = tl.where(hit, lb, cb)

                        else:  # segment (capsule)
                            ex = g2 - g0
                            ey = g3 - g1
                            wxx = px - g0
                            wyy = py - g1
                            t = (wxx * ex + wyy * ey) / (ex * ex + ey * ey + 1e-20)
                            t = tl.minimum(tl.maximum(t, 0.0), 1.0)
                            dx = wxx - t * ex
                            dy = wyy - t * ey
                            hw = tl.load(gb + 4)
                            hit = dx * dx + dy * dy <= hw * hw
                            cr = tl.where(hit, fr, cr)
                            cg = tl.where(hit, fg, cg)
                            cb = tl.where(hit, fb, cb)

                acc_r += cr
                acc_g += cg
                acc_b += cb

        inv_n = 1.0 / (SS * SS)
        o = out_ptr + obase
        tl.store(o + 0, (acc_r * inv_n + 0.5).to(tl.uint8), mask=m)
        tl.store(o + 1, (acc_g * inv_n + 0.5).to(tl.uint8), mask=m)
        tl.store(o + 2, (acc_b * inv_n + 0.5).to(tl.uint8), mask=m)


#: Key static images by values so geometry or appearance changes invalidate the cache.
_BASE_CACHE_MAX = 32
_BASE_CACHE = _RES.BoundedCache(_BASE_CACHE_MAX, name="raster2d.base")

#: Minimum estimated saved subpixel work before paying the host synchronization for a bake.
_BAKE_MIN_WORK = 8e6


def _row_bytes(t: torch.Tensor, B: int) -> torch.Tensor:
    """View batched values as (B, nbytes) uint8 for dtype-independent byte comparison."""
    return t.contiguous().reshape(B, -1).view(torch.uint8)


def _static_base(scene: Scene2D, Rh: int, Rw: int, block: int) -> torch.Tensor | None:
    """Return a cached static-prefix image, or None when baking is unsuitable.

    Require byte-identical prefixes across the batch and enough estimated work to
    justify one host synchronization. Bake at full supersample resolution; the key
    includes geometry, appearance, dimensions, and device.
    """
    K = int(scene.static_prefix)
    if K <= 0:
        return None
    B = scene.batch
    if K * B * Rh * Rw < _BAKE_MIN_WORK * scene.n_prims:
        return None  # too little work to justify the synchronization
    rows = torch.cat(
        [
            _row_bytes(t, B)
            for t in (
                scene.kind[:, :K],
                scene.geom[:, :K],
                scene.fill[:, :K],
                scene.line[:, :K],
                scene.line_w[:, :K],
                scene.background,
            )
        ],
        dim=1,
    )  # (B, M) uint8
    row0 = rows[0]
    if B > 1:
        ok = (rows == rows[:1]).all().reshape(1).to(torch.uint8)
        row0 = torch.cat([row0, ok])
    blob = row0.cpu().numpy().tobytes()  # the only synchronization
    if B > 1:
        blob, uniform = blob[:-1], blob[-1]
        if not uniform:
            return None  # per-environment prefix: draw it per frame
    # Include the device so a shared process cannot reuse another GPU's cached image.
    key = (Rh, Rw, scene.world, scene.flip_y, scene.vertices, K, str(scene.kind.device), blob)
    hit = _BASE_CACHE.get(key)
    if hit is None:
        prefix = Scene2D(
            kind=scene.kind[:1, :K].contiguous(),
            geom=scene.geom[:1, :K].contiguous(),
            fill=scene.fill[:1, :K].contiguous(),
            line=scene.line[:1, :K].contiguous(),
            line_w=scene.line_w[:1, :K].contiguous(),
            background=scene.background[:1].contiguous(),
            world=scene.world,
            flip_y=scene.flip_y,
            static_prefix=0,
        )
        hit = _BASE_CACHE.put(key, render_triton(prefix, (Rh, Rw), supersample=1, block=block)[0].contiguous())
    return hit


def _destination(out, B: int, H: int, W: int) -> torch.Tensor | None:
    """Validate a contiguous uint8 output tensor and view it as (B, H, W, 3), or return None."""
    if out is None:
        return None
    if not torch.is_tensor(out):
        raise TypeError(f"render(out=) expects a torch tensor, got {type(out).__name__}")
    if not out.is_contiguous():
        raise ValueError(
            "render(out=): the destination must be contiguous because frames are written through a flat "
            "view. For a strided slice such as `buf[:, t]`, assign instead: `buf[:, t] = render(...)`."
        )
    if out.numel() != B * H * W * 3:
        raise ValueError(
            f"render(out=): the destination holds {out.numel()} elements, and the frames need "
            f"{B * H * W * 3} ({B}x{H}x{W}x3)."
        )
    if out.dtype != torch.uint8:
        raise ValueError(f"render(out=): the destination must be uint8, got {out.dtype}")
    return out.view(B, H, W, 3)


#: Primitive-count threshold for paying the culling prepass cost.
_CULL_MIN_PRIMS = 24

#: Wide tiles balance local culling with contiguous RGB writes; values are (height, width, warps).
_CULL_TILE = (8, 128, 4)

#: Maximum masked-lane fraction before selecting a narrower tile.
TILE_WASTE_MAX = 0.125


def tile_shape(H: int, W: int, lanes: int, max_width: int) -> tuple[int, int]:
    """Choose a (height, width) tile within the lane and width budgets.

    Favor wide power-of-two tiles for contiguous RGB stores, shrinking width when
    masked lanes exceed TILE_WASTE_MAX.
    """
    bw = min(max_width, 1 << max(0, (W - 1).bit_length()))
    while bw > 1 and (bw * -(-W // bw) - W) > TILE_WASTE_MAX * W:
        bw //= 2
    bh = max(1, min(lanes // bw, 1 << max(0, (H - 1).bit_length())))
    return bh, bw


def _tiles(H: int, W: int, block: int, cull: bool) -> tuple[int, int, int, int, int]:
    """Return tile dimensions, counts, and warps; unculled work uses linear pixel runs."""
    if not cull:
        return 1, block, triton.cdiv(H * W, block), 1, 4
    warps = _CULL_TILE[2]
    bh, bw = tile_shape(H, W, lanes=_CULL_TILE[0] * _CULL_TILE[1], max_width=_CULL_TILE[1])
    ntx = triton.cdiv(W, bw)
    return bh, bw, ntx * triton.cdiv(H, bh), ntx, warps


@torch.no_grad()
def render_triton(
    scene: Scene2D,
    out_size: int | tuple[int, int],
    supersample: int = 1,
    out: torch.Tensor | None = None,
    block: int = 128,
    cull: bool | None = None,
) -> torch.Tensor:
    """Rasterize a CUDA Scene2D into (B, H, W, 3) uint8 frames.

    ``cull`` forces bounding-box culling on or off; None selects by primitive count.
    ``out`` must be contiguous uint8 storage with exactly B*H*W*3 elements.
    """
    if not HAVE_TRITON:
        raise ImportError("triton unavailable; use render_torch")
    H, W = _output_shape(out_size)
    ss = int(supersample)
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    x0, y0, x1, y1 = scene.world
    buf = _destination(out, B, H, W)
    if buf is None:
        buf = torch.empty((B, H, W, 3), dtype=torch.uint8, device=scene.kind.device)
    do_cull = (N >= _CULL_MIN_PRIMS) if cull is None else bool(cull)
    # Triton takes its stream from the current CUDA device; select the scene's device explicitly.
    with torch.cuda.device(scene.kind.device):
        base = _static_base(scene, H * ss, W * ss, block)
        dummy = buf  # unused pointer when a flag is False; Triton needs an argument
        box = dummy
        if do_cull:
            M = B * N
            box = torch.empty((M, 4), dtype=scene.geom.dtype, device=scene.kind.device)
            _bbox_kernel[(triton.cdiv(M, 256),)](scene.kind, scene.geom, scene.line_w, box, M, V=V, BLOCK=256)
        bh, bw, nt, ntx, warps = _tiles(H, W, block, do_cull)
        _scene_kernel[(B, nt)](
            scene.kind,
            scene.geom,
            scene.fill,
            scene.line,
            scene.line_w,
            scene.background,
            base if base is not None else dummy,
            box,
            buf,
            N,
            int(scene.static_prefix) if base is not None else 0,
            H=H,
            W=W,
            SS=ss,
            V=V,
            wx0=x0,
            wy0=y0,
            sx=(x1 - x0) / (W * ss),
            sy=(y1 - y0) / (H * ss),
            FLIP=bool(scene.flip_y),
            y_hi=(y0 + y1),
            HAS_BASE=base is not None,
            CULL=do_cull,
            NTX=ntx,
            BH=bh,
            BW=bw,
            num_warps=warps,
        )
    return buf


@torch.no_grad()
def render(
    scene: Scene2D,
    out_size: int | tuple[int, int],
    supersample: int = 1,
    backend: str = "auto",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render uint8 RGB frames using Torch or Triton on the scene's device.

    ``auto`` selects Triton only when available and the scene is on CUDA. Explicit
    Triton rendering rejects CPU scenes. Both paths require contiguous uint8 ``out``
    storage; ``out_size`` is an integer or (height, width).
    """
    if backend == "triton" and scene.kind.device.type != "cuda":
        raise ValueError(
            f"render(backend='triton') requires a CUDA scene, got {scene.kind.device}; use backend='torch'."
        )
    if backend == "torch" or (backend == "auto" and not (HAVE_TRITON and scene.kind.device.type == "cuda")):
        H, W = _output_shape(out_size)
        # Enforce the same output-buffer contract as Triton, even though Torch can copy into strided views.
        buf = _destination(out, scene.batch, H, W)
        r = render_torch(scene, out_size, supersample)
        if buf is not None:
            buf.copy_(r)
            return buf
        return r
    return render_triton(scene, out_size, supersample, out=out)
