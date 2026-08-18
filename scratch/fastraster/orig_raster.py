"""The rasterizer kernel EXACTLY as it stood before tiling and binning, kept only to diff against.

The claim under test is that the new pixel-to-program mapping cannot move a pixel: a program's index
never enters the arithmetic, only the absolute column and row do. That is an argument, and an argument
about a rasterizer is worth what the diff says it is worth. So the old kernel is kept verbatim here
and both are run over the same scenes.

Not in ``tests/``: this is a copy of code that no longer exists, so it can only ever go stale. The
permanent version of the same claim is ``tests/gpu/test_raster_culling.py``, which diffs the two paths
that both still exist.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from stable_envs.render.raster2d import _static_base
from stable_envs.render.scene import Scene2D


@triton.jit
def _orig_kernel(kind_ptr, geom_ptr, fill_ptr, line_ptr, lw_ptr, bg_ptr, base_ptr, out_ptr,
                 N, PSTART,
                 H: tl.constexpr, W: tl.constexpr, SS: tl.constexpr, V: tl.constexpr,
                 wx0, wy0, sx, sy, FLIP: tl.constexpr, y_hi, HAS_BASE: tl.constexpr,
                 BLOCK: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    npix = H * W
    m = offs < npix
    oy = offs // W
    ox = offs % W
    obase = b.to(tl.int64) * (npix * 3) + offs.to(tl.int64) * 3

    acc_r = tl.zeros([BLOCK], dtype=tl.float32)
    acc_g = tl.zeros([BLOCK], dtype=tl.float32)
    acc_b = tl.zeros([BLOCK], dtype=tl.float32)

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

            zero = tl.zeros([BLOCK], dtype=tl.float32)
            if HAS_BASE:
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
                k = tl.load(kind_ptr + pid)
                if k != 0:
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

                    if k == 1:
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

                    elif k == 2:
                        sdf = tl.full([BLOCK], -1e30, tl.float32)
                        edge_d2 = tl.full([BLOCK], 1e30, tl.float32)
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
                            degen = L2 < 1e-12
                            sdf = tl.maximum(sdf, tl.where(degen, -1e30, d))
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

                    else:
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


@torch.no_grad()
def render_orig(scene: Scene2D, out_size, supersample: int = 1, block: int = 128) -> torch.Tensor:
    H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
    if not isinstance(out_size, int):
        H, W = out_size
    ss = int(supersample)
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    x0, y0, x1, y1 = scene.world
    buf = torch.empty((B, H, W, 3), dtype=torch.uint8, device=scene.kind.device)
    with torch.cuda.device(scene.kind.device):
        base = _static_base(scene, H * ss, W * ss, block)
        _orig_kernel[(B, triton.cdiv(H * W, block))](
            scene.kind, scene.geom, scene.fill, scene.line, scene.line_w, scene.background,
            base if base is not None else buf, buf,
            N, int(scene.static_prefix) if base is not None else 0,
            H=H, W=W, SS=ss, V=V,
            wx0=x0, wy0=y0, sx=(x1 - x0) / (W * ss), sy=(y1 - y0) / (H * ss),
            FLIP=bool(scene.flip_y), y_hi=(y0 + y1), HAS_BASE=base is not None, BLOCK=block,
        )
    return buf
