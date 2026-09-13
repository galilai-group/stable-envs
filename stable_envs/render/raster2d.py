"""One fused kernel that rasterizes any ``Scene2D``. This is the engine the whole repo stands on.

Design, and why it is a single kernel rather than a library of them. Every colour in a flat 2-D scene is
a closed-form function of a handful of numbers per primitive, so an entire supersample block for one
output pixel can be evaluated in registers and reduced before anything is written. The alternative —
compositing with tensor ops — allocates a ``(B,3,R,R)`` float canvas and touches it once per primitive;
measured on the PushT prototype that was ~2 Gpixel/s on a card with far more bandwidth than that, and
the fused kernel reaches 3.2 M frames/s end-to-end at 96px and 5.6 M for the rasterizer alone
(155 GB/s). The gap between the two is ``scene()``: Python-issued small torch ops, flat regardless of
batch, i.e. 41% of wall time at B=16384 and the next thing worth attacking.

**That cost was recorded here as "~2.0 ms" and the number was PushT's.** It is ~2.0 ms for a scene of
nine primitives and 100-315 ms for one of 157 to 436 — still flat in the batch (roundabout-v0 measured
312 ms at B=16 and 312 ms at B=1024, 53,481 torch calls for 436 primitives), so still pure dispatch,
but two orders of magnitude larger and the dominant cost of four of the six highway-env envs rather
than a footnote. ``SceneBuilder`` now takes a batched push so an env issues one call per GROUP of
primitives; roundabout-v0's ``scene()`` is 3.5 ms.

This is the same failure the ``tl.static_range`` note below records, on the same kernel, for the
second time: a micro-benchmark on the smallest instance of a workload recommending the opposite of
what the workload wants. Both numbers on this page now name the scene they were measured on, because
that is the only thing that would have caught either.

Memory traffic is the output plus a small per-primitive parameter read. No canvas, no intermediate.

The primitive loop is a RUNTIME loop, not ``tl.static_range``. Unrolling it looked like a small win and
was in fact a large loss, which is worth recording because the measurement that justified it was taken
on the wrong scene. On PushT's 9 primitives the unroll was 7.6% faster (2.96 M vs 2.77 M frames/s at
96px). On a typical scene it was 4x SLOWER: highway-env measured 5,347 f/s unrolled against 21,342 with
the runtime loop at 134 primitives, and Merge 7,182 against 27,618 at 130. Past a few dozen primitives
the unrolled kernel collapses, long before ptxas refuses to compile it at all — which it does above
roughly 340 primitives (340 alone takes 35.6 s; 350 aborts with SIGABRT), so highway-env's 604-primitive
roundabout could not be rendered at any speed.

The lesson generalises past this kernel: a micro-benchmark on the smallest instance of a workload can
recommend the opposite of what the workload wants.

Past a few dozen primitives the loop is also mostly WASTED, and that is what ``_bbox_kernel`` is for.
Every primitive is tested against every pixel, so the cost is ``pixels x primitives`` even though a
highway-env vehicle covers ~200 of a frame's 262,144 pixels. A pre-pass computes each primitive's
world-space bounding box, the pixel block becomes a 2-D TILE rather than a strip of consecutive
addresses, and the loop skips any primitive whose box misses the tile. The skip is a scalar branch, so
what it saves is the whole per-pixel body — and what it costs is four broadcast loads and four
comparisons, which is why it is only switched on above ``_CULL_MIN_PRIMS``. Measured at 256px,
batch 1024, kernel only: highway-v0 189.5 -> 14.5 ms and roundabout-v0 65.2 -> 7.6 ms, i.e. 13.1x
and 8.6x.

The box is a BOUND, not an approximation, so the picture cannot change — with two cases that need
saying out loud.

A polygon whose vertices are collinear has no interior, and the kernel's inside test (``max`` over the
outward edge planes ``<= 0``) then describes a line or a halfplane rather than a polygon: unbounded,
hence not inside any box. Those are marked uncullable by their zero shoelace area and drawn the old
way. ``tests/gpu/test_fast_paths.py`` asserts the two paths agree byte for byte over every
registered env rather than leaving that as an argument.

The second case is not about culling at all and is recorded here because it looks exactly like it.
Two compilations of this kernel can round the polygon SDF differently — ptxas is free to contract
``a*b + c*d`` into an FMA or not, and the choice depends on the surrounding code, which the tile shape
changes. On a pixel whose centre lies within ~3e-5 world units of an edge that decides ``sdf <= 0``.
Measured over intersection-v0 and racetrack-v0 at 600px, batch 1024 (368 M pixels): the culled path
at two different tile shapes agrees EXACTLY, and the unculled path disagrees with ITSELF at two
different block sizes on the same 3 and 1 pixels. So it is one expression compiled twice, not a
bound that is too small — and the production path, which is the culled one above
``_CULL_MIN_PRIMS``, is the stable one.

Anti-aliasing is supersampling: evaluate ``ss x ss`` subpixels per output pixel and box-average. ``ss=1``
is the default because it is what the evidence supports — against gym_pusht, ss=1 gives mean |diff|
2.35/255 with **100% of interior pixels bit-identical**, and it is 1.58x faster than ss=2. Supersampling
buys edge treatment only, and the reference renderers are themselves aliased, so neither is ground truth
at an edge.

The torch path is kept, not deleted. It is the executable specification: it is short enough to read as a
definition of what the picture should be, it runs where Triton does not, and ``tests/gpu`` diffs the two
against each other. A fast kernel with no independent implementation to check it against is a kernel
nobody can trust.
"""
from __future__ import annotations

import torch

from .scene import KIND_DISC, KIND_POLY, KIND_SEGMENT, Scene2D

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                    # reported by the renderer, never silent
    HAVE_TRITON = False


def _pixel_grid(scene: Scene2D, H: int, W: int, device, dtype):
    """Subpixel centres in WORLD coordinates, shape (H,W) each.

    Pixel centres, not corners: a pixel covers ``[i, i+1)`` and samples at ``i+0.5``. Sampling at the
    corner shifts the whole image half a pixel, which is invisible next to a reference render and shows
    up immediately in a pixel diff.
    """
    x0, y0, x1, y1 = scene.world
    sx = (x1 - x0) / W
    sy = (y1 - y0) / H
    cx = x0 + (torch.arange(W, device=device, dtype=dtype) + 0.5) * sx
    cy = y0 + (torch.arange(H, device=device, dtype=dtype) + 0.5) * sy
    if scene.flip_y:
        cy = y1 + y0 - cy
    return cx.view(1, W).expand(H, W), cy.view(H, 1).expand(H, W)


def _prim_plan(scene: Scene2D, Rh: int, Rw: int) -> list[tuple]:
    """Per primitive: what it actually is, and where on the frame it can possibly paint.

    ``render_torch`` used to evaluate all three primitive families over the whole frame for every
    slot — a disc paid for four polygon edges and a capsule, and a wall covering 2% of the picture
    paid for the other 98%. Neither is arithmetic that reaches a pixel; both are B x Rh x Rw
    temporaries. This is the pre-pass that lets the loop skip them, and it is the CPU counterpart of
    what ``_bbox_kernel`` does per tile on the GPU.

    Returned per primitive, all of it decided on the HOST from one transfer:

      ``(disc, ring, poly, outline, seg, edges, r0, r1, c0, c1)``

      * ``disc``/``poly``/``seg`` — this family appears in this slot for at least one environment.
        ``kind`` is per environment, so a slot that is a polygon in every row costs nothing for the
        two families it is not; a slot that is disabled (``kind == 0``) in every row costs nothing
        at all. Skipping a family no row has changes no pixel, because the family's paint is masked
        by ``kind ==`` that family and the mask is empty.
      * ``ring`` / ``outline`` — the inner disc and the polygon outline exist only where
        ``inner_r > 0`` and ``line_w > 0``. Where no row has one, its ``where`` is a no-op over the
        whole frame, and the outline is the expensive half of a polygon: the inside test is one dot
        product per edge and the outline is a projection, a clamp, a rejection vector and a squared
        length.
      * ``edges`` — ``(e, mixed)`` for each edge some environment actually has. Polygons are padded
        to ``V`` by repeating the last vertex, so a triangle in a ``V=8`` builder carries five
        zero-length edges whose contribution the loop computes and then throws away with
        ``torch.where(degen, ...)``; an edge degenerate in EVERY row is dropped outright. ``mixed``
        says some rows are degenerate there and some are not, which is the only case that needs the
        ``where`` at all — and it is rare, because degeneracy comes from padding, which is a
        property of the vertex count rather than of the environment. Where no row is degenerate,
        ``torch.where(False, ..., d)`` is ``d``, and dropping it drops a full window-sized select
        and the ``full_like`` sentinel feeding it, per edge, per polygon.
      * ``r0:r1, c0:c1`` — the pixel window of the primitive's world-space bound, unioned over the
        batch. The BOUND is the encoding ``_bbox_kernel`` uses, deliberately including its two
        special cases: an empty box for ``kind == 0`` and an INFINITE one for a polygon whose twice
        signed area is exactly zero, which has no interior and whose halfplane intersection is
        unbounded. Using the same encoding on both devices means there is one culling rule in the
        repo rather than a second one that nothing tests; ``tests/gpu/test_fast_paths.py`` already
        holds it against every registered env.

    The window is a union over the batch because one torch op covers the whole batch: a primitive
    that is static across environments (a wall, a goal marker, PushT's four table edges) keeps a
    tight window at any batch size, and one that moves per environment widens to what the batch
    spans. That is a throughput property only — a window that is too large evaluates pixels whose
    mask then rejects them, and every pixel inside it is computed from exactly the values it was
    computed from before, so the picture is bit-identical either way. It is also why the plan is
    NOT cached: it is a pure function of the scene handed in, recomputed per call, with no state
    that depends on what was rendered before.
    """
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    k, dt = scene.kind, scene.geom.dtype
    g = scene.geom
    G = int(g.shape[2])
    is_disc, is_poly, is_seg = k == KIND_DISC, k == KIND_POLY, k == KIND_SEGMENT

    # --- what each slot is, and which of its optional parts exist (all (N,) bool) -----------
    rows = [is_disc.any(0), is_poly.any(0), is_seg.any(0),
            (is_disc & (g[:, :, 3] > 0)).any(0) if G > 3 else is_disc.new_zeros(N),
            (is_poly & (scene.line_w > 0)).any(0)]
    # An edge is skippable only if it is degenerate for every environment, and "degenerate" has to
    # be decided by the expression the loop uses, in the loop's dtype, or the plan and the loop
    # could disagree about one edge of one polygon.
    for e in range(V):
        j = (e + 1) % V
        ex = g[:, :, 2 * j] - g[:, :, 2 * e]
        ey = g[:, :, 2 * j + 1] - g[:, :, 2 * e + 1]
        degen = (ex * ex + ey * ey) < 1e-12
        rows.append(~degen.all(0))                       # some row has this edge
    for e in range(V):
        j = (e + 1) % V
        ex = g[:, :, 2 * j] - g[:, :, 2 * e]
        ey = g[:, :, 2 * j + 1] - g[:, :, 2 * e + 1]
        rows.append(((ex * ex + ey * ey) < 1e-12).any(0))   # ... and some row does not

    # --- the world-space bound, exactly as _bbox_kernel encodes it ---------------------------
    f64 = torch.float64
    gg = g.to(f64)
    lw = scene.line_w.abs().to(f64)
    vx, vy = gg[:, :, 0:2 * V:2], gg[:, :, 1:2 * V:2]
    # float64, and then the geom dtype as well: the kernel's ``flat`` is a float32 shoelace
    # tested against exactly zero, and a polygon it calls unbounded must be unbounded here too
    # or the two backends would cull by different rules.
    area2 = (vx * torch.roll(vy, -1, -1) - torch.roll(vx, -1, -1) * vy).sum(-1)
    v32x, v32y = g[:, :, 0:2 * V:2], g[:, :, 1:2 * V:2]
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
    bx0 = torch.where(is_disc, gg[:, :, 0] - r,
                      torch.where(is_poly, poly0x, torch.minimum(gg[:, :, 0], sx1) - hw))
    by0 = torch.where(is_disc, gg[:, :, 1] - r,
                      torch.where(is_poly, poly0y, torch.minimum(gg[:, :, 1], sy1) - hw))
    bx1 = torch.where(is_disc, gg[:, :, 0] + r,
                      torch.where(is_poly, poly1x, torch.maximum(gg[:, :, 0], sx1) + hw))
    by1 = torch.where(is_disc, gg[:, :, 1] + r,
                      torch.where(is_poly, poly1y, torch.maximum(gg[:, :, 1], sy1) + hw))
    off = k == 0
    # NaN geometry is not a bound, so it draws over the whole frame, which is what the unbounded
    # loop did with it. Silently shrinking a window around a NaN is how a bound stops being one.
    bx0 = torch.where(off, inf, bx0).nan_to_num(nan=float("-inf"))
    by0 = torch.where(off, inf, by0).nan_to_num(nan=float("-inf"))
    bx1 = torch.where(off, -inf, bx1).nan_to_num(nan=float("inf"))
    by1 = torch.where(off, -inf, by1).nan_to_num(nan=float("inf"))

    x0, y0, x1, y1 = scene.world
    sx, sy = (x1 - x0) / Rw, (y1 - y0) / Rh
    # Pixel r samples at its CENTRE, y0 + (r+0.5)*sy, or y1 - (r+0.5)*sy under flip_y. One pixel of
    # slack on every side so that no rounding of this conversion can cost a pixel the loop would
    # have painted; one pixel too many costs throughput and nothing else.
    ux0, uy0 = bx0.min(0).values, by0.min(0).values
    ux1, uy1 = bx1.max(0).values, by1.max(0).values
    c0 = torch.floor((ux0 - x0) / sx - 0.5) - 1
    c1 = torch.ceil((ux1 - x0) / sx - 0.5) + 2
    if scene.flip_y:
        r0 = torch.floor((y1 - uy1) / sy - 0.5) - 1
        r1 = torch.ceil((y1 - uy0) / sy - 0.5) + 2
    else:
        r0 = torch.floor((uy0 - y0) / sy - 0.5) - 1
        r1 = torch.ceil((uy1 - y0) / sy - 0.5) + 2
    win = torch.stack([r0.clamp(0, Rh), r1.clamp(0, Rh), c0.clamp(0, Rw), c1.clamp(0, Rw)], dim=1)

    # ONE host transfer for the whole plan. Eleven small ``.cpu()`` calls is how a decision that
    # costs nothing on paper becomes a dozen synchronisations per frame.
    flags = torch.stack(rows, dim=1).to(f64)
    plan = torch.cat([flags, win], dim=1).cpu().tolist()
    out = []
    for row in plan:
        disc, poly, seg, ring, outline = (bool(v) for v in row[:5])
        edges = [(e, bool(row[5 + V + e])) for e in range(V) if row[5 + e]]
        rr0, rr1, cc0, cc1 = (int(v) for v in row[5 + 2 * V:])
        out.append((disc, ring, poly, outline, seg, edges, rr0, rr1, cc0, cc1))
    return out


@torch.no_grad()
def render_torch(scene: Scene2D, out_size, supersample: int = 1) -> torch.Tensor:
    """Reference implementation: ``Scene2D`` -> ``(B,H,W,3)`` uint8. Readable, not fast.

    Still the executable specification — every per-pixel expression below is the one it always was,
    character for character, and it is still what ``tests/gpu`` diffs the fused kernel against. What
    changed in Step 6 is not the arithmetic but WHERE it is evaluated, and that is decided entirely
    by ``_prim_plan``:

      * a slot is evaluated only for the primitive families some environment actually has there;
      * a polygon's outline and a disc's inner ring are evaluated only where they exist;
      * padded, zero-length polygon edges are skipped;
      * every family is evaluated only inside the pixel window of its own world-space bound.

    All four are bounds and masks that were already being computed and thrown away, so the picture
    is bit-identical; ``scratch/step6/raster2d/cpu_digest.py`` and the frozen Step 4 replay hold
    that rather than leaving it as an argument. Measured on Oscar's batch partition at one thread,
    the whole-frame version cost 324 ms for one 500 px SimplePointMaze frame, of which 13 of the 13
    primitives paid for a capsule test the scene contains none of.

    The uint8 canvas is the second half. With ``supersample == 1`` nothing between a primitive's
    colour and the output is an average, so ``round -> clamp -> uint8`` can be applied to the
    (B,3) colours once instead of to a B x 3 x Rh x Rw float canvas at the end, and the canvas can
    be the output's own (B,Rh,Rw,3) layout. That removes the float canvas (four bytes a channel
    instead of one), the round, the clamp, the cast and the final permute+copy: five passes over
    the frame stack. Supersampling genuinely averages, so ``ss > 1`` keeps the float canvas and the
    conversion it always had.
    """
    H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
    if not isinstance(out_size, int):
        H, W = out_size
    ss = int(supersample)
    Rh, Rw = H * ss, W * ss
    dev, dt = scene.kind.device, scene.geom.dtype
    px, py = _pixel_grid(scene, Rh, Rw, dev, dt)
    B, N, V = scene.batch, scene.n_prims, scene.vertices

    flat_out = ss == 1
    if flat_out:
        u8 = lambda c: c.round().clamp(0, 255).to(torch.uint8).view(B, 1, 1, 3)   # noqa: E731
        canvas = u8(scene.background).expand(B, Rh, Rw, 3).clone()
    else:
        u8 = lambda c: c.view(B, 3, 1, 1)                                         # noqa: E731
        canvas = scene.background.view(B, 3, 1, 1).expand(B, 3, Rh, Rw).clone()

    for i, (has_disc, has_ring, has_poly, has_outline, has_seg, edges,
            r0, r1, c0, c1) in enumerate(_prim_plan(scene, Rh, Rw)):
        if r1 <= r0 or c1 <= c0 or not (has_disc or has_poly or has_seg):
            continue                       # disabled everywhere, or its bound misses the frame
        kind = scene.kind[:, i]                                   # (B,)
        g = scene.geom[:, i]                                      # (B,2V)
        fill = u8(scene.fill[:, i])
        line = u8(scene.line[:, i])
        lw = scene.line_w[:, i].view(B, 1, 1)
        p = px[r0:r1, c0:c1].unsqueeze(0)
        q = py[r0:r1, c0:c1].unsqueeze(0)
        view = ((lambda: canvas[:, r0:r1, c0:c1, :]) if flat_out
                else (lambda: canvas[:, :, r0:r1, c0:c1]))
        mdim = -1 if flat_out else 1

        def paint(hit, colour, _v=view, _m=mdim):
            """``canvas = torch.where(hit, colour, canvas)``, restricted to this primitive's window.

            In place through the window rather than by rebinding a whole new canvas: the old loop
            allocated and wrote a full B x 3 x Rh x Rw tensor for EVERY ``where``, which at batch 64
            and 500 px is 192 MB per primitive per paint, and then threw the old one away.

            Through a temporary and a ``copy_`` rather than ``out=`` the view it also reads. An
            aliased ``out=`` measured the same to within the noise of the machine it was tried on
            (46 ms against 58 on an isolated paint, indistinguishable in the renderer), and this is
            the file the fused kernel is checked against -- a destination that is also an input is a
            subtle thing to put in an executable specification for no measured gain.
            """
            w = _v()
            w.copy_(torch.where(hit.unsqueeze(_m), colour, w))

        # disc (with optional inner disc)
        if has_disc:
            d2 = (p - g[:, 0].view(B, 1, 1)) ** 2 + (q - g[:, 1].view(B, 1, 1)) ** 2
            is_disc = (kind == KIND_DISC).view(B, 1, 1)
            paint(is_disc & (d2 <= g[:, 2].view(B, 1, 1) ** 2), fill)
            if has_ring:
                paint(is_disc & (d2 <= g[:, 3].view(B, 1, 1) ** 2) & (g[:, 3].view(B, 1, 1) > 0),
                      line)

        # convex polygon: inside = max over outward edge planes <= 0; outline = union of edge capsules
        if has_poly:
            h, w = r1 - r0, c1 - c0
            sdf = torch.full((B, h, w), -1e30, device=dev, dtype=dt)
            edge_d2 = (torch.full((B, h, w), 1e30, device=dev, dtype=dt) if has_outline else None)
            for e, mixed in edges:
                ax, ay = g[:, 2 * e].view(B, 1, 1), g[:, 2 * e + 1].view(B, 1, 1)
                j = (e + 1) % V
                bx, by = g[:, 2 * j].view(B, 1, 1), g[:, 2 * j + 1].view(B, 1, 1)
                ex, ey = bx - ax, by - ay
                L2 = ex * ex + ey * ey
                degen = L2 < 1e-12
                inv = torch.rsqrt(L2.clamp_min(1e-20))
                d = (p - ax) * (ey * inv) + (q - ay) * (-ex * inv)
                sdf = torch.maximum(sdf, torch.where(degen, torch.full_like(d, -1e30), d)
                                    if mixed else d)
                if has_outline:
                    t = (((p - ax) * ex + (q - ay) * ey) / L2.clamp_min(1e-20)).clamp(0, 1)
                    qx, qy = p - ax - t * ex, q - ay - t * ey
                    e2 = qx * qx + qy * qy
                    edge_d2 = torch.minimum(
                        edge_d2, torch.where(degen, torch.full_like(qx, 1e30), e2) if mixed else e2)
            is_poly = (kind == KIND_POLY).view(B, 1, 1)
            paint(is_poly & (sdf <= 0), fill)
            if has_outline:
                paint(is_poly & (lw > 0) & (edge_d2 <= lw * lw), line)

        # segment: capsule, distance to segment <= half width
        if has_seg:
            ax, ay = g[:, 0].view(B, 1, 1), g[:, 1].view(B, 1, 1)
            ex, ey = g[:, 2].view(B, 1, 1) - ax, g[:, 3].view(B, 1, 1) - ay
            t = (((p - ax) * ex + (q - ay) * ey) / (ex * ex + ey * ey).clamp_min(1e-20)).clamp(0, 1)
            dx, dy = p - ax - t * ex, q - ay - t * ey
            paint((kind == KIND_SEGMENT).view(B, 1, 1)
                  & (dx * dx + dy * dy <= g[:, 4].view(B, 1, 1) ** 2), fill)

    if flat_out:
        return canvas
    if ss > 1:
        canvas = torch.nn.functional.avg_pool2d(canvas, ss)
    return canvas.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


if HAVE_TRITON:

    @triton.jit
    def _bbox_kernel(kind_ptr, geom_ptr, lw_ptr, box_ptr, M, V: tl.constexpr, BLOCK: tl.constexpr):
        """``(B*N,4)`` world-space bounds, one row per primitive: ``x0, y0, x1, y1``.

        Three encodings, and only the third is a special case:

          * a real bound for a disc (centre +- the larger of the two radii), a capsule (the endpoints
            grown by the half-width) and a convex polygon (the vertex hull grown by the outline
            half-width, since the outline is a union of capsules ALONG the edges);
          * an EMPTY box for ``kind == 0``, so a disabled primitive is skipped by the same test that
            skips a distant one rather than by a second branch;
          * an INFINITE box where the box would not be a bound. That is the collinear polygon: with
            every vertex on one line, the halfplane intersection the kernel tests is that whole line
            (or, for two live edges pointing the same way, a halfplane), and it paints pixels
            arbitrarily far from the vertices. Twice the signed area is exactly zero there and exactly
            zero is what is tested — a proper polygon whose area merely UNDERFLOWS to zero also lands
            in this branch and is drawn the slow way, which costs throughput and not a pixel.
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
        if 2 * V > 4:                        # a capsule's half-width; V is 3 or more in every env
            g4 = tl.load(gb + 4, mask=m, other=0.0)
        else:
            g4 = tl.zeros([BLOCK], dtype=tl.float32)

        # disc: the fill is |r| and the ring is drawn only where the inner radius is positive
        r = tl.maximum(tl.abs(g2), tl.maximum(g3, 0.0))
        # segment: distance to the segment <= half width
        hw = tl.abs(g4)
        # polygon: running hull over the padded vertex list, plus the shoelace
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
        x0 = tl.where(k == 1, g0 - r, tl.where(k == 2, tl.where(flat, -INF, px0 - lw),
                                               tl.minimum(g0, g2) - hw))
        y0 = tl.where(k == 1, g1 - r, tl.where(k == 2, tl.where(flat, -INF, py0 - lw),
                                               tl.minimum(g1, g3) - hw))
        x1 = tl.where(k == 1, g0 + r, tl.where(k == 2, tl.where(flat, INF, px1 + lw),
                                               tl.maximum(g0, g2) + hw))
        y1 = tl.where(k == 1, g1 + r, tl.where(k == 2, tl.where(flat, INF, py1 + lw),
                                               tl.maximum(g1, g3) + hw))
        off = k == 0
        ob = box_ptr + offs * 4
        tl.store(ob + 0, tl.where(off, INF, x0), mask=m)
        tl.store(ob + 1, tl.where(off, INF, y0), mask=m)
        tl.store(ob + 2, tl.where(off, -INF, x1), mask=m)
        tl.store(ob + 3, tl.where(off, -INF, y1), mask=m)

    @triton.jit
    def _scene_kernel(kind_ptr, geom_ptr, fill_ptr, line_ptr, lw_ptr, bg_ptr, base_ptr, box_ptr,
                      out_ptr, N, PSTART,
                      H: tl.constexpr, W: tl.constexpr, SS: tl.constexpr, V: tl.constexpr,
                      wx0, wy0, sx, sy, FLIP: tl.constexpr, y_hi, HAS_BASE: tl.constexpr,
                      CULL: tl.constexpr, NTX: tl.constexpr,
                      BH: tl.constexpr, BW: tl.constexpr):
        b = tl.program_id(0)
        tid = tl.program_id(1)
        npix = H * W
        if CULL:
            # A 2-D tile, not a run of consecutive pixel addresses. Both cover the same number of
            # pixels, and only the 2-D one has a compact footprint in the plane -- which is the whole
            # point when the loop below is skipping primitives that miss it. A 1x1024 strip of a
            # 1024-wide frame is missed by nothing.
            tiy = tid // NTX
            tix = tid % NTX
            oy = tiy * BH + tl.arange(0, BH)[:, None]
            ox = tix * BW + tl.arange(0, BW)[None, :]
            m = (oy < H) & (ox < W)
            # The tile in world coordinates, as four scalars, taken to whole PIXEL boundaries so that
            # every subsample this program evaluates lies inside it whatever SS is.
            bx_lo = wx0 + (tix * BW * SS) * sx
            bx_hi = wx0 + ((tix + 1) * BW * SS) * sx
            by_lo = wy0 + (tiy * BH * SS) * sy
            by_hi = wy0 + ((tiy + 1) * BH * SS) * sy
            if FLIP:
                by_lo, by_hi = y_hi - by_hi, y_hi - by_lo
        else:
            # One run of consecutive pixel ADDRESSES, which is what this kernel had before tiling
            # existed and is what it goes back to when there is no primitive list to skip. A run may
            # straddle the end of a row, so no lane is ever masked for the sake of a rectangle, and
            # BW consecutive pixels are BW*3 consecutive output bytes. Tiling a frame whose width is
            # not a multiple of BW instead costs either masked lanes or a narrower store: at PushT's
            # 96 px the best rectangle is 4x32, and 96 contiguous bytes per row rather than 384 is
            # worth 7% of the whole render.
            pos = tid * BW + tl.arange(0, BW)[None, :] + tl.zeros([BH, 1], dtype=tl.int32)
            m = pos < npix
            oy = pos // W
            ox = pos % W
            bx_lo, bx_hi, by_lo, by_hi = wx0, wx0, wy0, wy0        # never read; CULL is False
        offs = oy * W + ox
        # 64-bit output offset. The frame base is b * H * W * 3, which overflows int32 well inside the
        # batch sizes this repo exists to serve: at 224px, b = 16383 gives 16383*50176*3 = 2.47e9 against
        # an int32 ceiling of 2.15e9. It surfaced as `CUDA error: an illegal memory access`, i.e. a
        # crash rather than silent corruption, but only because the wrapped address happened to land
        # outside the allocation — a smaller overflow would have quietly written into another tensor.
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

                # Broadcast to the tile width immediately. The background loads as a scalar, but every
                # primitive branch assigns a tile-wide vector, and Triton requires a variable's type to
                # be invariant across a conditional — a scalar initialiser makes the first `elif` fail
                # to compile with "initial value is fp32[] but the then block redefines it as fp32[128]".
                zero = tl.zeros([BH, BW], dtype=tl.float32)
                if HAS_BASE:
                    # The static prefix was rasterized once at full supersample resolution, so this is
                    # an exact substitute for re-evaluating those primitives, not an approximation.
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
                    # One scalar branch decides whether this program evaluates the primitive at all.
                    # ``_bbox_kernel`` gives a disabled primitive an EMPTY box, so the culled path
                    # needs no separate ``kind == 0`` test and does not read ``kind`` for a primitive
                    # it is about to skip.
                    if CULL:
                        bb = box_ptr + pid * 4
                        live = ((tl.load(bb + 2) >= bx_lo) & (tl.load(bb + 0) <= bx_hi)
                                & (tl.load(bb + 3) >= by_lo) & (tl.load(bb + 1) <= by_hi))
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

                        if k == 1:                                   # disc (+ optional inner disc)
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

                        elif k == 2:                                 # convex polygon
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
                                # a padded (repeated) vertex makes a zero-length edge whose normal is
                                # undefined; including it would pin the sdf at 0 and paint everything
                                degen = L2 < 1e-12
                                sdf = tl.maximum(sdf, tl.where(degen, -1e30, d))
                                # Outline = union of CAPSULES along the edges, i.e. distance to the
                                # SEGMENT, not the |sdf| band. The band mitres the corners; pygame's
                                # draw_fat_segment explicitly paints a disc of the stroke radius at each
                                # endpoint, so the reference has ROUND corners. Measured: the band costs
                                # 2.2% of interior pixels on PushT (97.8% vs 100% exact), all of it at
                                # the eight corners of the T's two quads.
                                wxx = px - ax
                                wyy = py - ay
                                t = (wxx * ex + wyy * ey) / (L2 + 1e-20)
                                t = tl.minimum(tl.maximum(t, 0.0), 1.0)
                                qx = wxx - t * ex
                                qy = wyy - t * ey
                                edge_d2 = tl.minimum(edge_d2,
                                                     tl.where(degen, 1e30, qx * qx + qy * qy))
                            hit = sdf <= 0.0
                            cr = tl.where(hit, fr, cr)
                            cg = tl.where(hit, fg, cg)
                            cb = tl.where(hit, fb, cb)
                            hit = (lw > 0) & (edge_d2 <= lw * lw)
                            cr = tl.where(hit, lr, cr)
                            cg = tl.where(hit, lg, cg)
                            cb = tl.where(hit, lb, cb)

                        else:                                        # segment (capsule)
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


#: Cache of baked static-prefix images, keyed by (resolution, the static primitives' exact bytes).
#: Keyed on VALUES, not on the env object: a variation that recolours the goal must invalidate the bake,
#: and an identity keyed on the env would silently keep serving the old background.
_BASE_CACHE: dict = {}
_BASE_CACHE_MAX = 32

#: Subpixel-primitive evaluations the bake must save before it is worth asking whether it may run.
#: Deciding costs one device-to-host synchronisation per call (0.08-0.29 ms, flat in batch); the bake
#: saves roughly ``K/N`` of the rasterizer's work, i.e. ``K/N * B * Rh * Rw`` subpixel evaluations. The
#: two cross somewhere, and where they cross was measured on three cells that differ by 27x in pixels
#: and by 1.4x in the fraction of the scene that is static:
#:
#:     PushT 96px  (K/N = 6/9)   end-to-end 0.94x at B=256, 1.01x at B=1024, 1.23x at B=4096
#:     PushT 224px (K/N = 6/9)   0.94x at B=64, 1.04x at B=256, 1.77x at B=4096
#:     SimplePointMaze 500px (K/N = 6/13)   1.02x at B=64, 1.30x at B=256, 1.69x at B=1024
#:
#: All three break even at ``K/N * B * Rh * Rw`` between 6e6 and 9e6, which is what makes this a
#: measured constant rather than a tuned one: the same number predicts the crossover for a 9,216-pixel
#: frame and a 250,000-pixel one. Below it the request is dropped WITHOUT synchronising, so a small
#: batch pays nothing at all for a prefix it cannot use — and the batch-1 case, which is a loop of
#: single renders and the worst case for any per-call overhead, is the furthest below it.
_BAKE_MIN_WORK = 8e6


def _row_bytes(t: torch.Tensor, B: int) -> torch.Tensor:
    """``(B, ...)`` of any dtype as ``(B, nbytes)`` uint8, so unlike arrays compare in one kernel."""
    return t.contiguous().reshape(B, -1).view(torch.uint8)


def _static_base(scene: Scene2D, Rh: int, Rw: int, block: int):
    """Bake the leading primitives once, or decline; returns an ``(Rh,Rw,3)`` uint8 image or ``None``.

    Two reasons to decline, and neither is a failure mode — the caller draws every primitive instead
    and gets the same picture, byte for byte:

      * **the prefix is not shared.** Painting the whole batch with row 0 of it is the second of the
        three places this repo has collapsed a per-environment quantity to environment 0 (see
        ``core/style.py``), so sharedness is MEASURED — on ``K`` primitives, not on pixels — and a
        prefix that differs across the batch is not baked. That is exactly what a batch of 4,096
        different backgrounds must do.
      * **the bake cannot earn its own decision cost.** See ``_BAKE_MIN_WORK``: below it the request is
        dropped before anything synchronises, so a small batch pays nothing for a prefix it cannot use.

    Measuring costs nothing extra because the cache key needs the same bytes on the host anyway. The
    six arrays are reinterpreted as raw bytes and concatenated into one ``(B, M)`` block, so the whole
    question — "do the rows agree, and what is row 0" — is one comparison, one reduction and ONE
    device-to-host transfer. It used to be eleven ``.cpu()`` calls and six ``bool(...)`` reductions,
    i.e. twelve synchronisations, a flat 0.177 ms per render against the 0.047 ms it costs to
    rasterize all nine of PushT's primitives at batch 1. "The bake is a 4x loss at batch 1" was that
    transfer, not the bake.

    Bytes rather than values is deliberate on both counts: it is what makes one comparison serve five
    dtypes, and byte-equality is the right notion here anyway, since the cache is keyed on the same
    bytes and two values that are equal but differently encoded are two different cache entries.
    """
    K = int(scene.static_prefix)
    if K <= 0:
        return None
    B = scene.batch
    if K * B * Rh * Rw < _BAKE_MIN_WORK * scene.n_prims:
        return None                                      # too little work to earn the synchronisation
    rows = torch.cat([_row_bytes(t, B) for t in
                      (scene.kind[:, :K], scene.geom[:, :K], scene.fill[:, :K], scene.line[:, :K],
                       scene.line_w[:, :K], scene.background)], dim=1)          # (B, M) uint8
    row0 = rows[0]
    if B > 1:
        ok = (rows == rows[:1]).all().reshape(1).to(torch.uint8)
        row0 = torch.cat([row0, ok])
    blob = row0.cpu().numpy().tobytes()                   # the one synchronisation
    if B > 1:
        blob, uniform = blob[:-1], blob[-1]
        if not uniform:
            return None                                  # per-environment prefix: draw it per frame
    # The device is part of the key. Without it, a process that renders the same scene on two GPUs --
    # DataParallel, or any manual multi-device script -- gets back a plate that lives on the other one,
    # and Triton refuses the launch with "Pointer argument (at 0) cannot be accessed". DDP never sees
    # this, because one process per rank means one cache per rank; that is why it went unnoticed.
    key = (Rh, Rw, scene.world, scene.flip_y, scene.vertices, K, str(scene.kind.device), blob)
    hit = _BASE_CACHE.get(key)
    if hit is None:
        prefix = Scene2D(kind=scene.kind[:1, :K].contiguous(), geom=scene.geom[:1, :K].contiguous(),
                         fill=scene.fill[:1, :K].contiguous(), line=scene.line[:1, :K].contiguous(),
                         line_w=scene.line_w[:1, :K].contiguous(),
                         background=scene.background[:1].contiguous(),
                         world=scene.world, flip_y=scene.flip_y, static_prefix=0)
        hit = render_triton(prefix, (Rh, Rw), supersample=1, block=block)[0].contiguous()
        if len(_BASE_CACHE) >= _BASE_CACHE_MAX:
            _BASE_CACHE.clear()
        _BASE_CACHE[key] = hit
    return hit


def _destination(out, B: int, H: int, W: int) -> torch.Tensor:
    """The ``(B,H,W,3)`` uint8 buffer the kernel writes, checked rather than assumed.

    ``out.view(B,H,W,3)`` is not a check. When the requested shape is the shape the tensor already
    has, ``view`` returns the tensor with its strides intact, so a strided destination came back
    looking exactly like a contiguous one and the kernel then wrote ``B*H*W*3`` bytes linearly from
    its offset — the frames scattered across the caller's buffer, 255/255 wrong, no error. Both halves
    of the destination contract are asserted here, at the one place every 2-D render reaches, so a
    caller cannot land in the silent case from any entry point.
    """
    if out is None:
        return None
    if not torch.is_tensor(out):
        raise TypeError(f"render(out=) expects a torch tensor, got {type(out).__name__}")
    if not out.is_contiguous():
        raise ValueError(
            "render(out=): the destination must be contiguous. The kernel writes the frames linearly "
            "through a flat view of it, and a view of a strided tensor keeps its strides, so the "
            "frames would be scattered across your buffer rather than written into it. `buf[:, t]` of "
            "a (B, T, H, W, 3) rollout buffer is the usual way to reach this; assign into it instead "
            "(`buf[:, t] = render(...)`), or lay the buffer out with the frame axis outermost.")
    if out.numel() != B * H * W * 3:
        raise ValueError(
            f"render(out=): the destination holds {out.numel()} elements, and the frames need "
            f"{B * H * W * 3} ({B}x{H}x{W}x3).")
    if out.dtype != torch.uint8:
        raise ValueError(f"render(out=): the destination must be uint8, got {out.dtype}")
    return out.view(B, H, W, 3)


#: Primitives above which a scene is binned against screen tiles before it is drawn.
#: The trade is per BLOCK, not per frame: culling turns ``N`` full evaluations into ``N`` box tests
#: plus ``k`` full evaluations, where ``k`` is how many primitives reach one tile. A box test is ~13
#: instructions against ~110 for a quad, so it pays from about ``N > 8k`` — and ``k`` does not fall
#: below 1 (there is always a background-sized primitive somewhere). 24 is where the measured curves
#: cross for PushT-sized scenes; below it the pre-pass launch alone (~15 us, flat) is most of a
#: batch-1 render.
_CULL_MIN_PRIMS = 24

#: ``(BH, BW, num_warps)`` for the culled path: a WIDE tile of 1,024 pixels.
#:
#: Two forces pull against each other and the measurement settles which one wins. A tile's primitive
#: list grows with its perimeter while its work grows with its area, which argues for a square; but
#: the output is ``(B,H,W,3)`` uint8 and the three stores are strided by 3, so a narrow tile scatters
#: its writes. Measured over the highway family at 256px, batch 1024 (kernel only, ms, best of 11,
#: roundabout-v0 / highway-v0):
#:
#:     1x128   42.9 / 74.1        16x16   21.3 / 38.2        32x32   9.0 / 19.9
#:     32x16   12.8 / 26.3        16x32   11.9 / 22.3        16x64   8.0 / 15.8
#:     8x128    7.6 / 14.5
#:
#: i.e. wide wins at equal area, and 8x128 is 8.6x the unbinned kernel on roundabout-v0 and 13.1x on
#: highway-v0. num_warps 4 rather than 8 throughout: 8 was 1.5-1.6x SLOWER at every shape tried,
#: which is what a kernel with this much per-primitive scalar work looks like when the scalars stop
#: being amortised over enough lanes per warp.
_CULL_TILE = (8, 128, 4)

#: Fraction of a tile's lanes that may be masked off before a narrower tile is preferred. A tile that
#: hangs off the right edge still runs the whole primitive loop for the lanes it masks, so at 96 px a
#: flat 128-wide tile would pay for 33% more pixels than the frame has; at 600 px it pays for 6.7%,
#: which is cheaper than the scattered stores of a tile narrow enough to divide 600.
_TILE_WASTE_MAX = 0.125


def _fit(target: int, W: int) -> int:
    """Tile width: the widest power of two at most ``target`` that does not waste too many lanes."""
    cap = min(target, 1 << max(0, (W - 1).bit_length()))
    bw = cap
    while bw > 1 and (bw * -(-W // bw) - W) > _TILE_WASTE_MAX * W:
        bw //= 2
    return bw


def _tiles(H: int, W: int, block: int, cull: bool):
    """``(BH, BW, n_tiles, n_tiles_x, num_warps)`` for one frame geometry.

    Only the culled path is TILED. Without a primitive list to skip there is nothing for a compact
    footprint to buy, and a rectangle then costs something instead -- masked lanes where its width
    does not divide the frame's, and a shorter contiguous store where it does. So the unculled path
    keeps ``BH = 1`` and a linear run, which the kernel reads as consecutive addresses rather than as
    a row of a rectangle.
    """
    if not cull:
        return 1, block, triton.cdiv(H * W, block), 1, 4
    bh, bw, warps = _CULL_TILE
    bw = _fit(bw, W)
    bh = max(1, min((_CULL_TILE[0] * _CULL_TILE[1]) // bw, 1 << max(0, (H - 1).bit_length())))
    ntx = triton.cdiv(W, bw)
    return bh, bw, ntx * triton.cdiv(H, bh), ntx, warps


@torch.no_grad()
def render_triton(scene: Scene2D, out_size, supersample: int = 1, out=None,
                  block: int = 128, cull: bool | None = None) -> torch.Tensor:
    """Fused path: ``Scene2D`` -> ``(B,H,W,3)`` uint8, one launch, nothing materialized but the output.

    ``cull`` forces the tile-binning pre-pass on or off; ``None`` decides from the primitive count.
    It is an argument so that ``tests/gpu/test_fast_paths.py`` can render the same scene both ways
    and diff them, which is the only thing that makes the fast path safe to enable by default.
    """
    if not HAVE_TRITON:
        raise ImportError("triton unavailable; use render_torch")
    H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
    if not isinstance(out_size, int):
        H, W = out_size
    ss = int(supersample)
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    x0, y0, x1, y1 = scene.world
    buf = _destination(out, B, H, W)
    if buf is None:
        buf = torch.empty((B, H, W, 3), dtype=torch.uint8, device=scene.kind.device)
    do_cull = (N >= _CULL_MIN_PRIMS) if cull is None else bool(cull)
    # Launch on the device the SCENE lives on, not on whatever the process last called
    # ``set_device`` with. Triton takes its stream and context from the current device, so a scene on
    # cuda:1 rendered from a process whose current device is cuda:0 dies with "Pointer argument (at 0)
    # cannot be accessed from Triton (cpu tensor?)" -- a message that names neither the device nor the
    # real cause. One process per rank hides this completely, which is why DDP passes and a
    # single-process multi-GPU script does not.
    with torch.cuda.device(scene.kind.device):
        base = _static_base(scene, H * ss, W * ss, block)
        dummy = buf                   # unused pointer when a flag is False; Triton needs an argument
        box = dummy
        if do_cull:
            M = B * N
            box = torch.empty((M, 4), dtype=scene.geom.dtype, device=scene.kind.device)
            _bbox_kernel[(triton.cdiv(M, 256),)](scene.kind, scene.geom, scene.line_w, box, M,
                                                 V=V, BLOCK=256)
        bh, bw, nt, ntx, warps = _tiles(H, W, block, do_cull)
        _scene_kernel[(B, nt)](
            scene.kind, scene.geom, scene.fill, scene.line, scene.line_w, scene.background,
            base if base is not None else dummy, box, buf,
            N, int(scene.static_prefix) if base is not None else 0,
            H=H, W=W, SS=ss, V=V,
            wx0=x0, wy0=y0, sx=(x1 - x0) / (W * ss), sy=(y1 - y0) / (H * ss),
            FLIP=bool(scene.flip_y), y_hi=(y0 + y1), HAS_BASE=base is not None,
            CULL=do_cull, NTX=ntx, BH=bh, BW=bw, num_warps=warps,
        )
    return buf


@torch.no_grad()
def render(scene: Scene2D, out_size, supersample: int = 1, backend: str = "auto", out=None):
    """``auto`` prefers Triton and falls back to torch. Both produce the same picture by construction.

    ``auto`` is a question about THIS SCENE, not about this host. It used to be answered by
    ``HAVE_TRITON`` alone, which is a property of the interpreter: on a machine where ``import
    triton`` succeeds, a ``device="cpu"`` scene was routed into ``render_triton``, whose first act is
    ``with torch.cuda.device(scene.kind.device)`` — so every CPU render of a Scene2D env died with
    ``ValueError: Expected a cuda device, but got: cpu``. The laptop that measured the Step 3 CPU rows
    had no Triton and never saw it; the certification cluster has Triton in every venv and could not
    render PushT or SimplePointMaze on CPU at all. Triton compiles for CUDA here, so the predicate is
    "Triton is importable AND the scene lives on a CUDA device", and ``device="cpu"`` is honoured on
    every host.

    An explicit ``backend="triton"`` still raises on a CPU scene, and should: the caller named a
    backend that cannot draw the scene it was handed, and silently drawing it somewhere else is how a
    ``device="cpu"`` run ends up on a GPU.
    """
    if backend == "triton" and scene.kind.device.type != "cuda":
        raise ValueError(
            f"render(backend='triton') on a scene that lives on {scene.kind.device}. The fused "
            f"kernel is CUDA-only; use backend='torch', or build the scene on a cuda device.")
    if backend == "torch" or (backend == "auto"
                              and not (HAVE_TRITON and scene.kind.device.type == "cuda")):
        H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
        if not isinstance(out_size, int):
            H, W = out_size
        # The same destination contract as the fused path, deliberately. ``copy_`` would happily fill a
        # strided buffer, so the torch backend could accept destinations the Triton one refuses — and
        # then a caller's code would work under one backend and silently corrupt under the other,
        # which is worse than either behaviour on its own.
        buf = _destination(out, scene.batch, H, W)
        r = render_torch(scene, out_size, supersample)
        if buf is not None:
            buf.copy_(r)
            return buf
        return r
    return render_triton(scene, out_size, supersample, out=out)
