"""The fused polygon rasterizer must agree with ``gfxdraw.py`` and ``carracing._draw_poly_mask``.

Those two are the executable specification: each rule in them was recovered by pixel-diffing pygame
and each is short enough to read as a definition. The fused kernel is a rewrite of the same rules
into a closed form per pixel, and a closed form is exactly the kind of rewrite that is right almost
everywhere -- the first version of it differed on ONE pixel in 19.2 million, because Triton's float
division is free to use an approximate reciprocal and a crossing that lands on an exact integer then
floors to the wrong column. So the two paths are diffed on random polygons rather than on the states
one env happens to reach.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

from _shared import assert_frames_identical


def _poly_mask():
    """``carracing._draw_poly_mask``, or a skip naming why it is unavailable.

    The reference this file diffs against lives in ``envs/carracing.py``, and importing that module
    DIRECTLY bypasses the guarded discovery in ``envs/__init__.py``: CarRacing declares Box2D and pygame
    as optional, so on an interpreter without them the discovery records an entry in ``import_errors()``
    and simply leaves the env out of the registry, while a direct import re-raises. That is what made
    this file the only red mark in a 2,259-test suite run — four failures that say nothing about the
    kernel, which is pure torch and needs neither dependency.

    A missing optional dependency must skip, never fail. This project has now been bitten from both
    sides of that rule: a suite that stayed GREEN while collecting zero DM Control cases because the
    module could not import, and this one going RED for a dependency the code under test does not use.
    Both halves of the rule now live in ``tests/gpu/_shared.py`` — ``make_or_skip`` for the first,
    ``resolve_family`` for the second — and this is the one place that needs neither, because what it
    reaches for is a private symbol rather than a module or an env.
    """
    try:
        from stable_envs.envs.carracing import _draw_poly_mask
    except ImportError as e:  # noqa: BLE001
        pytest.skip(
            f"carracing's reference rasterizer is unavailable here ({e}); the fused kernel "
            f"under test is pure torch and does not need it"
        )
    return _draw_poly_mask


H, W = 256, 640


def _tables(verts, colours, nvs, n_gfx):
    """``(vert, meta, edge)`` for ``render_surface`` from ``(B,P,V,2)`` float64 screen vertices."""
    from stable_envs.render import gfxdraw_fused as F

    B, P, V, _ = verts.shape
    vert = torch.trunc(verts).to(torch.int32)
    vx, vy = vert[..., 0], vert[..., 1]
    nv = torch.as_tensor(nvs, dtype=torch.int32, device=verts.device).view(1, P)
    live = torch.arange(V, device=verts.device).view(1, 1, V) < nv.unsqueeze(-1)
    BIG = torch.iinfo(torch.int32).max // 4
    x0 = torch.where(live, vx, torch.full_like(vx, BIG)).amin(-1) - 1
    x1 = torch.where(live, vx, torch.full_like(vx, -BIG)).amax(-1) + 1
    y0 = torch.where(live, vy, torch.full_like(vy, BIG)).amin(-1) - 1
    y1 = torch.where(live, vy, torch.full_like(vy, -BIG)).amax(-1) + 1
    col = colours.to(torch.int32)
    meta = torch.stack(
        [nv.expand(B, P), col[..., 0], col[..., 1], col[..., 2], x0, H - 1 - y1, x1, H - 1 - y0], -1
    ).contiguous()
    edge = F.edge_table(
        vx[:, :n_gfx, :4].reshape(-1, 4).to(torch.int64), vy[:, :n_gfx, :4].reshape(-1, 4).to(torch.int64), H, W
    )
    return vert.contiguous(), meta, edge.reshape(B, n_gfx, 4, F.EG_W)


def _reference(verts, colours, nvs, n_gfx):
    """``gfxdraw.polygon`` for the first ``n_gfx`` and ``_draw_poly_mask`` for the rest, in order."""
    _draw_poly_mask = _poly_mask()
    from stable_envs.render import gfxdraw as G

    B, P = verts.shape[0], verts.shape[1]
    gfx = G.Gfx(H, W, verts.device)
    cv = G.StyleCanvas(gfx, B, torch.zeros(B, 3, device=verts.device))
    cv.aa = torch.ones(B, dtype=torch.bool, device=verts.device)
    bi = gfx.batch_index(B)
    for i in range(P):
        v = verts[:, i, : nvs[i]]
        cv.use(colours[:, i])
        cv.gated(torch.ones(B, dtype=torch.bool, device=verts.device))
        if i < n_gfx:
            G.polygon(cv, v, None, None, bi, 512, True)
        else:
            cv.paint(_draw_poly_mask(v, cv.ys, cv.xs))
    # ``rgb()`` and not ``flip(cv.buf)``: the canvas is planar ``(3,B,H,W)`` and the flip that turns
    # the pre-flip surface into the caller's frame is its job, not this test's. Reaching past it
    # coupled the test to a storage layout it does not test.
    return cv.rgb()


def _random_polys(rng, B, P, dev, radius=40.0):
    """Convex quads at random poses, kept away from degenerate zero-area cases."""
    ang = torch.as_tensor(rng.uniform(0, 2 * np.pi, (B, P)), dtype=torch.float64, device=dev)
    cx = torch.as_tensor(rng.uniform(-20, W + 20, (B, P)), dtype=torch.float64, device=dev)
    cy = torch.as_tensor(rng.uniform(-20, H + 20, (B, P)), dtype=torch.float64, device=dev)
    rx = torch.as_tensor(rng.uniform(6, radius, (B, P)), dtype=torch.float64, device=dev)
    ry = torch.as_tensor(rng.uniform(6, radius, (B, P)), dtype=torch.float64, device=dev)
    base = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64, device=dev)
    c, s = torch.cos(ang), torch.sin(ang)
    bx = base[:, 0].view(1, 1, 4) * rx.unsqueeze(-1)
    by = base[:, 1].view(1, 1, 4) * ry.unsqueeze(-1)
    x = cx.unsqueeze(-1) + bx * c.unsqueeze(-1) - by * s.unsqueeze(-1)
    y = cy.unsqueeze(-1) + bx * s.unsqueeze(-1) + by * c.unsqueeze(-1)
    return torch.stack([x, y], -1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_per_primitive_path(seed):
    from stable_envs.render import gfxdraw_fused as F

    dev = "cuda"
    rng = np.random.default_rng(seed)
    B, P, n_gfx = 4, 10, 6
    verts = _random_polys(rng, B, P, dev)
    colours = torch.as_tensor(rng.integers(0, 256, (B, P, 3)), dtype=torch.float32, device=dev)
    nvs = [4] * P
    vert, meta, edge = _tables(verts, colours, nvs, n_gfx)
    got = F.render_surface(vert, meta, edge, torch.ones(B, dtype=torch.int32, device=dev), H, W, n_gfx, P)
    want = _reference(verts, colours, nvs, n_gfx)
    assert_frames_identical(
        got,
        want,
        what=f"the fused rasterizer against the per-primitive one, seed {seed}",
        why="gfxdraw.py and carracing._draw_poly_mask are the executable specification — each rule in "
        "them was recovered by pixel-diffing pygame — and the fused kernel is a rewrite of the "
        "same rules into a closed form per pixel, which is admissible only while the two agree",
    )


def test_exact_integer_crossing_is_not_rounded_away():
    """A crossing that lands on an exact integer column must floor to it, not one short.

    The regression this pins: with an approximate reciprocal ``-42/14`` comes back a hair below -3,
    ``floor`` gives -4, the span starts a pixel late and the polygon loses a boundary pixel. It fired
    on ONE pixel in 19.2 M -- only where the geometry makes the quotient exact -- which is why it is
    pinned by the quad that produced it rather than left to a random search.
    """
    _draw_poly_mask = _poly_mask()
    from stable_envs.render import gfxdraw_fused as F

    dev = "cuda"
    quad = torch.tensor(
        [[[[522.0, 217.0], [516.0, 231.0], [509.0, 228.0], [515.0, 214.0]]]], dtype=torch.float64, device=dev
    )  # the CarRacing wheel
    colour = torch.full((1, 1, 3), 200.0, dtype=torch.float32, device=dev)
    vert, meta, edge = _tables(quad, colour, [4], 0)
    # PG=0 so nothing takes the gfxdraw branch and the edge table is never read; PPOST=1 so the one
    # primitive is drawn in PRE-flip coordinates, which is the frame the mask below is in.
    got = F.render_surface(vert, meta, vert, torch.ones(1, dtype=torch.int32, device=dev), H, W, 0, 1)
    fused = torch.flip(got[0, ..., 0] == 200, dims=(0,))  # undo the reference's flip
    want = _draw_poly_mask(quad[0], torch.arange(H, device=dev), torch.arange(W, device=dev))[0]
    assert torch.equal(fused, want), (
        f"the fused fill and pygame.draw.polygon disagree on {int((fused != want).sum())} pixels of this quad"
    )
