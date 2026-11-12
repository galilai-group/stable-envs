"""Shared 2-D rasterizer laws: the tile shape and the Wu edge setup used by both fused kernels."""

from __future__ import annotations

import pytest
import torch

from stable_envs import tuning
from stable_envs.envs.navigation import navdot as NV
from stable_envs.render import gfxdraw as G
from stable_envs.render import raster2d as R2
from stable_envs.render.gfxdraw import fused as GF

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- the tile law
#: Frame geometries that separate the interesting cases: sizes that are a power of two, sizes that
#: are not, sizes below one tile, and the registered sizes of the 2-D cells (64, 84, 96, 144, 224,
#: 300, 400x600, 500, 600, 1000x800).
SIZES = [1, 3, 14, 42, 63, 64, 84, 96, 100, 128, 144, 224, 300, 400, 500, 600, 800, 1000]


def test_navdot_tile_is_raster2d_tile_shape_at_its_lane_budget():
    """``navdot._tile`` is ``raster2d.tile_shape`` at navdot's lane budget, plus the tile counts."""
    for H in SIZES:
        for W in SIZES:
            bh, bw, ntx, nt = NV._tile(H, W)
            block = tuning.get("navdot_block")
            assert (bh, bw) == R2.tile_shape(H, W, lanes=block, max_width=block), (H, W)
            assert ntx == -(-W // bw) and nt == ntx * (-(-H // bh)), (H, W)


def test_a_tile_never_wastes_more_lanes_than_the_bound_allows():
    """The loop's exit condition, as a property: either the tile is one lane wide, or it is inside
    the waste bound the constant names. A tile that hangs off the right edge still runs the whole
    primitive loop for the lanes it masks, which is what the bound is protecting."""
    for max_width in (128, 1024):
        for W in SIZES:
            bh, bw = R2.tile_shape(64, W, lanes=1024, max_width=max_width)
            assert bw & (bw - 1) == 0, f"tile width {bw} is not a power of two"
            assert bw <= max_width
            assert bw == 1 or (bw * -(-W // bw) - W) <= R2.TILE_WASTE_MAX * W, (W, bw)
            assert 1 <= bh and bh * bw <= 1024


# --------------------------------------------------------------------------- the Wu edge law
def _random_quads(seed, n=9, H=37, W=53, spread=25):
    g = torch.Generator().manual_seed(seed)
    vx = torch.randint(-spread, W + spread, (n, 4), generator=g, dtype=torch.int64)
    vy = torch.randint(-spread, H + spread, (n, 4), generator=g, dtype=torch.int64)
    return vx, vy, H, W


@pytest.mark.parametrize("seed", range(8))
def test_the_fused_edge_table_is_the_per_primitive_walks_own_setup(seed):
    """``edge_table``'s shared columns equal ``edge_setup``'s tuple, edge for edge.

    Quads cross every branch: clipped, unclipped, vertical, horizontal, x-major, y-major.
    """
    vx, vy, H, W = _random_quads(seed)
    tab = GF.edge_table(vx, vy, H, W)  # (M,4,EG_W) int32
    nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
    a, b, c, d = (t.reshape(-1) for t in (vx, vy, nx, ny))
    (draw, sx, sy, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo) = G.edge_setup(a, b, c, d, W, H)
    flat = tab.permute(0, 1, 2).reshape(-1, GF.EG_W).to(torch.int64)
    cols = {
        GF.EG_DRAW: draw.to(torch.int64),
        GF.EG_SX: sx,
        GF.EG_SY: sy,
        GF.EG_X0: xx0,
        GF.EG_Y0: yy0,
        GF.EG_DX: dx,
        GF.EG_DY: dy,
        GF.EG_XDIR: xdir,
        GF.EG_MAJOR: major,
        GF.EG_E: E,
        GF.EG_LO: lo,
    }
    for col, want in cols.items():
        assert torch.equal(flat[:, col], want), f"edge column {col} disagrees with edge_setup"
    kind = flat[:, GF.EG_KIND]
    assert torch.equal(kind == GF.K_VERT, vert)
    assert torch.equal(kind == GF.K_HORIZ, horiz)
    assert torch.equal((kind == GF.K_GEN + 4), ~vert & ~horiz & ymaj)


@pytest.mark.parametrize("seed", range(4))
def test_every_branch_of_the_edge_law_is_actually_reached(seed):
    """The edge-table comparison inputs hit all four edge kinds and both sides of the clip."""
    vx, vy, H, W = _random_quads(seed)
    # add one deliberately axis-aligned and one fully outside quad, which random integers rarely give
    vx = torch.cat([vx, torch.tensor([[5, 5, 20, 20], [-40, -40, -30, -30]])])
    vy = torch.cat([vy, torch.tensor([[5, 20, 20, 5], [-40, -30, -30, -40]])])
    tab = GF.edge_table(vx, vy, H, W).reshape(-1, GF.EG_W)
    kinds = set(int(k) for k in tab[:, GF.EG_KIND].unique())
    assert {GF.K_VERT, GF.K_HORIZ, GF.K_GEN, GF.K_GEN + 4} <= kinds, kinds
    drawn = tab[:, GF.EG_DRAW]
    assert int((drawn != 0).sum()) and int((drawn == 0).sum()), "clip never fired both ways"


@pytest.mark.parametrize("spread", [3, 40, 2000])
def test_the_numba_edge_table_equals_the_torch_edge_table(spread, monkeypatch):
    """On CPU, ``edge_table`` fills its rows with Numba; every column equals the Torch table."""
    from stable_envs.render import cpu_raster

    if not cpu_raster.available():
        pytest.skip("numba is not installed")
    vx, vy, H, W = _random_quads(spread, n=4000, H=400, W=600, spread=spread)
    fast = GF.edge_table(vx, vy, H, W)
    monkeypatch.setitem(cpu_raster._STATE, "ok", False)
    assert torch.equal(fast, GF.edge_table(vx, vy, H, W))
