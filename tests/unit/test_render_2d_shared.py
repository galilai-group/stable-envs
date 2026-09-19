"""What the Step-9 render-2d consolidation bought, pinned so it cannot quietly come apart.

Three laws in this package were each written out twice, in modules that are certified against one
another pixel for pixel. Each is now one definition, and each test below is the specific way the
duplication could return: not "the two agree today" (two copies of an expression agree today, which
is why nobody noticed), but "there is ONE object", checked by identity wherever identity is the real
property, and by value over the inputs that separate the two spellings where it is not.
"""

from __future__ import annotations

import pytest
import torch

from stable_envs.render import gfxdraw as G
from stable_envs.render import gfxdraw_classic as GC
from stable_envs.render import gfxdraw_fused as GF
from stable_envs.render import navdot as NV
from stable_envs.render import raster2d as R2

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- the tile law
#: Frame geometries that separate the interesting cases: sizes that are a power of two, sizes that
#: are not, sizes below one tile, and the registered sizes of the 2-D cells (64, 84, 96, 144, 224,
#: 300, 400x600, 500, 600, 1000x800).
SIZES = [1, 3, 14, 42, 63, 64, 84, 96, 100, 128, 144, 224, 300, 400, 500, 600, 800, 1000]


def test_the_tile_law_is_the_one_navdot_uses():
    """``navdot._tile`` is ``raster2d.tile_shape`` at its own lane budget, not a second copy of it.

    The two kernels differ only in the width cap (unbounded against 128), so that is the argument;
    everything else -- the power-of-two ceiling, the waste bound, the height from the leftover lanes
    -- is the same arithmetic, and a second copy of it is what this checks for.
    """
    for H in SIZES:
        for W in SIZES:
            bh, bw, ntx, nt = NV._tile(H, W)
            assert (bh, bw) == R2.tile_shape(H, W, lanes=NV._BLOCK, max_width=NV._BLOCK), (H, W)
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


def test_the_tile_law_has_exactly_one_definition():
    """``raster2d`` no longer holds the private width fit that ``navdot`` had transcribed."""
    assert not hasattr(R2, "_fit"), "the old private tile-width fit is back"
    assert "tile_shape" in open(NV.__file__).read(), "navdot stopped using the shared law"


# --------------------------------------------------------------------------- the Wu edge law
def _random_quads(seed, n=9, H=37, W=53, spread=25):
    g = torch.Generator().manual_seed(seed)
    vx = torch.randint(-spread, W + spread, (n, 4), generator=g, dtype=torch.int64)
    vy = torch.randint(-spread, H + spread, (n, 4), generator=g, dtype=torch.int64)
    return vx, vy, H, W


@pytest.mark.parametrize("seed", range(8))
def test_the_fused_edge_table_is_the_per_primitive_walks_own_setup(seed):
    """``edge_table``'s twelve shared columns ARE ``edge_setup``'s tuple, edge for edge.

    These two are the reason the consolidation matters: ``gfxdraw`` blends a fringe from the tuple
    and ``gfxdraw_fused`` blends one from the table, the GPU test suite diffs the two pictures, and
    until now the integers behind them were two transcriptions of one rule. Compared over quads that
    cross every branch -- clipped, unclipped, vertical, horizontal, x-major, y-major -- rather than
    over tidy ones.
    """
    vx, vy, H, W = _random_quads(seed)
    tab = GF.edge_table(vx, vy, H, W)  # (M,4,EG_W) int32
    nx, ny = vx.roll(-1, 1), vy.roll(-1, 1)
    a, b, c, d = (t.reshape(-1) for t in (vx, vy, nx, ny))
    (draw, sx, sy, xx0, yy0, dx, dy, xdir, vert, horiz, ymaj, major, E, lo) = G.edge_setup(a, b, c, d, W, H)
    flat = tab.permute(0, 1, 2).reshape(-1, GF.EG_W).to(torch.int64)
    # ``edge_table`` is laid out edge-major within each quad, which is how the flat tuple is ordered
    # too, so the two line up row for row.
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
    """The test above is only worth its name if its inputs hit all four edge kinds and both sides of
    the clip. Asserted, because a generator that quietly stopped producing vertical edges would leave
    the comparison passing on a third of the rule."""
    vx, vy, H, W = _random_quads(seed)
    # add one deliberately axis-aligned and one fully outside quad, which random integers rarely give
    vx = torch.cat([vx, torch.tensor([[5, 5, 20, 20], [-40, -40, -30, -30]])])
    vy = torch.cat([vy, torch.tensor([[5, 20, 20, 5], [-40, -30, -30, -40]])])
    tab = GF.edge_table(vx, vy, H, W).reshape(-1, GF.EG_W)
    kinds = set(int(k) for k in tab[:, GF.EG_KIND].unique())
    assert {GF.K_VERT, GF.K_HORIZ, GF.K_GEN, GF.K_GEN + 4} <= kinds, kinds
    drawn = tab[:, GF.EG_DRAW]
    assert int((drawn != 0).sum()) and int((drawn == 0).sum()), "clip never fired both ways"


def test_the_edge_law_has_exactly_one_definition():
    """``edge_setup`` is a re-ordering of ``edge_geometry``, and ``edge_table`` calls the same one."""
    src = open(GF.__file__).read()
    assert "G.edge_geometry(" in src, "gfxdraw_fused stopped using gfxdraw's edge law"
    assert "clip_line" not in src.split("def bin_primitives")[0].split("edge_geometry")[-1], (
        "gfxdraw_fused clips its own edges again"
    )


# --------------------------------------------------------------------------- the smaller copies
def test_the_unsigned_32_mask_is_one_constant():
    """``gfxdraw_fused._blend`` is ``gfxdraw.blend`` in Triton; they must mask at the same width."""
    assert GF.M32 is G.M32 == (1 << 32) - 1


def test_the_classic_path_blends_through_gfxdraws_own_rule():
    """``gfxdraw_classic`` had a one-line wrapper around ``gfxdraw.blend``; there is no second name
    for the blend any more, and the torch specification calls the rule itself."""
    assert not hasattr(GC, "_blend_t")
    assert "G.blend(" in open(GC.__file__).read()


def test_row_groups_is_one_function_under_two_names():
    """The public spelling and the Step-8 one that ``envs/navigation.py`` still imports are the SAME
    object -- a second function would be a second partition, and the two paths would group the batch
    differently."""
    assert R2.row_groups is R2._row_groups


def test_the_scene_builder_broadcast_contract_is_in_the_signatures():
    """``Batched`` is what every builder argument accepts; it exists so the promise is on the methods
    and not only in the class docstring."""
    import inspect

    from stable_envs.render import scene as SC

    assert hasattr(SC, "Batched")
    for name in ("add_disc", "add_poly", "add_segment", "add_rect"):
        sig = inspect.signature(getattr(SC.SceneBuilder, name))
        assert sig.return_annotation is not inspect.Signature.empty, name
        assert all(p.annotation is not inspect.Parameter.empty for n, p in sig.parameters.items() if n != "self"), name
