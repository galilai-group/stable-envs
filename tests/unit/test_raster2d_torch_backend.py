"""``render_torch`` matches a frozen unoptimized torch rasterizer, on CPU, byte for byte.

``render_torch`` skips primitive families a slot lacks, unrequested outlines and rings, padded
degenerate edges and pixels outside a primitive's world-space bound, and groups rows with distant
bounds into separate windows. Each skip claims that no pixel could have changed.

The oracle is the unoptimized rasterizer in ``scripts/regression/raster2d/fuzz_render_torch.py``.
``tests/gpu/test_conformance.py::test_backend_agreement`` checks the fused kernel on CUDA.

The generated scenes cover cases the registered envs do not produce: supersample above 1, collinear
polygons, negative radii.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from stable_envs.render import raster2d as R2
from stable_envs.render.scene import Scene2D

ROOT = Path(__file__).resolve().parents[2]
FUZZ = ROOT / "scripts" / "regression" / "raster2d" / "fuzz_render_torch.py"


def _oracle():
    if not FUZZ.is_file():  # a checkout without the raster2d fuzz oracle still runs the suite
        pytest.skip(f"{FUZZ.relative_to(ROOT)} is not in this checkout")
    spec = importlib.util.spec_from_file_location("se_fuzz_oracle", FUZZ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fuzz():
    return _oracle()


def _same(want, got, what):
    if torch.equal(want, got):
        return
    d = (want.to(torch.int16) - got.to(torch.int16)).abs()
    raise AssertionError(
        f"{what}: render_torch differs from the frozen baseline. "
        f"{int((d.amax(-1) > 0).sum())} pixels differ, max |diff| {int(d.max())} of 255. "
        f"One of render_torch's skips discards work that reaches a pixel."
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_the_frozen_baseline_on_random_scenes(fuzz, seed):
    """The generated corpus: disabled slots, collinear and all-degenerate polygons, padded
    triangles, negative radii, rings larger than their disc, primitives off the frame and larger
    than the world, flip_y, non-square worlds and outputs, supersample 2."""
    gen = torch.Generator().manual_seed(seed)
    for c in range(40):
        sc = fuzz.random_scene(gen, "cpu")
        ss = 1 + (c % 3 == 2)
        size = 17 if c % 4 else (17, 22)
        _same(
            fuzz.render_torch_baseline(sc, size, ss),
            R2.render_torch(sc, size, ss),
            f"seed {seed} case {c} (B={sc.batch} N={sc.n_prims} V={sc.vertices} ss={ss})",
        )


def _scene(kind, geom, *, line_w=0.0, world=(0.0, 0.0, 1.0, 1.0), flip_y=False, fill=200.0, line=40.0, bg=10.0):
    B, N = kind.shape
    f = lambda v: torch.full((B, N, 3), float(v))  # noqa: E731
    return Scene2D(
        kind=kind.to(torch.int32),
        geom=geom.to(torch.float32),
        fill=f(fill),
        line=f(line),
        line_w=torch.full((B, N), float(line_w)),
        background=torch.full((B, 3), float(bg)),
        world=world,
        flip_y=flip_y,
    ).validate()


def _quad(B, pts):
    return torch.tensor(pts, dtype=torch.float32).reshape(1, 1, 8).expand(B, 1, 8).contiguous()


def _cases():
    """Cases a random generator reaches rarely or never.

    The bound does not cull a polygon with exactly zero signed area: its halfplane intersection is a
    line or a halfplane, not bounded by the vertex hull. A uniform draw never produces exact zero.
    """
    B = 3
    yield (
        "exactly collinear, CCW order",
        _scene(torch.ones(B, 1, dtype=torch.int32) * 2, _quad(B, [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7])),
    )
    yield (
        "exactly collinear, bow-tie order",
        _scene(torch.ones(B, 1, dtype=torch.int32) * 2, _quad(B, [0.1, 0.1, 0.5, 0.5, 0.3, 0.3, 0.7, 0.7])),
    )
    yield (
        "exactly collinear, vertical",
        _scene(torch.ones(B, 1, dtype=torch.int32) * 2, _quad(B, [0.5, 0.0, 0.5, 0.3, 0.5, 0.6, 0.5, 1.0])),
    )
    yield (
        "collinear beside a real quad",
        _scene(
            torch.tensor([[2, 2]] * B, dtype=torch.int32),
            torch.cat(
                [
                    _quad(B, [0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 0.9, 0.9]),
                    _quad(B, [0.2, 0.2, 0.6, 0.2, 0.6, 0.6, 0.2, 0.6]),
                ],
                dim=1,
            ),
        ),
    )
    for w, tag in (((0.0, 0.0, 1e6, 1e6), "1e6-scale world"), ((0.0, 0.0, 1e-4, 1e-4), "1e-4-scale world")):
        s = w[2]
        yield (
            tag,
            _scene(
                torch.tensor([[1, 2]] * B, dtype=torch.int32),
                torch.cat(
                    [
                        torch.tensor([[s * 0.3, s * 0.3, s * 0.2, s * 0.1, 0, 0, 0, 0]]).expand(B, 1, 8),
                        _quad(B, [s * 0.5, s * 0.5, s * 0.9, s * 0.5, s * 0.9, s * 0.9, s * 0.5, s * 0.9]),
                    ],
                    dim=1,
                ),
                world=w,
            ),
        )
    for bad, tag in ((float("nan"), "NaN geometry"), (float("inf"), "inf geometry")):
        g = _quad(B, [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]).clone()
        g[1, 0, 3] = bad
        yield tag, _scene(torch.ones(B, 1, dtype=torch.int32) * 2, g)
    yield (
        "radius exactly zero",
        _scene(
            torch.ones(B, 1, dtype=torch.int32),
            torch.tensor([[0.5, 0.5, 0.0, -1.0, 0, 0, 0, 0]]).expand(B, 1, 8).contiguous(),
        ),
    )
    yield (
        "negative line_w",
        _scene(
            torch.ones(B, 1, dtype=torch.int32) * 2, _quad(B, [0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7]), line_w=-0.2
        ),
    )
    yield (
        "line_w wider than the polygon",
        _scene(
            torch.ones(B, 1, dtype=torch.int32) * 2,
            _quad(B, [0.45, 0.45, 0.55, 0.45, 0.55, 0.55, 0.45, 0.55]),
            line_w=0.4,
        ),
    )
    yield (
        "every slot disabled",
        _scene(
            torch.zeros(B, 2, dtype=torch.int32),
            _quad(B, [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8]).expand(B, 2, 8).contiguous(),
        ),
    )
    yield (
        "one row enabled, the rest off",
        _scene(torch.tensor([[2], [0], [0]], dtype=torch.int32), _quad(B, [0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8])),
    )


@pytest.mark.parametrize("tag,scene", list(_cases()), ids=lambda v: v if isinstance(v, str) else "")
def test_matches_the_frozen_baseline_on_named_cases(fuzz, tag, scene):
    for ss, size in ((1, 24), (1, (24, 31)), (2, 13), (2, 11)):
        _same(
            fuzz.render_torch_baseline(scene, size, ss),
            R2.render_torch(scene, size, ss),
            f"{tag} at ss={ss} size={size}",
        )


def test_row_grouping_is_exercised_by_this_suite(fuzz):
    """At least one scene draws through a grouped window, so ``row_groups`` is exercised.

    Grouping engages only where the batch-union window is much larger than one row's box and the cost
    model accepts. One scattered disc per row at B=256 qualifies; at B=64 and 64 px the model declines.
    """
    B, S = 256, 64
    gen = torch.Generator().manual_seed(4)
    g = torch.zeros(B, 1, 8)
    g[:, 0, 0:2] = torch.rand(B, 2, generator=gen) * 0.9 + 0.05
    g[:, 0, 2], g[:, 0, 3] = 0.02, -1.0
    sc = _scene(torch.ones(B, 1, dtype=torch.int32), g)
    plan = R2._prim_plan(sc, S, S)
    assert len(plan[0][6]) > 1, (
        "no primitive in this suite is drawn through a grouped window, so row_groups is never exercised"
    )
    _same(fuzz.render_torch_baseline(sc, S, 1), R2.render_torch(sc, S, 1), "grouped rows")


def test_the_grouped_and_ungrouped_windowings_agree(fuzz):
    """Grouped and ungrouped windowings draw the same picture; the cost model only picks throughput."""
    B = 128
    gen = torch.Generator().manual_seed(5)
    g = torch.zeros(B, 1, 8)
    g[:, 0, 0:2] = torch.rand(B, 2, generator=gen) * 0.9 + 0.05
    g[:, 0, 2], g[:, 0, 3] = 0.03, 0.015
    sc = _scene(torch.ones(B, 1, dtype=torch.int32), g)
    saved = R2._GROUP_DISPATCH_PIXELS
    try:
        R2._GROUP_DISPATCH_PIXELS = 10**12
        ungrouped = R2.render_torch(sc, 64, 1).clone()
        assert len(R2._prim_plan(sc, 64, 64)[0][6]) == 1
        R2._GROUP_DISPATCH_PIXELS = 0
        grouped = R2.render_torch(sc, 64, 1)
        assert len(R2._prim_plan(sc, 64, 64)[0][6]) > 1
    finally:
        R2._GROUP_DISPATCH_PIXELS = saved
    _same(ungrouped, grouped, "grouped vs ungrouped windowing")
    _same(fuzz.render_torch_baseline(sc, 64, 1), grouped, "grouped vs the frozen baseline")
