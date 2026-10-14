"""Byte-exact comparisons of compact compositing against the independent full-window oracle."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
R3 = pytest.importorskip("stable_envs.render.raycast3d")
from tests.helpers.raycast_reference import composite_window, scene, split_scene


def _both(sc, *, cover_compaction=True, **kw):
    sc, scales = split_scene(sc)
    if scales is not None:
        kw.setdefault("size_scale", scales)
    a = R3.composite(**sc, **kw).clone()
    b = composite_window(**sc, **kw, cover_compaction=cover_compaction).clone()
    return a, b


def _assert_identical(a, b, what):
    assert a.dtype == b.dtype == torch.uint8 and a.shape == b.shape, what
    if not torch.equal(a, b):
        diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
        n = int((diff > 0).sum())
        raise AssertionError(
            f"{what}: {n} of {diff.numel()} channel values differ, max |delta| {int(diff.max())}. "
            "The compact tail must equal the full-window path exactly."
        )


# --------------------------------------------------------------------------- the differential fuzz

CASES = [
    # (id, scene kwargs, composite kwargs)
    ("small-sphere-tiled", dict(seed=1, K=1, kinds=(R3.SPHERE,)), dict(tile=16)),
    ("two-movers-tiled", dict(seed=2, K=2), dict(tile=16)),
    ("odd-frame-ss2", dict(seed=3, H=43, W=29, ss=2), dict(tile=11)),
    ("odd-frame-ss1", dict(seed=4, H=43, W=29, ss=1), dict(tile=11)),
    ("ss1-square", dict(seed=5, ss=1), dict(tile=16)),
    ("no-tile-full-frame", dict(seed=6), dict(tile=None)),
    ("tile-too-small-falls-back", dict(seed=7, spread=1.4), dict(tile=4)),
    ("batch-of-one", dict(seed=8, B=1), dict(tile=16)),
    ("large-batch", dict(seed=9, B=32), dict(tile=16)),
    ("plate-t-infinite", dict(seed=10, plate_t="inf"), dict(tile=16)),
    # coverage in these three depends on which pixel's threshold a sample reads; see ``scene``
    ("plate-t-band-ss2", dict(seed=14, K=1, kinds=(R3.SPHERE,), plate_t="band"), dict(tile=16)),
    ("plate-t-band-ss1", dict(seed=15, K=1, kinds=(R3.SPHERE,), ss=1, plate_t="band"), dict(tile=16)),
    ("plate-t-band-two-movers", dict(seed=16, K=2, plate_t="band"), dict(tile=16)),
    ("box-and-cylinder", dict(seed=11, K=2, kinds=(R3.BOX, R3.CYLINDER)), dict(tile=16)),
    ("far-behind-camera", dict(seed=12, depth=-4.0), dict(tile=16)),
    # ``offset`` moves the movers until ``_fit_tile``'s ``x0.clamp(0, W - t)`` engages;
    # ``test_the_edge_cases_actually_clamp`` asserts that it does.
    (
        "mover-clamped-left",
        dict(seed=13, B=6, K=1, kinds=(R3.SPHERE,), spread=0.35, offset=(-1.4, 0.0)),
        dict(tile=16),
    ),
    (
        "mover-clamped-right",
        dict(seed=17, B=6, K=1, kinds=(R3.SPHERE,), spread=0.35, offset=(1.4, 0.0)),
        dict(tile=16),
    ),
    (
        "mover-clamped-corner",
        dict(seed=18, B=6, K=1, kinds=(R3.SPHERE,), spread=0.35, offset=(-1.15, 0.95)),
        dict(tile=16),
    ),
    (
        "mover-clamped-band",
        dict(seed=19, B=6, K=1, kinds=(R3.SPHERE,), spread=0.35, offset=(-1.15, 0.95), plate_t="band"),
        dict(tile=16),
    ),
    # the clamp with a per-row size_scale, which moves both the fitted span and the cast primitive
    (
        "clamped-ss2-per-row-scale",
        dict(
            seed=20,
            B=6,
            K=1,
            kinds=(R3.SPHERE,),
            ss=2,
            spread=0.35,
            offset=(-1.15, 0.95),
            plate_t="band",
            scales=(0.5, 0.8, 1.0, 1.4, 1.9, 2.4),
        ),
        dict(tile=16),
    ),
]


@pytest.mark.parametrize("cid,skw,ckw", CASES, ids=[c[0] for c in CASES])
def test_the_compact_tail_is_bit_identical_to_the_window_it_replaces(cid, skw, ckw):
    sc = scene(**skw)
    a, b = _both(sc, **ckw)
    _assert_identical(a, b, cid)


@pytest.mark.parametrize("seed", range(24))
def test_random_scenes_agree_on_every_byte(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    ri = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=g))  # noqa: E731
    H, W = ri(17, 48), ri(17, 48)
    ss = ri(1, 2)
    K = ri(1, 3)
    kinds = tuple([R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX][ri(0, 3)] for _ in range(K))
    sc = scene(
        seed=2000 + seed,
        B=ri(1, 9),
        K=K,
        H=H,
        W=W,
        ss=ss,
        kinds=kinds,
        spread=float(torch.rand(1, generator=g)) * 1.2 + 0.05,
        depth=float(torch.rand(1, generator=g)) - 0.5,
        plate_t=("band", "floor", "inf")[ri(0, 2)],
    )
    tile = ri(3, min(H, W) - 1)
    a, b = _both(sc, tile=tile)
    _assert_identical(a, b, f"random seed {seed} H={H} W={W} ss={ss} K={K} tile={tile}")


def _shaded_count(sc, tile):
    """Number of samples ``composite`` shades for this scene."""
    sc, _ = split_scene(sc)
    seen = {}
    orig = R3.shade

    def probe(n, *a, **kw):
        seen["shaded"] = int(n.reshape(-1, 3).shape[0])
        return orig(n, *a, **kw)

    R3.shade = probe
    try:
        R3.composite(**sc, tile=tile)
    finally:
        R3.shade = orig
    return seen.get("shaded", 0)


def _clamp_counts(sc, tile):
    """(rows clamped in x, rows clamped in y, the fitted tile) for this scene."""
    sc, _ = split_scene(sc)
    span = R3.screen_span(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], "cpu")
    fit = R3._fit_tile(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], tile, "cpu")
    if span is None or fit is None:
        return 0, 0, None
    x0, y0, _, _ = span
    t = fit[1]
    return (int((x0 < 0).sum() + (x0 > sc["W"] - t).sum()), int((y0 < 0).sum() + (y0 > sc["H"] - t).sum()), t)


CLAMPED_IDS = (
    "mover-clamped-left",
    "mover-clamped-right",
    "mover-clamped-corner",
    "mover-clamped-band",
    "clamped-ss2-per-row-scale",
)


@pytest.mark.parametrize("cid", CLAMPED_IDS)
def test_the_edge_cases_actually_clamp(cid):
    """A case named for the origin clamp must reach it; a centred mover makes ``x0.clamp(0, W - t)`` the identity."""
    skw = dict(next(c[1] for c in CASES if c[0] == cid))
    ckw = next(c[2] for c in CASES if c[0] == cid)
    cx, cy, t = _clamp_counts(scene(**skw), ckw["tile"])
    assert t is not None, f"{cid}: no window was fitted, so no clamp can be exercised"
    assert cx + cy > 0, (
        f"{cid}: the fitted window ({t} px) clamped no row in either axis, so this case is not "
        "exercising the origin clamp it is named for"
    )


def test_the_fuzz_actually_reaches_empty_and_full_cover_rows():
    """The fuzz reaches both an empty cover (the plate alone) and a full cover."""
    empty = scene(seed=41, K=1, kinds=(R3.SPHERE,), depth=-6.0, spread=0.02)
    assert _shaded_count(empty, 16) == 0, "the empty-cover case is covering something"
    a, b = _both(empty, tile=16)
    _assert_identical(a, b, "empty cover")
    assert torch.equal(a, empty["plate"].view(1, *empty["plate"].shape).expand(a.shape[0], *empty["plate"].shape)), (
        "with nothing covered the composite must be the plate on every row"
    )

    full = scene(seed=42, K=1, kinds=(R3.BOX,), spread=0.02, plate_t="inf")
    full["xpos"][..., 2] = 0.35
    full["prims"] = R3.Primitives(
        kind=np.array([R3.BOX]),
        size=np.full((1, 3), 0.9),
        mat_rgb=np.array([[0.8, 0.2, 0.2]]),
        tex_rgb=np.ones((1, 3)),
        spec=np.array([0.4]),
    )
    a, b = _both(full, tile=16)
    _assert_identical(a, b, "full cover")
    assert not torch.equal(a, full["plate"].view(1, *full["plate"].shape).expand(a.shape[0], *full["plate"].shape)), (
        "the full-cover case did not draw anything, so it is not testing the merge"
    )


def test_a_caller_supplied_buffer_and_a_fresh_one_agree():
    """Compositing into a caller's ``out`` writes every pixel and matches a fresh buffer."""
    sc, _ = split_scene(scene(seed=51, B=6, K=1, kinds=(R3.SPHERE,)))
    buf = torch.full((6, sc["H"], sc["W"], 3), 7, dtype=torch.uint8)
    given = R3.composite(**sc, tile=16, out=buf).clone()
    fresh = R3.composite(**sc, tile=16).clone()
    _assert_identical(given, fresh, "caller buffer vs fresh")
    assert int((given == 7).all(-1).sum()) < given.shape[1] * given.shape[2] * given.shape[0], (
        "the caller's buffer was left at its fill value, so the plate broadcast did not happen"
    )


def test_per_environment_colour_and_scale_are_still_per_environment():
    sc = scene(seed=61, B=4, K=1, kinds=(R3.SPHERE,))
    mat = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0]]])
    scale = torch.tensor([0.6, 1.0, 1.4, 1.8])
    a, b = _both(sc, tile=16, mat_rgb=mat, size_scale=scale)
    _assert_identical(a, b, "per-env colour and scale")
    frames = [a[i] for i in range(4)]
    assert not all(torch.equal(frames[0], f) for f in frames[1:]), (
        "every row came out identical, so the per-environment appearance is not reaching the shade"
    )


# --------------------------------------------------------------------------- the window choice


@pytest.mark.parametrize("seed", range(12))
def test_the_chosen_window_always_contains_the_span_it_was_derived_from(seed):
    """The whole identity rests on ``_fit_tile`` never returning a window smaller than the bound."""
    sc, _ = split_scene(scene(seed=3000 + seed, B=7, K=2, spread=0.1 + 0.15 * seed, depth=0.2 * (seed % 3)))
    span = R3.screen_span(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], "cpu")
    if span is None:
        pytest.skip("no finite bound for this scene")
    x0, y0, w, h = span
    fit = R3._fit_tile(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], 32, "cpu")
    if fit is None:
        assert max(w, h) > 32, "the fit was refused for a span that fits"
        return
    (fx, fy), t = fit
    assert t >= w and t >= h, f"window {t} is smaller than the span {w}x{h}"
    assert t <= 32, "the fit exceeded the caller's ceiling"
    lo_x, lo_y = fx.min().item(), fy.min().item()
    assert lo_x >= 0 and lo_y >= 0 and fx.max() + t <= sc["W"] and fy.max() + t <= sc["H"], "the window left the frame"
    # every row's own span box must sit inside its own window
    assert bool(((fx <= x0.clamp(min=0)) | (x0 < 0)).all()), "a row's window starts after its span"
    assert bool(((fy <= y0.clamp(min=0)) | (y0 < 0)).all()), "a row's window starts after its span"


def test_a_span_wider_than_the_ceiling_still_falls_back_to_the_full_frame():
    sc = scene(seed=71, B=3, K=2, spread=2.0)
    assert R3._fit_tile(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], 4, "cpu") is None
    a, b = _both(sc, tile=4)
    _assert_identical(a, b, "fallback to the full frame")


def test_compact_tail_matches_full_window_without_cover_compaction():
    sc = scene(seed=81, B=5, K=1, kinds=(R3.SPHERE,))
    a, b = _both(sc, tile=16, cover_compaction=False)
    _assert_identical(a, b, "compact tail vs uncompact window")


@pytest.mark.parametrize("seed", range(8))
def test_a_row_renders_the_same_alone_as_it_does_in_a_batch(seed):
    """The fitted tile comes from the batch's worst span, so the batch can change the window.

    It must not change the picture: the window the batch picks has to be admissible for every row.
    """
    sc, _ = split_scene(scene(seed=4000 + seed, B=8, K=2, spread=0.2 + 0.1 * seed))
    together = R3.composite(**sc, tile=16).clone()
    for i in range(sc["xpos"].shape[0]):
        one = dict(sc, xpos=sc["xpos"][i : i + 1].contiguous(), xmat=sc["xmat"][i : i + 1].contiguous())
        alone = R3.composite(**one, tile=16).clone()
        _assert_identical(alone, together[i : i + 1], f"row {i} alone vs in a batch of 8")
