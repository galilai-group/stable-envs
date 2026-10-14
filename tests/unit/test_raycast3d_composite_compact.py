"""Byte-exact comparisons of compact compositing against the independent full-window oracle."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
R3 = pytest.importorskip("stable_envs.render.raycast3d")
from tests.helpers.raycast_reference import composite_window


def _camera():
    return R3.Camera(
        pos=(0.0, -2.6, 1.1),
        forward=(0.0, 1.0, -0.35),
        up=(0.0, 0.35, 1.0),
        near=0.05,
        top=0.026,
        bottom=-0.026,
        center=0.0,
    )


def _lights():
    return R3.Lights(
        dirs=np.array([[0.3, -0.5, 0.81], [-0.6, 0.2, 0.77]]),
        ambient=np.array([0.25, 0.1]),
        diffuse=np.array([0.7, 0.4]),
        specular=np.array([0.35, 0.2]),
        exponent=12.0,
    )


def _prims(kinds, g):
    K = len(kinds)
    size = torch.rand(K, 3, generator=g).numpy() * 0.12 + 0.03
    return R3.Primitives(
        kind=np.asarray(kinds, int),
        size=size,
        mat_rgb=torch.rand(K, 3, generator=g).numpy(),
        tex_rgb=torch.rand(K, 3, generator=g).numpy() * 0.5 + 0.5,
        spec=torch.rand(K, generator=g).numpy(),
    )


#: ``t`` of a mover at this scene's default placement, from the camera in ``_camera()``. The
#: occlusion band below is built around it: ``plate_t`` has to CUT THROUGH the movers to be a real
#: threshold, and a band chosen without reference to where they are would sit entirely in front of
#: them or entirely behind, which is the same as not varying at all.
_MOVER_T = 2.72


def _scene(
    seed,
    *,
    B=5,
    K=2,
    H=41,
    W=37,
    ss=2,
    spread=0.25,
    depth=0.0,
    kinds=(R3.SPHERE, R3.CAPSULE),
    plate_t="floor",
    offset=(0.0, 0.0),
    scales=None,
):
    """A plate, its ray parameter, and a batch of mover poses in front of the camera.

    ``plate_t`` selects the occlusion threshold, and the choice is load-bearing rather than
    decorative:

    * ``"floor"`` -- the analytic depth of a ground plane through the origin. Smooth, and far enough
      behind the movers that every sample of the window is covered or not covered for reasons that
      have nothing to do with which pixel's threshold it read.
    * ``"inf"`` -- nothing static in the way, so ``INF < INF`` has to stay False.
    * ``"band"`` -- **a FINITE per-pixel field that cuts through the movers**, so a sample's coverage
      depends on WHICH pixel's ``plate_t`` it was handed. This is the only mode that exercises the
      ``bt`` gather: the compact path reads ``plate_t`` at the window's output pixels and blows the
      value up over the ``ss x ss`` subgrid, where the path it replaces blew the whole frame up first
      and read the window out of that. Those are the same numbers only if the blow-up maps
      ``(ty, a, tx, b) -> (y0+ty, x0+tx)``; transposing it is a real defect that the ``"floor"`` and
      ``"inf"`` scenes cannot see, because under them every sample of a window shares an answer.
      The field depends on ``y`` and ``x`` with different periods so a transposition cannot alias
      back onto itself, and ``R3.INF`` never appears in it -- the sentinel would collapse the
      comparison to ``t < INF`` and hide exactly what this mode is for.

    ``offset`` shifts every mover in world x and z, which is how a window gets pushed against a frame
    edge and the origin clamp in ``_fit_tile`` is made to engage. ``scales`` is a per-row
    ``size_scale``.
    """
    g = torch.Generator().manual_seed(seed)
    cam = _camera()
    prims = _prims(kinds[:K], g)
    plate = (torch.rand(H, W, 3, generator=g) * 255).to(torch.uint8)
    if plate_t == "inf":
        plate_t_v = torch.full((H * W,), R3.INF)
    elif plate_t == "band":
        y = torch.arange(H).view(H, 1).to(torch.float32)
        x = torch.arange(W).view(1, W).to(torch.float32)
        phase = (3.0 * y + 7.0 * x) % 11.0 / 11.0 + 0.017 * ((y * 5 + x * 2) % 3)
        plate_t_v = (_MOVER_T - 0.22 + 0.44 * phase).reshape(-1).contiguous()
        assert torch.isfinite(plate_t_v).all() and float(plate_t_v.max()) < R3.INF
    else:
        floor = R3.Primitives(
            kind=np.array([R3.PLANE]),
            size=np.zeros((1, 3)),
            mat_rgb=np.zeros((1, 3)),
            tex_rgb=np.ones((1, 3)),
            spec=np.zeros(1),
        )
        plate_t_v = R3.static_depth(cam, floor, np.zeros((1, 3)), np.eye(3).reshape(1, 3, 3), H, W, "cpu")
    xpos = torch.zeros(B, K, 3)
    xpos[..., 0] = (torch.rand(B, K, generator=g) - 0.5) * spread + offset[0]
    xpos[..., 1] = (torch.rand(B, K, generator=g) - 0.5) * spread + depth
    xpos[..., 2] = torch.rand(B, K, generator=g) * spread + 0.12 + offset[1]
    xmat = torch.eye(3).view(1, 1, 3, 3).expand(B, K, 3, 3).contiguous()
    sc = dict(
        plate=plate,
        plate_t=plate_t_v,
        cam=cam,
        prims=prims,
        xpos=xpos,
        xmat=xmat,
        lights=_lights(),
        H=H,
        W=W,
        supersample=ss,
    )
    if scales is not None:
        sc["_scales"] = torch.as_tensor(scales, dtype=torch.float32)
    return sc


def _split(sc):
    """The composite kwargs, and the per-row ``size_scale`` the scene carried beside them."""
    sc = dict(sc)
    return sc, sc.pop("_scales", None)


def _both(sc, *, cover_compaction=True, **kw):
    sc, scales = _split(sc)
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
            "The compact tail is meant to be the identity of the window path, so any difference "
            "is a defect in one of the three arguments in this file's docstring, not a tolerance."
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
    # the three cases below are the only ones in this file whose coverage depends on WHICH
    # pixel's occlusion threshold a sample was handed -- see ``_scene``'s "band" mode
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
    # the clamp and a per-row size_scale together: the scale feeds ``bound_scale``, so it moves the
    # span the tile is fitted to AND the primitive the cast tests, on a window already pinned
    # against two frame edges
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
    sc = _scene(**skw)
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
    sc = _scene(
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


def _cover_fraction(sc, tile):
    """What fraction of the window each row covers -- the quantity the fuzz has to span."""
    sc, _ = _split(sc)
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
    sc, _ = _split(sc)
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
    cx, cy, t = _clamp_counts(_scene(**skw), ckw["tile"])
    assert t is not None, f"{cid}: no window was fitted, so no clamp can be exercised"
    assert cx + cy > 0, (
        f"{cid}: the fitted window ({t} px) clamped no row in either axis, so this case is not "
        "exercising the origin clamp it is named for"
    )


def test_the_fuzz_actually_reaches_empty_and_full_cover_rows():
    """A differential test that never varies the cover has verified one point, not an identity.

    A dump comparison once printed ``ALL_IDENTICAL=True`` over two empty
    directories because ``all([])`` is True. The same shape of false green is available here: if
    every fuzz scene happened to put the movers off screen, both paths would return the plate and
    every assertion would pass having compared nothing.
    """
    empty = _scene(seed=41, K=1, kinds=(R3.SPHERE,), depth=-6.0, spread=0.02)
    assert _cover_fraction(empty, 16) == 0, "the empty-cover case is covering something"
    a, b = _both(empty, tile=16)
    _assert_identical(a, b, "empty cover")
    assert torch.equal(a, empty["plate"].view(1, *empty["plate"].shape).expand(a.shape[0], *empty["plate"].shape)), (
        "with nothing covered the composite must be the plate on every row"
    )

    full = _scene(seed=42, K=1, kinds=(R3.BOX,), spread=0.02, plate_t="inf")
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
    """``maze2d`` reads ``with_t != plate`` to find the target, so every pixel of ``out`` is load-bearing."""
    sc, _ = _split(_scene(seed=51, B=6, K=1, kinds=(R3.SPHERE,)))
    buf = torch.full((6, sc["H"], sc["W"], 3), 7, dtype=torch.uint8)
    given = R3.composite(**sc, tile=16, out=buf).clone()
    fresh = R3.composite(**sc, tile=16).clone()
    _assert_identical(given, fresh, "caller buffer vs fresh")
    assert int((given == 7).all(-1).sum()) < given.shape[1] * given.shape[2] * given.shape[0], (
        "the caller's buffer was left at its fill value, so the plate broadcast did not happen"
    )


def test_per_environment_colour_and_scale_are_still_per_environment():
    sc = _scene(seed=61, B=4, K=1, kinds=(R3.SPHERE,))
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
    sc, _ = _split(_scene(seed=3000 + seed, B=7, K=2, spread=0.1 + 0.15 * seed, depth=0.2 * (seed % 3)))
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
    sc = _scene(seed=71, B=3, K=2, spread=2.0)
    assert R3._fit_tile(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], 4, "cpu") is None
    a, b = _both(sc, tile=4)
    _assert_identical(a, b, "fallback to the full frame")


def test_compact_tail_matches_full_window_without_cover_compaction():
    sc = _scene(seed=81, B=5, K=1, kinds=(R3.SPHERE,))
    a, b = _both(sc, tile=16, cover_compaction=False)
    _assert_identical(a, b, "compact tail vs uncompact window")


@pytest.mark.parametrize("seed", range(8))
def test_a_row_renders_the_same_alone_as_it_does_in_a_batch(seed):
    """The fitted tile is derived from the batch's WORST span, so the batch can change the window.

    It must not change the picture. That is not automatic and it is not what the identity argument
    above says: the argument is that any admissible window gives the same frame, and this is the test
    that the window the batch happens to pick is still admissible for every row in it. RAYCAST-4 made
    the same distinction for its per-primitive bound -- batch invariance is a claim there, not an
    identity -- and it earned its own test for the same reason.
    """
    sc, _ = _split(_scene(seed=4000 + seed, B=8, K=2, spread=0.2 + 0.1 * seed))
    together = R3.composite(**sc, tile=16).clone()
    for i in range(sc["xpos"].shape[0]):
        one = dict(sc, xpos=sc["xpos"][i : i + 1].contiguous(), xmat=sc["xmat"][i : i + 1].contiguous())
        alone = R3.composite(**one, tile=16).clone()
        _assert_identical(alone, together[i : i + 1], f"row {i} alone vs in a batch of 8")
