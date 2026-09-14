"""``composite``'s compact tail must be the IDENTITY of the window it replaces, byte for byte.

``raycast3d.composite`` used to cast, box-average, gather, blend and scatter the whole ``tile x tile``
window the caller asked for, shading the UNION over the batch of the covered samples. With
``COMPOSITE_COMPACT_MERGE`` it sizes the window from the bound instead of from the caller's ceiling,
shades the covered ``(row, sample)`` pairs rather than their union, and gathers/blends/scatters only
the output pixels some row covers. Each of those rests on an argument about VALUES rather than about
approximation:

* a ray outside the projected bound cannot reach a mover, so ``t`` is ``INF``, ``cov`` is False, and
  the composite writes back the plate byte the broadcast already wrote;
* ``shade`` is elementwise in every argument, so shading a subset returns what the superset held;
* at an output pixel with ``covf == 0`` the blend is ``patch * 1.0 + 0.0`` on a float that came out
  of a uint8, which ``round`` returns and ``clamp(0,255)`` cannot move.

If any of those is wrong the symptom is a handful of pixels on the agent's silhouette -- which no
mean-absolute-difference catches -- so the threshold here is zero and the comparison is on the bytes.

``tests/gpu/test_fast_paths.py`` carries the same claim on the environments themselves. This file is
the differential fuzz on the scenes the environments do not reach: empty cover rows, fully covered
rows, ``ss = 1`` and ``ss = 2`` at ODD frame sizes, a window that does not fit and falls back to the
full frame, a caller-supplied ``out`` and a fresh one, per-environment colour and scale, and
primitives placed so the span lands on and over the tile boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
R3 = pytest.importorskip("stable_envs.render.raycast3d")


@pytest.fixture(autouse=True)
def _switch_is_on():
    """Refuse to run against a switch an earlier test leaked, which would compare a path with itself."""
    assert R3.COMPOSITE_COMPACT_MERGE is True, (
        "raycast3d.COMPOSITE_COMPACT_MERGE was already off when this test started, so every "
        "comparison below would be the window path against itself")
    assert R3.COVER_COMPACTION is True
    yield
    R3.COMPOSITE_COMPACT_MERGE = True
    R3.COVER_COMPACTION = True


def _camera():
    return R3.Camera(pos=(0.0, -2.6, 1.1), forward=(0.0, 1.0, -0.35), up=(0.0, 0.35, 1.0),
                     near=0.05, top=0.026, bottom=-0.026, center=0.0)


def _lights():
    return R3.Lights(dirs=np.array([[0.3, -0.5, 0.81], [-0.6, 0.2, 0.77]]),
                     ambient=np.array([0.25, 0.1]), diffuse=np.array([0.7, 0.4]),
                     specular=np.array([0.35, 0.2]), exponent=12.0)


def _prims(kinds, g):
    K = len(kinds)
    size = torch.rand(K, 3, generator=g).numpy() * 0.12 + 0.03
    return R3.Primitives(kind=np.asarray(kinds, int), size=size,
                         mat_rgb=torch.rand(K, 3, generator=g).numpy(),
                         tex_rgb=torch.rand(K, 3, generator=g).numpy() * 0.5 + 0.5,
                         spec=torch.rand(K, generator=g).numpy())


def _scene(seed, *, B=5, K=2, H=41, W=37, ss=2, spread=0.25, depth=0.0,
           kinds=(R3.SPHERE, R3.CAPSULE), plate_t_inf=False):
    """A plate, its ray parameter, and a batch of mover poses in front of the camera.

    ``plate_t`` is the analytic depth of a ground plane through the origin, so the composite's
    ``t < plate_t`` test is a real occlusion decision rather than a constant; ``plate_t_inf`` takes
    the other branch, where nothing static is in the way and ``INF < INF`` has to stay False.
    """
    g = torch.Generator().manual_seed(seed)
    cam = _camera()
    prims = _prims(kinds[:K], g)
    plate = (torch.rand(H, W, 3, generator=g) * 255).to(torch.uint8)
    if plate_t_inf:
        plate_t = torch.full((H * W,), R3.INF)
    else:
        floor = R3.Primitives(kind=np.array([R3.PLANE]), size=np.zeros((1, 3)),
                              mat_rgb=np.zeros((1, 3)), tex_rgb=np.ones((1, 3)), spec=np.zeros(1))
        plate_t = R3.static_depth(cam, floor, np.zeros((1, 3)), np.eye(3).reshape(1, 3, 3),
                                  H, W, "cpu")
    xpos = torch.zeros(B, K, 3)
    xpos[..., 0] = (torch.rand(B, K, generator=g) - 0.5) * spread
    xpos[..., 1] = (torch.rand(B, K, generator=g) - 0.5) * spread + depth
    xpos[..., 2] = torch.rand(B, K, generator=g) * spread + 0.12
    xmat = torch.eye(3).view(1, 1, 3, 3).expand(B, K, 3, 3).contiguous()
    return dict(plate=plate, plate_t=plate_t, cam=cam, prims=prims, xpos=xpos, xmat=xmat,
                lights=_lights(), H=H, W=W, supersample=ss)


def _both(sc, **kw):
    """``composite`` with the compact tail, then with the window it replaces. Same inputs."""
    R3.COMPOSITE_COMPACT_MERGE = True
    a = R3.composite(**sc, **kw).clone()
    R3.COMPOSITE_COMPACT_MERGE = False
    try:
        b = R3.composite(**sc, **kw).clone()
    finally:
        R3.COMPOSITE_COMPACT_MERGE = True
    return a, b


def _assert_identical(a, b, what):
    assert a.dtype == b.dtype == torch.uint8 and a.shape == b.shape, what
    if not torch.equal(a, b):
        diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
        n = int((diff > 0).sum())
        raise AssertionError(
            f"{what}: {n} of {diff.numel()} channel values differ, max |delta| {int(diff.max())}. "
            "The compact tail is meant to be the identity of the window path, so any difference "
            "is a defect in one of the three arguments in this file's docstring, not a tolerance.")


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
    ("plate-t-infinite", dict(seed=10, plate_t_inf=True), dict(tile=16)),
    ("box-and-cylinder", dict(seed=11, K=2, kinds=(R3.BOX, R3.CYLINDER)), dict(tile=16)),
    ("far-behind-camera", dict(seed=12, depth=-4.0), dict(tile=16)),
    ("mover-at-frame-edge", dict(seed=13, spread=0.05), dict(tile=16)),
]


@pytest.mark.parametrize("cid,skw,ckw", CASES, ids=[c[0] for c in CASES])
def test_the_compact_tail_is_bit_identical_to_the_window_it_replaces(cid, skw, ckw):
    sc = _scene(**skw)
    a, b = _both(sc, **ckw)
    _assert_identical(a, b, cid)


@pytest.mark.parametrize("seed", range(24))
def test_random_scenes_agree_on_every_byte(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    ri = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=g))     # noqa: E731
    H, W = ri(17, 48), ri(17, 48)
    ss = ri(1, 2)
    K = ri(1, 3)
    kinds = tuple([R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX][ri(0, 3)] for _ in range(K))
    sc = _scene(seed=2000 + seed, B=ri(1, 9), K=K, H=H, W=W, ss=ss, kinds=kinds,
                spread=float(torch.rand(1, generator=g)) * 1.2 + 0.05,
                depth=float(torch.rand(1, generator=g)) - 0.5,
                plate_t_inf=bool(ri(0, 1)))
    tile = ri(3, min(H, W) - 1)
    a, b = _both(sc, tile=tile)
    _assert_identical(a, b, f"random seed {seed} H={H} W={W} ss={ss} K={K} tile={tile}")


def _cover_fraction(sc, tile):
    """What fraction of the window each row covers -- the quantity the fuzz has to span."""
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


def test_the_fuzz_actually_reaches_empty_and_full_cover_rows():
    """A differential test that never varies the cover has verified one point, not an identity.

    ``compare_dumps`` in this workstream once printed ``ALL_IDENTICAL=True`` over two empty
    directories because ``all([])`` is True. The same shape of false green is available here: if
    every fuzz scene happened to put the movers off screen, both paths would return the plate and
    every assertion would pass having compared nothing.
    """
    empty = _scene(seed=41, K=1, kinds=(R3.SPHERE,), depth=-6.0, spread=0.02)
    assert _cover_fraction(empty, 16) == 0, "the empty-cover case is covering something"
    a, b = _both(empty, tile=16)
    _assert_identical(a, b, "empty cover")
    assert torch.equal(a, empty["plate"].view(1, *empty["plate"].shape)
                       .expand(a.shape[0], *empty["plate"].shape)), (
        "with nothing covered the composite must be the plate on every row")

    full = _scene(seed=42, K=1, kinds=(R3.BOX,), spread=0.02, plate_t_inf=True)
    full["xpos"][..., 2] = 0.35
    full["prims"] = R3.Primitives(kind=np.array([R3.BOX]), size=np.full((1, 3), 0.9),
                                  mat_rgb=np.array([[0.8, 0.2, 0.2]]), tex_rgb=np.ones((1, 3)),
                                  spec=np.array([0.4]))
    a, b = _both(full, tile=16)
    _assert_identical(a, b, "full cover")
    assert not torch.equal(a, full["plate"].view(1, *full["plate"].shape)
                           .expand(a.shape[0], *full["plate"].shape)), (
        "the full-cover case did not draw anything, so it is not testing the merge")


def test_a_caller_supplied_buffer_and_a_fresh_one_agree():
    """``maze2d`` reads ``with_t != plate`` to find the target, so every pixel of ``out`` is load-bearing."""
    sc = _scene(seed=51, B=6, K=1, kinds=(R3.SPHERE,))
    buf = torch.full((6, sc["H"], sc["W"], 3), 7, dtype=torch.uint8)
    R3.COMPOSITE_COMPACT_MERGE = True
    given = R3.composite(**sc, tile=16, out=buf).clone()
    fresh = R3.composite(**sc, tile=16).clone()
    _assert_identical(given, fresh, "caller buffer vs fresh")
    assert int((given == 7).all(-1).sum()) < given.shape[1] * given.shape[2] * given.shape[0], (
        "the caller's buffer was left at its fill value, so the plate broadcast did not happen")


def test_per_environment_colour_and_scale_are_still_per_environment():
    sc = _scene(seed=61, B=4, K=1, kinds=(R3.SPHERE,))
    mat = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 1.0]], [[1.0, 1.0, 0.0]]])
    scale = torch.tensor([0.6, 1.0, 1.4, 1.8])
    a, b = _both(sc, tile=16, mat_rgb=mat, size_scale=scale)
    _assert_identical(a, b, "per-env colour and scale")
    frames = [a[i] for i in range(4)]
    assert not all(torch.equal(frames[0], f) for f in frames[1:]), (
        "every row came out identical, so the per-environment appearance is not reaching the shade")


# --------------------------------------------------------------------------- the window choice

@pytest.mark.parametrize("seed", range(12))
def test_the_chosen_window_always_contains_the_span_it_was_derived_from(seed):
    """The whole identity rests on ``_fit_tile`` never returning a window smaller than the bound."""
    sc = _scene(seed=3000 + seed, B=7, K=2,
                spread=0.1 + 0.15 * seed, depth=0.2 * (seed % 3))
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
    assert lo_x >= 0 and lo_y >= 0 and fx.max() + t <= sc["W"] and fy.max() + t <= sc["H"], (
        "the window left the frame")
    # every row's own span box must sit inside its own window
    assert bool(((fx <= x0.clamp(min=0)) | (x0 < 0)).all()), "a row's window starts after its span"
    assert bool(((fy <= y0.clamp(min=0)) | (y0 < 0)).all()), "a row's window starts after its span"


def test_a_span_wider_than_the_ceiling_still_falls_back_to_the_full_frame():
    sc = _scene(seed=71, B=3, K=2, spread=2.0)
    assert R3._fit_tile(sc["cam"], sc["prims"], sc["xpos"], sc["H"], sc["W"], 4, "cpu") is None
    a, b = _both(sc, tile=4)
    _assert_identical(a, b, "fallback to the full frame")


def test_the_compact_tail_does_not_depend_on_cover_compaction_being_on():
    """The two switches are independent claims; the off-path of one must not silently disable the other."""
    sc = _scene(seed=81, B=5, K=1, kinds=(R3.SPHERE,))
    try:
        R3.COVER_COMPACTION = False
        a, b = _both(sc, tile=16)
    finally:
        R3.COVER_COMPACTION = True
    _assert_identical(a, b, "compact tail with COVER_COMPACTION off")


def test_the_switch_exists_and_defaults_on():
    """A renamed switch would leave every case above comparing the shipped path with itself."""
    assert isinstance(R3.COMPOSITE_COMPACT_MERGE, bool) and R3.COMPOSITE_COMPACT_MERGE
    assert callable(R3._composite_window) and callable(R3._fit_tile)
