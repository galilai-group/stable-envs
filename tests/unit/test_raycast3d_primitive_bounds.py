"""Per-primitive ray bounds leave ``raycast3d``'s CPU cast bit-identical.

The cast skips rays whose line passes further from a primitive than its bounding radius. That cannot
change an answer unless the bound is too small, which shows as a few silhouette pixels, so the
comparison is on bit patterns.

The envs are covered by ``tests/gpu/test_fast_paths.py``; this is a differential fuzz on scenes the
envs do not reach: rays tangent to a bound, primitives behind or beside the origin, degenerate
sizes, an origin inside the primitive, and non-unit ``d`` (the bound multiplies through by ``|d|^2``
so that ``d = 0`` cannot produce a NaN).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
R3 = pytest.importorskip("stable_envs.render.raycast3d")
from tests.helpers.raycast_reference import random_primitive_scene


def _both(fn, *a, **kw):
    return fn(*a, **kw, bounds=True), fn(*a, **kw, bounds=False)


def _same(a, b, what):
    if a.dtype.is_floating_point:
        ok = torch.equal(a.view(torch.int32), b.view(torch.int32))
    else:
        ok = torch.equal(a, b)
    if not ok:
        diff = a != b
        bad = int(diff.sum())
        mx = float((a.float() - b.float()).abs().max())
        raise AssertionError(
            f"{what}: bounded and unbounded casts differ on {bad} of {diff.numel()} values, "
            f"max {mx:.3e}. The bound excludes a ray the primitive can hit."
        )


CASES = [
    dict(rng=1, B=3, K=5, N=257),
    dict(rng=2, B=1, K=1, N=1024),
    dict(rng=3, B=4, K=9, N=64),
    dict(rng=4, B=2, K=5, N=333, degenerate=True),
    dict(rng=5, B=2, K=5, N=333, tangent=True),
    dict(rng=6, B=2, K=5, N=333, nonunit_d=True),
    dict(rng=7, B=2, K=7, N=200, degenerate=True, tangent=True, nonunit_d=True),
    # the far field, at three lever arms; `tangent` aims the rays AT the bound, which is where a
    # canceling distance test prunes a real hit.
    dict(rng=8, B=2, K=6, N=512, far=30.0, tangent=True),
    dict(rng=9, B=2, K=6, N=512, far=100.0, tangent=True),
    dict(rng=10, B=2, K=6, N=512, far=300.0, tangent=True),
    dict(rng=11, B=1, K=9, N=1024, far=100.0),
    # Pinned kind mixes: every kind alone, bounded and unbounded kinds adjacent, one of each.
    dict(rng=12, B=2, K=4, N=333, kinds=(R3.SPHERE,) * 4),
    dict(rng=13, B=2, K=4, N=333, kinds=(R3.CAPSULE,) * 4),
    dict(rng=14, B=2, K=4, N=333, kinds=(R3.PLANE,) * 4),
    dict(rng=15, B=2, K=6, N=333, kinds=(R3.SPHERE, R3.CAPSULE) * 3),
    dict(rng=16, B=2, K=6, N=333, kinds=(R3.CAPSULE, R3.SPHERE) * 3),
    dict(rng=17, B=2, K=5, N=333, kinds=(R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX)),
    dict(
        rng=18, B=2, K=5, N=512, far=100.0, tangent=True, kinds=(R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX)
    ),
    dict(rng=19, B=3, K=6, N=333, kinds=(R3.SPHERE, R3.CAPSULE) * 3, degenerate=True),
    # At the edge of the admission window at far=100: the cap-sphere test drops the leading |d|^2, so
    # the identity holds only for |d|^2 within _UNIT_D_TOL of 1. These feed the worst admitted bundle.
    dict(rng=20, B=2, K=6, N=512, far=100.0, tangent=True, dd_at_tol=+1),
    dict(rng=21, B=2, K=6, N=512, far=100.0, tangent=True, dd_at_tol=-1),
    dict(
        rng=22, B=2, K=5, N=512, far=100.0, dd_at_tol=+1, kinds=(R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX)
    ),
    # ELLIPSOID is unbounded by the table, so it takes the verbatim body beside bounded kinds; pinned
    # alone, mixed, degenerate, tangent-at-distance and at the admission edge.
    dict(rng=23, B=2, K=4, N=333, kinds=(R3.ELLIPSOID,) * 4),
    dict(rng=24, B=2, K=6, N=333, kinds=(R3.ELLIPSOID, R3.CAPSULE) * 3),
    dict(rng=25, B=3, K=6, N=333, kinds=(R3.ELLIPSOID, R3.SPHERE) * 3, degenerate=True),
    dict(
        rng=26,
        B=2,
        K=6,
        N=512,
        far=100.0,
        tangent=True,
        kinds=(R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX, R3.ELLIPSOID),
    ),
    dict(
        rng=27,
        B=2,
        K=6,
        N=512,
        far=100.0,
        dd_at_tol=-1,
        kinds=(R3.ELLIPSOID, R3.CAPSULE, R3.ELLIPSOID, R3.BOX, R3.ELLIPSOID, R3.CYLINDER),
    ),
]


def _case_id(c):
    tag = [f"B{c['B']}K{c['K']}N{c['N']}"]
    tag += [k for k in ("degenerate", "tangent", "nonunit_d") if c.get(k)]
    if c.get("dd_at_tol"):
        tag.append("ddtol" + ("hi" if c["dd_at_tol"] > 0 else "lo"))
    if c.get("far"):
        tag.append(f"far{int(c['far'])}")
    if c.get("kinds"):
        names = {
            R3.PLANE: "pl",
            R3.SPHERE: "sp",
            R3.CAPSULE: "ca",
            R3.CYLINDER: "cy",
            R3.BOX: "bx",
            R3.ELLIPSOID: "el",
        }
        tag.append("".join(names[k] for k in c["kinds"]))
    return "-".join(tag)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_cast_nearest_is_unchanged_by_the_bound(case):
    kinds, sizes, pos, R, o, d = random_primitive_scene(**case)
    (bt, bn, bi), (st, sn, si) = _both(R3.cast_nearest_torch, kinds, sizes, pos, R, o, d)
    _same(bt, st, "t")
    _same(bn, sn, "normal")
    _same(bi, si, "winning primitive index")


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_any_blocking_is_unchanged_by_the_bound(case):
    kinds, sizes, pos, R, o, d = random_primitive_scene(**case)
    B, N = o.shape[0], o.shape[1]
    for lim_kind in ("finite", "infinite"):
        lim = torch.full((B, N), R3.INF) if lim_kind == "infinite" else torch.rand(B, N) * 6.0
        for skip in (None, torch.randint(0, max(1, len(kinds)), (B, N))):
            b, s = _both(R3.any_blocking_torch, kinds, sizes, pos, R, o, d, lim, skip)
            _same(b, s, f"blocked ({lim_kind} lim, skip={'yes' if skip is not None else 'no'})")


def test_a_ray_aimed_straight_at_a_primitive_is_never_pruned():
    """The bound keeps the rays that hit; the test fails if the bound prunes everything."""
    kinds = [R3.SPHERE, R3.CAPSULE]
    sizes = torch.tensor([[[0.2, 0.0, 0.0], [0.1, 0.3, 0.0]]])
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    R = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3).contiguous()
    o = torch.tensor([[[0.0, -3.0, 0.0], [1.0, -3.0, 0.0]]])
    d = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    t, _, idx = R3.cast_nearest_torch(kinds, sizes, pos, R, o, d)
    assert torch.isfinite(t).all() and (t < R3.INF).all(), f"a direct hit was pruned: t={t}"
    assert idx.tolist() == [[0, 1]], f"wrong primitive won: {idx.tolist()}"


def test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose():
    """``_BOUNDED_KINDS`` equals the predicate in the committed intersector-cost measurement."""
    import json
    from pathlib import Path

    root = Path(R3.__file__).resolve().parents[2] / "tests/fixtures/measurements/raycast3d"
    f = root / "r6_kind_cost.json"
    if not f.is_file():
        pytest.skip(f"no committed measurement at {f}")
    m = json.loads(f.read_text())
    names = {
        "PLANE": R3.PLANE,
        "SPHERE": R3.SPHERE,
        "CAPSULE": R3.CAPSULE,
        "CYLINDER": R3.CYLINDER,
        "BOX": R3.BOX,
        "ELLIPSOID": R3.ELLIPSOID,
    }
    measured = {names[n] for n, on in m["predicate"].items() if on}
    # ELLIPSOID is priced in its own committed aggregate by the same rule; absent that file the kind is
    # unbounded and must not be in the table.
    f7 = Path(R3.__file__).resolve().parents[2] / "tests/fixtures/measurements/raycast3d/kind_cost_ellipsoid.json"
    if f7.is_file():
        m7 = json.loads(f7.read_text())
        if m7["predicate"].get("ELLIPSOID"):
            measured.add(R3.ELLIPSOID)
        m["ratio_worst_per_run"]["ELLIPSOID"] = m7["ratio_worst_per_run"]["ELLIPSOID"]
        m["ratio_best_per_run"]["ELLIPSOID"] = m7["ratio_best_per_run"]["ELLIPSOID"]
        m["predicate"]["ELLIPSOID"] = bool(m7["predicate"].get("ELLIPSOID"))
    assert set(R3._BOUNDED_KINDS) == measured, (
        f"_BOUNDED_KINDS is {sorted(R3._BOUNDED_KINDS)} but the committed measurement "
        f"({f.name}) chose {sorted(measured)}"
    )
    # The rule must hold in the numbers, not only the summary field: bounded iff the worst ratio is
    # > 1 in every run.
    for kind, on in m["predicate"].items():
        if kind not in m["ratio_worst_per_run"]:
            continue
        worst = m["ratio_worst_per_run"][kind]
        if on:
            assert min(worst) > 1.0, f"{kind} is bounded but its worst ratio is {min(worst):.2f}"
        else:
            assert min(worst) <= 1.0, f"{kind} is unbounded but its worst ratio is > 1 in every run ({min(worst):.2f})"


def test_an_unbounded_kind_never_reaches_the_gather():
    """A kind outside ``_BOUNDED_KINDS`` skips the bound's ``nonzero``/gather entirely.

    Pixels are identical either way, so this asserts on the call.
    """
    seen = []
    real = torch.nonzero

    def spy(*a, **kw):
        seen.append(1)
        return real(*a, **kw)

    kinds, sizes, pos, R, o, d = random_primitive_scene(rng=31, B=2, K=3, N=257, kinds=(R3.SPHERE,) * 3)
    torch.nonzero = spy
    try:
        R3.cast_nearest_torch(kinds, sizes, pos, R, o, d)
        assert not seen or R3.SPHERE in R3._BOUNDED_KINDS, (
            "an unbounded kind still went through the bound's nonzero/gather"
        )
        kinds2, s2, p2, R2, o2, d2 = random_primitive_scene(rng=31, B=2, K=3, N=257, kinds=(R3.CAPSULE,) * 3)
        seen.clear()
        R3.cast_nearest_torch(kinds2, s2, p2, R2, o2, d2)
        assert seen or R3.CAPSULE not in R3._BOUNDED_KINDS, (
            "a bounded kind did not go through the bound at all; the test would pass vacuously"
        )
    finally:
        torch.nonzero = real


@pytest.mark.parametrize("rng", [21, 22, 23])
def test_a_row_is_unchanged_by_the_rows_it_is_batched_WITH(rng):
    """Row 0's result does not depend on rows 1..7.

    A primitive's ray set is the union over the batch, so batching changes which rays reach the
    intersector; the answer must not change. This compares one row against itself at B=1 and B=8.
    """
    kinds, s1, p1, R1, o1, d1 = random_primitive_scene(rng=rng, B=1, K=6, N=311)
    # Rows 1..7 are a different scene and different rays, so the union is wider.
    _, s8, p8, R8, o8, d8 = random_primitive_scene(rng=rng + 100, B=8, K=6, N=311)

    def cat(one, many):  # row 0 from the B=1 case, rows 1..7 from the other
        out = torch.cat([one, many[1:]], 0).contiguous()
        assert out.shape[0] == 8 and torch.equal(out[0], one[0])
        return out

    s8, p8, R8 = cat(s1, s8), cat(p1, p8), cat(R1, R8)
    o8, d8 = cat(o1, o8), cat(d1, d8)

    t1, n1, i1 = R3.cast_nearest_torch(kinds, s1, p1, R1, o1, d1)
    t8, n8, i8 = R3.cast_nearest_torch(kinds, s8, p8, R8, o8, d8)
    _same(t1[0], t8[0], "t, row 0 alone vs row 0 in a batch of 8")
    _same(n1[0], n8[0], "normal, row 0 alone vs row 0 in a batch of 8")
    _same(i1[0], i8[0], "index, row 0 alone vs row 0 in a batch of 8")

    lim1 = torch.rand(1, 311) * 6.0
    lim8 = torch.cat([lim1, torch.rand(7, 311) * 6.0], 0).contiguous()
    b1 = R3.any_blocking_torch(kinds, s1, p1, R1, o1, d1, lim1)
    b8 = R3.any_blocking_torch(kinds, s8, p8, R8, o8, d8, lim8)
    _same(b1[0], b8[0], "blocked, row 0 alone vs row 0 in a batch of 8")


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize(
    "backend,installed,reason",
    [
        ("typo", False, "unknown raycast backend"),
        ("triton", False, "Triton is not installed"),
        ("triton", True, "rays must be on CUDA"),
    ],
)
def test_backend_errors_are_explicit(monkeypatch, blocking, backend, installed, reason):
    args = random_primitive_scene(rng=1, B=2, K=3, N=17)
    fn = R3.any_blocking if blocking else R3.cast_nearest
    if blocking:
        args += (torch.full((2, 17), 3.0),)
    monkeypatch.setattr(R3, "HAVE_TRITON", installed)
    with pytest.raises(ValueError, match=reason):
        fn(*args, backend=backend)


@pytest.mark.parametrize("blocking", [False, True])
def test_auto_stays_on_torch_for_cpu_even_with_triton_importable(monkeypatch, blocking):
    args = random_primitive_scene(rng=2, B=2, K=3, N=17)
    fn = R3.any_blocking if blocking else R3.cast_nearest
    if blocking:
        args += (torch.full((2, 17), 3.0),)
    monkeypatch.setattr(R3, "HAVE_TRITON", True)

    def forbidden(*args):
        raise AssertionError("CPU inputs reached the CUDA launch")

    monkeypatch.setattr(R3, "_launch", forbidden)
    actual, expected = fn(*args), fn(*args, backend="torch")
    if blocking:
        assert torch.equal(actual, expected)
    else:
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)
