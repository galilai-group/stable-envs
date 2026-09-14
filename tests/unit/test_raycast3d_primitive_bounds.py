"""Per-primitive bounds must be the IDENTITY, not an approximation of it.

``raycast3d``'s CPU cast used to evaluate every primitive against every ray. It now skips the rays
whose line passes further from a primitive's origin than that primitive's bounding radius, which
cannot change an answer — such a ray gets ``t = INF`` and loses the ``t < best`` test — but which CAN
change one if the bound is ever computed too small. The failure mode is a handful of pixels on a
silhouette, which no mean-absolute-difference catches, so the threshold here is zero and the
comparison is on bit patterns.

The envs are covered by ``tests/gpu/test_fast_paths.py``; this file is the differential fuzz, on
scenes the envs do not reach: rays exactly tangent to a bound, primitives entirely behind the origin
or off to the side, degenerate sizes, an origin inside the primitive, and a ``d`` that is not unit
(the bound's derivation divides by ``|d|^2``, and the shipped form multiplies through instead
precisely so that ``d = 0`` cannot produce a NaN that silently drops a ray).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
R3 = pytest.importorskip("stable_envs.render.raycast3d")

KINDS = (R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX)


def _rand_scene(rng, B, K, N, *, degenerate=False, nonunit_d=False, tangent=False, far=0.0,
                kinds=None):
    g = torch.Generator().manual_seed(rng)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float32)          # noqa: E731
    # ``kinds`` PINNED rather than sampled is what makes the per-kind dispatch testable. RAYCAST-5
    # applies the bound only to the kinds whose intersector costs more than the bound does, so a
    # scene now mixes primitives that take the bounded path with primitives that take the verbatim
    # unbounded body, and they write into the same ``best`` / ``blocked``. Sampling reaches those
    # mixes eventually; pinning them means the boundary is covered on every run.
    kinds = ([KINDS[int(v)] for v in torch.randint(0, len(KINDS), (K,), generator=g)]
             if kinds is None else list(kinds))
    K = len(kinds)
    sizes = (r(B, K, 3) * 0.4 + 0.02)
    if degenerate:
        sizes[:, ::2] = 0.0                                   # zero radius / half-length / extent
        sizes[0, 0] = 0.0
    pos = (r(B, K, 3) - 0.5) * 1.5
    q = torch.randn(B, K, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    -1).view(B, K, 3, 3)
    # ``far`` puts the camera metres away from a small cluster, which is the GIANT maze's geometry:
    # ~100 m to a 5 cm capsule. That is the regime the first version of the bound got wrong, and the
    # regime this file did not cover until the cluster found it -- |oc|^2|d|^2 and (oc.d)^2 are each
    # ~1e4 there and differ by ~r^2 ~ 2.5e-3, which is float32's resolution at that magnitude, so the
    # subtraction cancelled away exactly the quantity it was meant to produce and one true hit was
    # pruned. Without this case the suite passed and one pixel of 3,211,264 still moved.
    o = ((r(B, N, 3) - 0.5) * 0.2 + torch.tensor([0.0, -2.0, 0.4])).contiguous()
    if far:
        sizes = sizes * 0.05                              # centimetre-scale primitives...
        pos = pos * 0.05
        o = o + torch.tensor([0.0, -float(far), 0.35 * float(far)])      # ...seen from far away
    if tangent:
        # aim a third of the rays exactly at each primitive's bound radius: the grazing case, which is
        # where a bound that is one ulp too small drops a pixel.
        tgt = pos[:, torch.randint(0, K, (N,), generator=g)]
        rad = (sizes[:, :, 0] + sizes[:, :, 1]).mean()
        off = torch.nn.functional.normalize(torch.randn(B, N, 3, generator=g), dim=-1)
        tgt = tgt + off * rad
        d = torch.nn.functional.normalize(tgt - o, dim=-1)
    else:
        d = torch.nn.functional.normalize((r(B, N, 3) - 0.5) + torch.tensor([0.0, 1.0, 0.0]), dim=-1)
    if nonunit_d:
        d = d * (r(B, N, 1) * 3.0)                            # includes exact zeros
    return kinds, sizes.contiguous(), pos.contiguous(), R.contiguous(), o, d.contiguous()


def _both(fn, *a, **kw):
    """Run ``fn`` with the bound on and with it off, restoring whatever the default was.

    The default is NOT assumed here. RAYCAST-4 shipped with ``PRIMITIVE_BOUNDS`` on; RAYCAST-5 turned
    it off while the maze regression is open and will turn it back on when the sweep says it may.
    A differential test that asserts the default is testing the release decision rather than the
    identity, and would have to be edited every time that decision moves -- which is exactly when it
    most needs to keep working.
    """
    was = R3.PRIMITIVE_BOUNDS
    try:
        R3.PRIMITIVE_BOUNDS = True
        fast = fn(*a, **kw)
        R3.PRIMITIVE_BOUNDS = False
        slow = fn(*a, **kw)
    finally:
        R3.PRIMITIVE_BOUNDS = was
    return fast, slow


def _same(a, b, what):
    if a.dtype.is_floating_point:
        ok = torch.equal(a.view(torch.int32), b.view(torch.int32))
    else:
        ok = torch.equal(a, b)
    if not ok:
        diff = (a != b)
        bad = int(diff.sum())
        mx = float((a.float() - b.float()).abs().max())
        raise AssertionError(
            f"{what}: bounded and unbounded casts differ on {bad} of {diff.numel()} values, "
            f"max {mx:.3e}. A bound that excludes a ray the primitive can actually hit drops a "
            f"pixel, and it drops it on a silhouette where nothing else will notice.")


CASES = [
    dict(rng=1, B=3, K=5, N=257),
    dict(rng=2, B=1, K=1, N=1024),
    dict(rng=3, B=4, K=9, N=64),
    dict(rng=4, B=2, K=5, N=333, degenerate=True),
    dict(rng=5, B=2, K=5, N=333, tangent=True),
    dict(rng=6, B=2, K=5, N=333, nonunit_d=True),
    dict(rng=7, B=2, K=7, N=200, degenerate=True, tangent=True, nonunit_d=True),
    # the far field, at three lever arms; `tangent` aims the rays AT the bound, which is where a
    # cancelling distance test prunes a real hit.
    dict(rng=8, B=2, K=6, N=512, far=30.0, tangent=True),
    dict(rng=9, B=2, K=6, N=512, far=100.0, tangent=True),
    dict(rng=10, B=2, K=6, N=512, far=300.0, tangent=True),
    dict(rng=11, B=1, K=9, N=1024, far=100.0),
    # PINNED KIND MIXES (RAYCAST-5). Under the per-kind predicate a scene can hold primitives on both
    # paths at once, so these pin the combinations rather than hoping the sampler finds them: every
    # kind alone, the two sides of the predicate adjacent, and a scene with one of each.
    dict(rng=12, B=2, K=4, N=333, kinds=(R3.SPHERE,) * 4),
    dict(rng=13, B=2, K=4, N=333, kinds=(R3.CAPSULE,) * 4),
    dict(rng=14, B=2, K=4, N=333, kinds=(R3.PLANE,) * 4),
    dict(rng=15, B=2, K=6, N=333, kinds=(R3.SPHERE, R3.CAPSULE) * 3),
    dict(rng=16, B=2, K=6, N=333, kinds=(R3.CAPSULE, R3.SPHERE) * 3),
    dict(rng=17, B=2, K=5, N=333, kinds=(R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX)),
    dict(rng=18, B=2, K=5, N=512, far=100.0, tangent=True,
         kinds=(R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX)),
    dict(rng=19, B=3, K=6, N=333, kinds=(R3.SPHERE, R3.CAPSULE) * 3, degenerate=True),
]


def _case_id(c):
    tag = [f"B{c['B']}K{c['K']}N{c['N']}"]
    tag += [k for k in ("degenerate", "tangent", "nonunit_d") if c.get(k)]
    if c.get("far"):
        tag.append(f"far{int(c['far'])}")
    if c.get("kinds"):
        names = {R3.PLANE: "pl", R3.SPHERE: "sp", R3.CAPSULE: "ca", R3.CYLINDER: "cy", R3.BOX: "bx"}
        tag.append("".join(names[k] for k in c["kinds"]))
    return "-".join(tag)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_cast_nearest_is_unchanged_by_the_bound(case):
    kinds, sizes, pos, R, o, d = _rand_scene(**case)
    (bt, bn, bi), (st, sn, si) = _both(R3.cast_nearest_torch, kinds, sizes, pos, R, o, d)
    _same(bt, st, "t")
    _same(bn, sn, "normal")
    _same(bi, si, "winning primitive index")


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_any_blocking_is_unchanged_by_the_bound(case):
    kinds, sizes, pos, R, o, d = _rand_scene(**case)
    B, N = o.shape[0], o.shape[1]
    for lim_kind in ("finite", "infinite"):
        lim = (torch.full((B, N), R3.INF) if lim_kind == "infinite"
               else torch.rand(B, N) * 6.0)
        for skip in (None, torch.randint(0, max(1, len(kinds)), (B, N))):
            b, s = _both(R3.any_blocking_torch, kinds, sizes, pos, R, o, d, lim, skip)
            _same(b, s, f"blocked ({lim_kind} lim, skip={'yes' if skip is not None else 'no'})")


def test_a_ray_aimed_straight_at_a_primitive_is_never_pruned():
    """The bound must keep the rays that obviously hit — a test that fails if it prunes everything."""
    kinds = [R3.SPHERE, R3.CAPSULE]
    sizes = torch.tensor([[[0.2, 0.0, 0.0], [0.1, 0.3, 0.0]]])
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    R = torch.eye(3).view(1, 1, 3, 3).expand(1, 2, 3, 3).contiguous()
    o = torch.tensor([[[0.0, -3.0, 0.0], [1.0, -3.0, 0.0]]])
    d = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    t, _, idx = R3.cast_nearest_torch(kinds, sizes, pos, R, o, d)
    assert torch.isfinite(t).all() and (t < R3.INF).all(), f"a direct hit was pruned: t={t}"
    assert idx.tolist() == [[0, 1]], f"wrong primitive won: {idx.tolist()}"


def test_the_switch_exists_and_both_paths_are_reachable():
    assert isinstance(R3.PRIMITIVE_BOUNDS, bool)
    assert callable(R3._cast_nearest_torch_unbounded)
    assert callable(R3._any_blocking_torch_unbounded)


def test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose():
    """``_BOUNDED_KINDS`` must equal the table in the experiment file, not a hand-edited guess.

    The predicate is the deliverable of RAYCAST-5 and its whole claim is that it was derived from a
    measurement of intersector cost rather than fitted to eight sweep rows. A constant in the module
    that has drifted from the JSON it cites would leave that claim true of the document and false of
    the code, which is the failure this check exists to make impossible. If the measurement is re-run
    and the table moves, this test is what says the constant has to move with it.
    """
    import json
    from pathlib import Path
    f = (Path(R3.__file__).resolve().parents[2]
         / "examples/results/step6/raycast3d/experiments/r5_kind_cost.json")
    if not f.is_file():
        pytest.skip(f"no committed measurement at {f}")
    m = json.loads(f.read_text())
    names = {"PLANE": R3.PLANE, "SPHERE": R3.SPHERE, "CAPSULE": R3.CAPSULE,
             "CYLINDER": R3.CYLINDER, "BOX": R3.BOX}
    measured = {names[n] for n, on in m["predicate"].items() if on}
    assert set(R3._BOUNDED_KINDS) == measured, (
        f"_BOUNDED_KINDS is {sorted(R3._BOUNDED_KINDS)} but the committed measurement "
        f"({f.name}, margin {m['margin']['value']:.3f}) chose {sorted(measured)}")


def test_an_unbounded_kind_never_reaches_the_gather():
    """A kind outside the table must take the verbatim unbounded body, bound or no bound.

    Pointmaze's regression was one SPHERE over a 200,704-ray bundle: the bound spent an O(N)
    ``nonzero`` plus gather plus scatter to narrow the cheapest intersector in the file. The repair is
    only real if the sphere genuinely stops going through that machinery, so this asserts on the call
    rather than on the pixels -- pixels are identical either way, which is precisely why they cannot
    detect the thing being fixed.
    """
    seen = []
    real = torch.nonzero

    def spy(*a, **kw):
        seen.append(1)
        return real(*a, **kw)

    kinds, sizes, pos, R, o, d = _rand_scene(rng=31, B=2, K=3, N=257, kinds=(R3.SPHERE,) * 3)
    was = R3.PRIMITIVE_BOUNDS
    torch.nonzero = spy
    try:
        R3.PRIMITIVE_BOUNDS = True
        R3.cast_nearest_torch(kinds, sizes, pos, R, o, d)
        assert not seen or R3.SPHERE in R3._BOUNDED_KINDS, (
            "an unbounded kind still went through the bound's nonzero/gather")
        kinds2, s2, p2, R2, o2, d2 = _rand_scene(rng=31, B=2, K=3, N=257, kinds=(R3.CAPSULE,) * 3)
        seen.clear()
        R3.cast_nearest_torch(kinds2, s2, p2, R2, o2, d2)
        assert seen or R3.CAPSULE not in R3._BOUNDED_KINDS, (
            "a bounded kind did not go through the bound at all; the test would pass vacuously")
    finally:
        torch.nonzero = real
        R3.PRIMITIVE_BOUNDS = was


@pytest.mark.parametrize("rng", [21, 22, 23])
def test_a_row_is_unchanged_by_the_rows_it_is_batched_WITH(rng):
    """Row 0's pixels must not depend on rows 1..7, and for THIS design that is not free.

    The bound's ray set for a primitive is the union over the batch -- the rays near it in ANY row --
    because a per-row set would need a ragged gather. So batching changes WHICH rays reach the
    intersector, and the claim that it does not change the ANSWER is a claim about the bound being
    conservative, not an identity anyone can read off the code. The suite above compares bounded
    against unbounded at a fixed B; this compares one row against ITSELF at two different B, which is
    the property a dataset loader actually depends on and the one a too-tight bound would break
    asymmetrically -- row 0 alone would lose a hit that row 0 in company keeps.
    """
    kinds, s1, p1, R1, o1, d1 = _rand_scene(rng=rng, B=1, K=6, N=311)
    # Rows 1..7 are a DIFFERENT scene and different rays, so the union is genuinely wider.
    _, s8, p8, R8, o8, d8 = _rand_scene(rng=rng + 100, B=8, K=6, N=311)
    def cat(one, many):                       # row 0 from the B=1 case, rows 1..7 from the other
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
