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


def _rand_scene(rng, B, K, N, *, degenerate=False, nonunit_d=False, tangent=False, far=0.0):
    g = torch.Generator().manual_seed(rng)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float32)          # noqa: E731
    kinds = [KINDS[int(v)] for v in torch.randint(0, len(KINDS), (K,), generator=g)]
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
    assert R3.PRIMITIVE_BOUNDS is True, "a previous test leaked the switch"
    try:
        R3.PRIMITIVE_BOUNDS = True
        fast = fn(*a, **kw)
        R3.PRIMITIVE_BOUNDS = False
        slow = fn(*a, **kw)
    finally:
        R3.PRIMITIVE_BOUNDS = True
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
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(
    [f"B{c['B']}K{c['K']}N{c['N']}"] + [k for k in ("degenerate", "tangent", "nonunit_d") if c.get(k)]
    + ([f"far{int(c['far'])}"] if c.get("far") else [])))
def test_cast_nearest_is_unchanged_by_the_bound(case):
    kinds, sizes, pos, R, o, d = _rand_scene(**case)
    (bt, bn, bi), (st, sn, si) = _both(R3.cast_nearest_torch, kinds, sizes, pos, R, o, d)
    _same(bt, st, "t")
    _same(bn, sn, "normal")
    _same(bi, si, "winning primitive index")


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(
    [f"B{c['B']}K{c['K']}N{c['N']}"] + [k for k in ("degenerate", "tangent", "nonunit_d") if c.get(k)]
    + ([f"far{int(c['far'])}"] if c.get("far") else [])))
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


def test_the_switch_exists_and_defaults_on():
    assert R3.PRIMITIVE_BOUNDS is True
    assert callable(R3._cast_nearest_torch_unbounded)
    assert callable(R3._any_blocking_torch_unbounded)
