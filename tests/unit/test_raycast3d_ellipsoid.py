"""``raycast3d._ellipsoid`` against closed forms.

(1) Axis-aligned rays, an axis-permuting rotation and an origin inside give the written-down hit
distance and normal. (2) Against a float64 reference (MuJoCo's ``1/size^2``-weighted quadratic,
``mujoco_warp/_src/ray.py::ray_ellipsoid``) it agrees on hit/miss and to float32 precision on ``t``
and the normal, over random rotated scenes. (3) Three equal semi-axes give ``_sphere``; a zero
semi-axis is a miss, never a NaN. Torch/Triton bit-identity is checked in
``tests/gpu/test_fast_paths.py::test_the_ellipsoid_twins_are_bit_identical_on_both_casts``.
"""

from __future__ import annotations

import numpy as np
import pytest

from stable_envs.render.primitives import bound_radius_tensor

torch = pytest.importorskip("torch")
R3 = pytest.importorskip("stable_envs.render.raycast3d")


def _rot(axis: str, deg: float) -> torch.Tensor:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = {
        "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }[axis]
    return torch.tensor(m, dtype=torch.float32)


def _cast(o, d, pos, R, size):
    """One ellipsoid against ``(N,3)`` rays through the public dispatch."""
    N = o.shape[0]
    t, n = R3.intersect(
        R3.ELLIPSOID, size.view(1, 3).expand(N, 3), pos.view(1, 3).expand(N, 3), R.view(1, 3, 3).expand(N, 3, 3), o, d
    )
    return t, n


def test_axis_aligned_rays_hit_where_the_semi_axes_say():
    size = torch.tensor([0.3, 0.27, 0.2])
    pos = torch.zeros(3)
    R = torch.eye(3)
    o = torch.tensor([[2.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0], [-1.5, 0.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t, n = _cast(o, d, pos, R, size)
    want_t = torch.tensor([2.0 - 0.3, 2.0 - 0.27, 3.0 - 0.2, 1.5 - 0.3])
    want_n = -d  # the outward normal faces the ray back
    assert torch.allclose(t, want_t, atol=1e-6), t
    assert torch.allclose(n, want_n, atol=1e-6), n


def test_a_rotation_that_permutes_the_axes_moves_the_hit_with_them():
    """Rotate the (a, b, c) ellipsoid by 90 degrees about z: a ray along -x now meets semi-axis b."""
    size = torch.tensor([0.3, 0.1, 0.2])
    pos = torch.tensor([0.5, -0.25, 1.0])
    R = _rot("z", 90.0)
    o = pos + torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    d = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    t, n = _cast(o, d, pos, R, size)
    assert torch.allclose(t, torch.tensor([2.0 - 0.1, 2.0 - 0.3]), atol=1e-6), t
    assert torch.allclose(n, -d, atol=1e-6), n


def test_an_origin_inside_takes_the_far_root_and_a_miss_is_INF():
    size = torch.tensor([0.3, 0.27, 0.2])
    pos = torch.zeros(3)
    R = torch.eye(3)
    o = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    t, n = _cast(o, d, pos, R, size)
    assert torch.allclose(t[[0, 1, 3]], torch.tensor([0.3, 0.2, 0.2]), atol=1e-6), t
    assert torch.allclose(n[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)  # outward, at the exit
    assert torch.allclose(n[3], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert t[2] == R3.INF  # passes beside it
    assert torch.isfinite(n).all()


def _reference(o, d, pos, R, size):
    """MuJoCo's formulation in float64: ``(x lvec + lpnt)' diag(1/size^2) (x lvec + lpnt) = 1``."""
    o64, d64, R64 = (x.double().numpy() for x in (o, d, R))
    p64, s64 = pos.double().numpy(), size.double().numpy()
    lp = (o64 - p64) @ R64
    lv = d64 @ R64
    w = 1.0 / s64**2
    a = (lv * lv * w).sum(-1)
    b = (lv * lp * w).sum(-1)
    c = (lp * lp * w).sum(-1) - 1.0
    disc = b * b - a * c
    sq = np.sqrt(np.clip(disc, 0, None))
    t0 = (-b - sq) / a
    t1 = (-b + sq) / a
    t = np.where(t0 > 0, t0, t1)
    hit = (disc > 0) & (t > 0)
    l = lp + lv * t[:, None]
    n = l * w
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    n = n @ R64.T
    return hit, t, n


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fuzz_against_the_float64_reference(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    size = r(3) * 0.4 + 0.02
    pos = (r(3) - 0.5) * 1.5
    q = torch.randn(4, generator=g)
    q = q / q.norm()
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ]
    ).view(3, 3)
    N = 20000
    o = (r(N, 3) - 0.5) * 0.4 + torch.tensor([0.0, -2.0, 0.4])
    # a third of the rays aimed at the ellipsoid's neighbourhood so the hit set is not empty
    aim = pos + (r(N, 3) - 0.5) * 2 * size.max()
    d = torch.nn.functional.normalize(
        torch.where(r(N, 1) < 0.34, aim - o, r(N, 3) - 0.5 + torch.tensor([0.0, 1.0, 0.0])), dim=-1
    )
    t, n = _cast(o, d, pos, R, size)
    hit_ref, t_ref, n_ref = _reference(o, d, pos, R, size)
    hit = (t < R3.INF).numpy()
    # hit/miss may legitimately differ only on grazing rays, where disc is within float32 noise of 0
    disc_margin = np.abs(_disc(o, d, pos, R, size))
    decided = disc_margin > 1e-5
    assert (hit[decided] == hit_ref[decided]).all(), "hit/miss disagrees away from grazing incidence"
    both = hit & hit_ref & decided
    assert both.sum() > 1000, "the fuzz did not reach the ellipsoid"
    # Derived tolerance: ``disc = b*b - a*c`` cancels in float32 with terms of order
    # ``|ol'|^2 = (|o - pos| / size_min)^2``, so the relative error in ``t`` is ``~eps * |ol'|^2``.
    # The bound is 16 ulp of that (``_BOUND_SLACK_NOISE``), about 30x the worst measured error.
    cond = 16.0 * 2.0**-23 * float(((o - pos).norm(dim=-1).max() / size.min()) ** 2)
    rel = np.abs(t.numpy()[both] - t_ref[both]) / t_ref[both]
    assert rel.max() <= cond, (rel.max(), cond)
    # the normal is checked AT THE RETURNED POINT, so the check is of the normal formula and not of
    # the ``t`` error propagated through the surface's curvature
    p = (o + d * t.view(-1, 1)).double().numpy()[both]
    lp = (p - pos.double().numpy()) @ R.double().numpy()
    n_at_p = lp / size.double().numpy() ** 2
    n_at_p = (n_at_p / np.linalg.norm(n_at_p, axis=-1, keepdims=True)) @ R.double().numpy().T
    assert np.abs(n.numpy()[both] - n_at_p).max() <= 1e-4
    assert np.abs(n_ref[both] - n_at_p).max() <= 10 * cond**0.5  # curvature x sqrt(t error)


def _disc(o, d, pos, R, size):
    o64, d64, R64 = (x.double().numpy() for x in (o, d, R))
    lp = (o64 - pos.double().numpy()) @ R64
    lv = d64 @ R64
    w = 1.0 / size.double().numpy() ** 2
    a = (lv * lv * w).sum(-1)
    b = (lv * lp * w).sum(-1)
    c = (lp * lp * w).sum(-1) - 1.0
    return b * b - a * c


def test_three_equal_semi_axes_is_the_sphere():
    g = torch.Generator().manual_seed(11)
    r = 0.23
    pos = torch.tensor([0.1, -0.2, 0.3])
    N = 5000
    o = (torch.rand(N, 3, generator=g) - 0.5) * 0.3 + torch.tensor([0.0, -1.5, 0.2])
    d = torch.nn.functional.normalize(pos - o + (torch.rand(N, 3, generator=g) - 0.5) * 0.6, dim=-1)
    te, ne = _cast(o, d, pos, _rot("x", 37.0), torch.tensor([r, r, r]))
    ts, ns = R3._sphere(o, d, pos.view(1, 3).expand(N, 3), torch.full((N,), r))
    disc_margin = np.abs(_disc(o, d, pos, torch.eye(3), torch.tensor([r, r, r])))
    decided = torch.as_tensor(disc_margin > 1e-5)
    assert torch.equal((te < R3.INF)[decided], (ts < R3.INF)[decided])
    both = (te < R3.INF) & (ts < R3.INF) & decided
    assert both.sum() > 1000
    # same derived bound as the fuzz: two spellings of one quadratic, each with its own rounding
    cond = 16.0 * 2.0**-23 * float(((o - pos).norm(dim=-1).max() / r) ** 2)
    assert float(((te - ts).abs() / ts)[both].max()) <= cond
    assert float((ne - ns).abs()[both].max()) <= 10 * cond**0.5


def test_a_zero_semi_axis_is_a_miss_and_never_a_nan():
    o = torch.tensor([[0.0, -2.0, 0.0]] * 3)
    d = torch.tensor([[0.0, 1.0, 0.0]] * 3)
    for size in ([0.0, 0.2, 0.2], [0.2, 0.0, 0.2], [0.2, 0.2, 0.0], [0.0, 0.0, 0.0]):
        t, n = _cast(o, d, torch.zeros(3), torch.eye(3), torch.tensor(size))
        assert bool((t == R3.INF).all()), size
        assert bool(torch.isfinite(n).all()), size


def test_the_dispatch_and_the_bounds_know_the_kind():
    assert R3.ELLIPSOID == 4
    size = torch.tensor([[[0.3, 0.27, 0.2]]])
    assert float(bound_radius_tensor(R3.ELLIPSOID, size)) == pytest.approx(0.3)
    prims = R3.Primitives(
        kind=np.array([R3.ELLIPSOID, R3.SPHERE]),
        size=np.array([[0.3, 0.27, 0.2], [0.1, 0, 0]]),
        mat_rgb=np.ones((2, 3)),
        tex_rgb=np.ones((2, 3)),
        spec=np.zeros(2),
    )
    assert prims.bound_radius().tolist() == [0.3, 0.1]
    assert R3.ELLIPSOID not in R3._BOUNDED_KINDS, "membership is a measurement (r6_kind_cost), not a default"
    assert R3._kind_tensor([R3.ELLIPSOID, R3.SPHERE], "cpu").tolist() == [4, 2]
    # the cast walks it in the nearest-hit loop beside the other kinds, on the definition path too
    B, N = 2, 64
    kinds = [R3.SPHERE, R3.ELLIPSOID]
    sizes = torch.tensor([[[0.1, 0, 0], [0.3, 0.27, 0.2]]] * B)
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]] * B)
    R = torch.eye(3).view(1, 1, 3, 3).expand(B, 2, 3, 3).contiguous()
    o = torch.tensor([2.0, 0.0, 0.0]).view(1, 1, 3).expand(B, N, 3).contiguous()
    d = torch.tensor([-1.0, 0.0, 0.0]).view(1, 1, 3).expand(B, N, 3).contiguous()
    t, n, i = R3.cast_nearest_torch(kinds, sizes, pos, R, o, d)
    assert torch.allclose(t, torch.full((B, N), 1.7)) and bool((i == 1).all())
    t2, n2, i2 = R3._cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    assert torch.equal(t, t2) and torch.equal(i, i2) and torch.equal(n, n2)
