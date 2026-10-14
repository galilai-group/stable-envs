"""``render/primitives.py``: the mjtGeom codes, 3-vector helpers and bounding-radius law shared by
``raycast3d`` and ``cpu3d``.

Both backends must use the shared objects, and the bounding radius's two spellings (numpy for a baked
scene, torch for scaled sizes) must agree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

P = pytest.importorskip("stable_envs.render.primitives")
R3 = pytest.importorskip("stable_envs.render.raycast3d")
C3 = pytest.importorskip("stable_envs.render.cpu3d")


@pytest.mark.unit
def test_the_kind_codes_are_mjtGeoms_own_numbers():
    # a scene is built straight from an MjModel's geom_type, so these are not free to be re-numbered
    assert (P.PLANE, P.SPHERE, P.CAPSULE, P.ELLIPSOID, P.CYLINDER, P.BOX) == (0, 2, 3, 4, 5, 6)
    assert P.PRIMITIVE_KINDS == (0, 2, 3, 4, 5, 6)


@pytest.mark.unit
def test_both_backends_read_the_kind_codes_from_one_place():
    for name in ("PLANE", "SPHERE", "CAPSULE", "ELLIPSOID", "CYLINDER", "BOX"):
        assert getattr(R3, name) == getattr(P, name), name
    # cpu3d's movers are the intersectable kinds minus the plane, derived rather than re-listed
    assert C3.MOVER_KINDS == tuple(k for k in P.PRIMITIVE_KINDS if k != P.PLANE)
    assert P.ELLIPSOID in C3.MOVER_KINDS, "the quadruped torso is a mover"


@pytest.mark.unit
def test_the_unrolled_vector_helpers_are_shared_objects_not_copies():
    # identity, not equality: a second copy would still pass a value comparison
    assert R3.dot3 is P.dot3
    assert R3.mv3 is P.mv3
    assert R3.mtv3 is P.mtv3
    assert R3.bound_radius_tensor is P.bound_radius_tensor
    assert R3.Primitives is P.Primitives
    assert R3.INF is P.INF


@pytest.mark.unit
def test_mv_and_mtv_are_the_matrix_product_and_its_transpose():
    g = torch.Generator().manual_seed(0)
    M = torch.rand(4, 3, 3, generator=g, dtype=torch.float64)
    v = torch.rand(4, 3, generator=g, dtype=torch.float64)
    assert torch.allclose(P.mv3(M, v), torch.einsum("bij,bj->bi", M, v))
    assert torch.allclose(P.mtv3(M, v), torch.einsum("bji,bj->bi", M, v))
    assert torch.allclose(P.dot3(M[:, 0], v), (M[:, 0] * v).sum(-1))


@pytest.mark.unit
def test_the_unrolled_product_does_not_depend_on_the_batch_size():
    """An elementwise product cannot see its neighbours, so the batch size cannot move it.

    ``einsum`` lowers to ``bmm`` and cuBLAS picks a kernel by batch count, which can move the product
    by an ulp, and an ulp in the ray quadratic is a hit/miss decision. ``tests/gpu`` checks the device.
    """
    g = torch.Generator().manual_seed(1)
    M = torch.rand(1, 3, 3, generator=g)
    v = torch.rand(1, 3, generator=g)
    one = P.mv3(M, v)
    for n in (2, 8, 257):
        many = P.mv3(M.expand(n, 3, 3), v.expand(n, 3))
        assert torch.equal(many, one.expand(n, 3)), f"batch {n} moved the product"


@pytest.mark.unit
@pytest.mark.parametrize("kind", [P.SPHERE, P.CAPSULE, P.CYLINDER, P.BOX, P.ELLIPSOID])
def test_the_two_bound_radius_spellings_agree(kind):
    """numpy (baked scene) and torch (scaled sizes) are one law written twice; they must not drift."""
    rng = np.random.default_rng(0)
    size = rng.uniform(0.01, 2.0, size=(32, 3))
    want = P.bound_radius(np.full(32, kind), size)
    got = P.bound_radius_tensor(kind, torch.as_tensor(size, dtype=torch.float64))
    assert got is not None
    assert np.allclose(want, got.numpy(), rtol=0, atol=1e-12)


@pytest.mark.unit
def test_an_unbounded_kind_says_so_in_each_spellings_own_way():
    """``bound_radius_tensor`` returns ``None`` for a plane; the numpy form returns ``INF``.

    ``INF`` is ``1e30``, a finite float, so the ``np.isfinite`` guard in ``Primitives.radii`` does not
    fire on a plane. ``screen_span`` rejects it instead, through ``torch.isfinite`` on the projected
    corners, which overflow.
    """
    assert P.bound_radius_tensor(P.PLANE, torch.ones(4, 3)) is None
    r = P.bound_radius(np.array([P.PLANE]), np.ones((1, 3)))
    assert r[0] == P.INF
    assert np.isfinite(r).all(), "1e30 is finite; see Primitives.radii and screen_span"


@pytest.mark.unit
def test_the_radius_actually_contains_the_primitive():
    """A bound is a claim that nothing outside it can be hit, so sample the surface and check it."""
    rng = np.random.default_rng(3)
    for kind, size in (
        (P.SPHERE, [0.3, 0, 0]),
        (P.CAPSULE, [0.1, 0.4, 0]),
        (P.CYLINDER, [0.2, 0.5, 0]),
        (P.BOX, [0.2, 0.3, 0.4]),
        (P.ELLIPSOID, [0.1, 0.5, 0.25]),
    ):
        r = float(P.bound_radius(np.array([kind]), np.array([size], float))[0])
        s = np.asarray(size, float)
        if kind == P.SPHERE:
            pts = s[0] * _unit(rng)
        elif kind == P.ELLIPSOID:
            pts = _unit(rng) * s
        elif kind in (P.CAPSULE, P.CYLINDER):
            u = _unit(rng)
            pts = np.stack(
                [
                    s[0] * u[:, 0],
                    s[0] * u[:, 1],
                    np.clip(2 * s[1] * (rng.random(len(u)) - 0.5) * 2, -s[1], s[1])
                    + (s[0] * u[:, 2] if kind == P.CAPSULE else 0.0),
                ],
                -1,
            )
        else:
            pts = (2 * rng.random((512, 3)) - 1) * s
        assert np.linalg.norm(pts, axis=-1).max() <= r + 1e-9, f"kind {kind} escapes its bound"


def _unit(rng, n: int = 512) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.mark.unit
def test_primitives_bound_radius_is_the_free_function():
    kinds = np.array([P.SPHERE, P.CAPSULE, P.BOX, P.ELLIPSOID, P.PLANE])
    size = np.array([[0.2, 0, 0], [0.1, 0.3, 0], [0.2, 0.3, 0.4], [0.1, 0.5, 0.2], [1.0, 1.0, 0.1]])
    prims = P.Primitives(kind=kinds, size=size, mat_rgb=np.zeros((5, 3)), tex_rgb=np.ones((5, 3)), spec=np.zeros(5))
    assert np.array_equal(prims.bound_radius(), P.bound_radius(kinds, size))
    # the device form is the same numbers scaled, INF included, memoised per (device, dtype, scale)
    radii = prims.radii("cpu", torch.float32, 2.0)
    assert torch.allclose(radii, torch.as_tensor(P.bound_radius(kinds, size) * 2.0, dtype=torch.float32))
    assert prims.radii("cpu", torch.float32, 2.0) is radii
    assert prims.radii("cpu", torch.float32, 1.0) is not radii


@pytest.mark.unit
def test_the_memo_evicts_lru_at_its_cap():
    m = R3._Memo(4, name="test")
    for i in range(4):
        m.put(i, i * i)
    assert len(m) == 4 and m.get(3) == 9
    m.put(99, 1)  # at the cap: the least recently used key goes, the rest stay
    assert len(m) == 4 and m.get(99) == 1 and m.get(3) == 9 and m.get(0) is None


@pytest.mark.unit
def test_every_raycast3d_cache_is_bounded():
    """An unbounded memo on a per-frame path is a leak; each one carries its own cap."""
    for cache in (R3._PROJ_CACHE, R3._CORNER_CACHE, R3._KIND_CACHE, R3._RAY_CACHE, R3._TILE_CACHE):
        assert isinstance(cache, R3._Memo)
        assert 0 < cache.capacity <= 64


@pytest.mark.unit
def test_an_unknown_geom_is_refused_by_both_dispatchers_with_the_same_reason():
    """mjGEOM_MESH (7) has no analytic intersector, and neither backend may draw it as something else."""
    o = torch.zeros(1, 4, 3)
    d = torch.zeros(1, 4, 3)
    d[..., 0] = 1.0
    with pytest.raises(ValueError, match="no intersector for mjtGeom"):
        R3.intersect(7, torch.ones(1, 1, 3), torch.zeros(1, 1, 3), torch.eye(3).view(1, 1, 3, 3), o, d)
    with pytest.raises(ValueError, match="no intersector for mjtGeom"):
        R3._kind_tensor([7], "cpu")
