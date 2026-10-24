"""Exact diagnostic geometry must distinguish ties from close surfaces."""

from fractions import Fraction

import numpy as np
import pytest

from scripts.testing.check_object_permanence_visibility import exact_nearest, exact_triangle


def triangle(z):
    return np.array([[[-1.0, -1.0, z], [1.0, -1.0, z], [0.0, 1.0, z]]])


def test_exact_nearest_preserves_all_coplanar_objects_and_independent_triangulations():
    a = triangle(-3)
    # Same plane/coverage, different corner order and object insertion order.
    for meshes in ({"a": a, "b": a[:, ::-1]}, {"b": a[:, ::-1], "a": a}):
        hit = exact_nearest([0, 0, 0], [0, 0, -1], meshes)
        assert hit["objects"] == ["a", "b"] and hit["distance"] == 3


def test_one_ulp_separation_is_not_collapsed_to_a_tie():
    far = -np.nextafter(3.0, np.inf)
    hit = exact_nearest([0, 0, 0], [0, 0, -1], {"far": triangle(far), "near": triangle(-3)})
    assert hit["objects"] == ["near"]


def test_exact_closed_edges_parallel_and_behind_camera():
    face = triangle(-3)[0]
    result = exact_triangle([-1, -1, 0], [0, 0, -1], face)
    assert result == (Fraction(3), Fraction(0), Fraction(0))
    assert exact_triangle([0, 0, 0], [1, 0, 0], face) is None
    assert exact_triangle([0, 0, 0], [0, 0, 1], face) is None


def test_exact_aabb_culling_keeps_a_thin_hit_and_rejects_a_near_miss():
    x = 0.0625
    face = np.array([[[x - 1e-10, -0.1, -3], [x + 1e-10, -0.1, -3], [x, 0.1, -3]]])
    assert exact_nearest([x, 0, 0], [0, 0, -1], {"thin": face})["objects"] == ["thin"]
    assert exact_nearest([x + 2e-10, 0, 0], [0, 0, -1], {"thin": face})["objects"] == []


@pytest.mark.parametrize(
    "origin,direction", [([0, 0], [0, 0, -1]), ([np.nan, 0, 0], [0, 0, -1]), ([0, 0, 0], [0, 0, 0])]
)
def test_invalid_rays_fail_explicitly(origin, direction):
    with pytest.raises(ValueError, match="ray"):
        exact_nearest(origin, direction, {"a": triangle(-3)})


def test_exact_distance_limit_applies_to_physical_distance_for_nonunit_rays():
    assert not exact_nearest([0, 0, 0], [0, 0, -2], {"a": triangle(-3)}, max_distance=2)["objects"]
    assert exact_nearest([0, 0, 0], [0, 0, -2], {"a": triangle(-3)}, max_distance=3)["objects"] == ["a"]


def test_canonical_policy_uses_declared_ids_and_never_overrides_a_distinct_nearer_object():
    from scripts.testing.check_object_permanence_visibility import canonical_object

    ids = {"a": 20, "b": 10}
    nearest = exact_nearest([0, 0, 0], [0, 0, -1], {"a": triangle(-3), "b": triangle(-3)})
    assert canonical_object(nearest, ids) == "b"
    closer = exact_nearest([0, 0, 0], [0, 0, -1], {"a": triangle(-3), "b": triangle(-np.nextafter(3.0, np.inf))})
    assert canonical_object(closer, ids) == "a"
    with pytest.raises(ValueError, match="distinct"):
        canonical_object(nearest, {"a": 1, "b": 1})
