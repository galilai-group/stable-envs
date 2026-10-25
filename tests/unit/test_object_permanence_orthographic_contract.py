"""Orthographic rays preserve axial depth, projection and opaque visibility."""

import numpy as np
import pytest

from scripts.testing.object_permanence_ray_reference import CONTRACT_V2 as REFERENCE_V2
from scripts.testing.object_permanence_ray_reference import draw
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT_V2, validate_camera
from tests.unit.test_object_permanence_ray_contract import camera, quad


def orthographic_camera(near=0.1, far=10.0, shift=(0.0, 0.0)):
    return {
        "projection_type": "orthographic",
        "world": np.eye(4).tolist(),
        "projection": [
            [1, 0, 0, shift[0]],
            [0, 1, 0, shift[1]],
            [0, 0, -2 / (far - near), -(far + near) / (far - near)],
            [0, 0, 0, 1],
        ],
        "near": near,
        "far": far,
    }


def test_versioned_camera_validation_preserves_v1_and_rejects_ambiguous_projection():
    assert CONTRACT_V2 == REFERENCE_V2
    validate_camera(camera())
    validate_camera(camera(), contract=CONTRACT_V2)
    validate_camera(orthographic_camera(), contract=CONTRACT_V2)
    with pytest.raises(ValueError, match="contract"):
        validate_camera(orthographic_camera())
    for kind in ("missing", "perspective", "unknown"):
        bad = orthographic_camera()
        if kind == "missing":
            del bad["projection_type"]
        else:
            bad["projection_type"] = kind
        with pytest.raises(ValueError, match="contract"):
            validate_camera(bad, contract=CONTRACT_V2)


def test_orthographic_size_is_depth_independent_and_depth_is_axial_not_ray_length():
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    cam = orthographic_camera(shift=(0.21, -0.13))
    near, ids = draw({"object": quad(-3)[faces]}, cam, 31, 47, visibility_contract=CONTRACT_V2)
    far, farther_ids = draw({"object": quad(-6)[faces]}, cam, 31, 47, visibility_contract=CONTRACT_V2)
    np.testing.assert_array_equal(ids, farther_ids)
    assert (near[ids > 0] == 3).all() and (far[ids > 0] == 6).all()
    assert (ids > 0).sum() > 500
    with pytest.raises(ValueError, match="requires v2"):
        draw({"object": quad()[faces]}, cam, 31, 47)


def test_orthographic_clipping_uses_hit_depth_without_moving_the_origin_to_near():
    cam = orthographic_camera(near=0.5, far=1.5)
    points = quad()
    points[:, 2] = -(1 + points[:, 0])
    depth, ids = draw({"slope": points[[[0, 1, 2], [0, 2, 3]]]}, cam, 16, 32, visibility_contract=CONTRACT_V2)
    x = 2 * (np.arange(32) + 0.5) / 32 - 1
    expected = np.broadcast_to((x >= -0.5) & (x <= 0.5), ids.shape)
    np.testing.assert_array_equal(ids > 0, expected)
    np.testing.assert_array_equal(depth[expected], np.broadcast_to(1 + x, depth.shape)[expected].astype(np.float32))


def test_orthographic_camera_pose_and_exact_semantic_ties():
    cam = orthographic_camera(shift=(-0.1, 0.2))
    c, s = np.cos(0.37), np.sin(0.37)
    world = np.array([[c, 0, s, 2], [0, 1, 0, -1], [-s, 0, c, 4], [0, 0, 0, 1]])
    cam["world"] = world.tolist()
    points = quad() @ world[:3, :3].T + world[:3, 3]
    triangles = points[[[0, 1, 2], [0, 2, 3]]]
    depth, ids = draw({"b": triangles[:, ::-1], "a": triangles}, cam, 31, 47, visibility_contract=CONTRACT_V2)
    assert (ids[ids > 0] == 1).all()
    np.testing.assert_allclose(depth[ids > 0], 3, atol=1e-6, rtol=0)


@pytest.mark.parametrize("field,value", [("near", np.nan), ("far", np.inf), ("near", 0), ("far", 0.05)])
def test_orthographic_depth_interval_must_be_positive_finite_and_ordered(field, value):
    metadata = orthographic_camera()
    metadata[field] = value
    with pytest.raises(ValueError, match="near < far"):
        validate_camera(metadata, contract=CONTRACT_V2)
