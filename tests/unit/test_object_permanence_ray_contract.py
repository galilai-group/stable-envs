"""Versioned ray semantics and independently implemented source reference."""

import ast
from pathlib import Path

import numpy as np
import pytest

from scripts.testing.object_permanence_ray_reference import CONTRACT as REFERENCE_CONTRACT
from scripts.testing.object_permanence_ray_reference import draw, plane
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT, local_planes, validate_camera


def camera():
    near, far = 0.1, 10.0
    return {
        "world": np.eye(4).tolist(),
        "projection": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        "near": near,
        "far": far,
    }


def quad(z=-3.0):
    return np.array([[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]])


def test_plane_normalization_is_triangulation_and_winding_independent():
    points = quad()
    points[:, 2] += points[:, 0] * 0.5 + points[:, 1] * 0.25
    faces = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 1], [3, 1, 0]])
    candidate = local_planes(points, faces)
    expected = np.array([plane(points[f]) for f in faces])
    np.testing.assert_array_equal(candidate, expected)
    np.testing.assert_array_equal(candidate, np.repeat(candidate[:1], 4, axis=0))


def test_source_semantic_ties_ignore_object_enumeration_and_keep_distinct_nearer_surfaces():
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    triangles = quad()[faces]
    for meshes in ({"a": triangles, "b": triangles[:, ::-1]}, {"b": triangles[:, ::-1], "a": triangles}):
        depth, ids = draw(meshes, camera(), 31, 47)
        assert ids[15, 23] == 1 and depth[15, 23] == 3
    # A one-ULP gap is not a tie; b is nearer despite its higher semantic ID.
    far = quad(-np.nextafter(3.0, np.inf))[faces]
    depth, ids = draw({"a": far, "b": triangles}, camera(), 31, 47)
    assert ids[15, 23] == 2 and depth[15, 23] == 3


def test_reference_is_independent_of_runtime_and_candidate_data():
    assert CONTRACT == REFERENCE_CONTRACT
    tree = ast.parse(Path("scripts/testing/object_permanence_ray_reference.py").read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert set(imports) <= {"fractions", "numpy"}
    freezer = Path("scripts/testing/freeze_object_permanence_rays.py").read_text()
    assert "stable_envs" not in freezer and "scripts.export" not in freezer
    assert "ray_cast" not in freezer


def test_camera_and_unsupported_conventions_fail_explicitly():
    metadata = camera()
    validate_camera(metadata)
    for mutate in ("projection", "world", "near"):
        bad = camera()
        if mutate == "projection":
            bad["projection"][3][2] = 0
        elif mutate == "world":
            bad["world"][3][3] = 0
        else:
            bad["near"] = 0
        with pytest.raises(ValueError, match="contract"):
            validate_camera(bad)
