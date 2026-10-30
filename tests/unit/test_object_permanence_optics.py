"""Analytic optical laws and explicit volume/material rejection."""

import numpy as np
import pytest

from stable_envs.envs.object_permanence.rendering.optics import closed_volume, snell


def test_snell_normal_incidence_and_fresnel_energy():
    reflected, refracted, f = snell([0, 0, -1], [0, 0, 1], 1.0, 1.5)
    np.testing.assert_array_equal(reflected, [0, 0, 1])
    np.testing.assert_allclose(refracted, [0, 0, -1], atol=1e-15)
    assert f == pytest.approx(0.04)


def test_snell_oblique_refraction_obeys_sine_law():
    angle = np.deg2rad(60)
    _, d, f = snell([np.sin(angle), 0, -np.cos(angle)], [0, 0, 1], 1.0, 1.5)
    assert np.linalg.norm(d) == pytest.approx(1)
    assert d[0] * 1.5 == pytest.approx(np.sin(angle))
    assert 0 < f < 1


def test_snell_total_internal_reflection_and_equal_medium_grazing():
    angle = np.deg2rad(60)
    reflected, _, f = snell([np.sin(angle), 0, -np.cos(angle)], [0, 0, 1], 1.5, 1.0)
    assert f == 1
    assert np.linalg.norm(reflected) == pytest.approx(1)
    _, _, f = snell([1, 0, 0], [0, 0, 1], 1.45, 1.45)
    assert f == 0


def tetrahedron():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    return p[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]


def test_volume_validation_uses_positions_despite_split_corner_vertices():
    closed_volume(tetrahedron())
    for mesh in (tetrahedron()[:-1], tetrahedron()[[0, 0, 2, 3]], np.zeros((1, 3, 3))):
        with pytest.raises(ValueError):
            closed_volume(mesh)


def test_v5_exact_transform_file_preserves_legacy_pose_shape_and_memory_ownership(tmp_path):
    import hashlib
    import json

    from scripts.export.object_permanence import array_digest, write_recording
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import MATERIAL, POLICY

    triangles = tetrahedron().astype(np.float32)
    mesh = {
        "vertices": triangles.reshape(-1, 3),
        "triangles": np.arange(12, dtype=np.int32).reshape(-1, 3),
        "normals": np.ones((12, 3), np.float32),
    }
    key = array_digest(mesh)
    poses = np.zeros((3, 1, 7), np.float32)
    poses[..., 3] = 1
    poses[:, 0, 0] = [0, 0.1, 0.2]
    transforms = np.broadcast_to(np.eye(4), (3, 1, 4, 4)).copy()
    transforms[:, :, :3, 3] = poses[:, :, :3]
    recording = {
        "poses": poses,
        "transforms": transforms,
        "frames": np.arange(1, 4, dtype=np.int64),
        "dynamic_nodes": ["glass"],
        "meshes": {key: mesh},
        "fps": 24,
        "material_convention": MATERIAL,
        "optical_policy": dict(POLICY),
        "camera": {
            "world": np.eye(4).tolist(),
            "projection": np.eye(4).tolist(),
            "width": 16,
            "height": 9,
            "near": 0.1,
            "far": 100,
        },
        "objects": [
            {
                "name": "glass",
                "id": 1,
                "asset": key,
                "node": 0,
                "pose": poses[0, 0].tolist(),
                "rgba": [0.3, 0.7, 1, 0.28],
                "optics": {"transmission": 0.25, "ior": 1.45, "roughness": 0.18},
                "transform": transforms[0, 0].tolist(),
            }
        ],
    }
    path = write_recording(tmp_path, "example", recording, {"license_text": "fixture"})
    with RigidTrajectory(tmp_path, "example") as clip:
        assert clip.poses([1]).dtype == np.float32 and clip.poses([1]).shape == (1, 1, 7)
        actual = clip.transforms([2, 0, 2])
        np.testing.assert_array_equal(actual, transforms[[2, 0, 2]])
        actual.fill(9)
        assert clip.transforms([0])[0, 0, 3, 3] == 1
        assert clip.manifest["schema"] == "stable-envs/object-permanence-rigid-v5"
    original = json.loads(path.read_text())
    m = json.loads(path.read_text())
    m["optical_policy"]["samples"] = 16
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="conventions"):
        RigidTrajectory(tmp_path, "example")
    path.write_text(json.dumps(original))
    broken = transforms.copy()
    broken[1, 0, 0, 3] = 100
    np.save(path.parent / "transforms.npy", broken, allow_pickle=False)
    original["arrays"]["transforms"]["sha256"] = hashlib.sha256(
        (path.parent / "transforms.npy").read_bytes()
    ).hexdigest()
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="exact source transforms"):
        RigidTrajectory(tmp_path, "example")
