"""Analytic optical laws and explicit volume/material rejection."""

import numpy as np
import pytest

from stable_envs.envs.object_permanence.rendering.optics import closed_volume, hierarchy, snell, validate_material


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


@pytest.mark.parametrize(
    "change",
    [
        {"transmission": -1},
        {"transmission": 1.01},
        {"ior": 0.9},
        {"ior": float("nan")},
        {"roughness": 2},
        {"volume": 1},
    ],
)
def test_material_policy_rejects_unknown_or_invalid_parameters(change):
    m = {"transmission": 0.25, "ior": 1.45, "roughness": 0.18}
    m.update(change)
    with pytest.raises(ValueError):
        validate_material(m)


def tetrahedron():
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    return p[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]


def test_volume_validation_uses_positions_despite_split_corner_vertices():
    closed_volume(tetrahedron())
    for mesh in (tetrahedron()[:-1], tetrahedron()[[0, 0, 2, 3]], np.zeros((1, 3, 3))):
        with pytest.raises(ValueError):
            closed_volume(mesh)


def test_local_hierarchy_escape_pointers_cover_every_original_face_once():
    rng = np.random.default_rng(4)
    triangles = rng.normal(size=(37, 3, 3))
    b, n, order = hierarchy(triangles)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(triangles)))
    assert n[0, 0] == len(n)
    assert np.all(n[:, 0] > np.arange(len(n)))
    for index, (_, start, count) in enumerate(n):
        if count:
            faces = triangles[order[start : start + count]]
            assert np.all(faces >= b[index, :3]) and np.all(faces <= b[index, 3:])


def test_independent_reference_has_no_candidate_imports():
    import ast
    from pathlib import Path

    for name in ("object_permanence_optical_reference.py", "freeze_object_permanence_optical.py"):
        tree = ast.parse((Path(__file__).parents[2] / "scripts/testing" / name).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                assert not (n.module or "").startswith(("stable_envs", "scripts.export"))
            if isinstance(n, ast.Import):
                assert all(not a.name.startswith(("stable_envs", "torch", "scripts.export")) for a in n.names)


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


def test_certified_optical_members_bind_policy_generators_and_source_provenance():
    import json
    import re

    from scripts.export.object_permanence_optical import MEMBERS
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.optics import CONTRACT, MATERIAL, POLICY
    from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS

    for task, member in MEMBERS:
        clip = task.lower() + ("-nominal" if member == 0 else "-member1")
        oracle = (
            FIXTURES
            / f"semantics/object_permanence/{task.lower()}/sampled_optical_rays_v1"
            / ("nominal" if member == 0 else "member1")
        )
        m = json.loads((oracle / "manifest.json").read_text())
        assert m["optical_policy"] == POLICY
        assert m["visibility_contract"] == CONTRACT and m["material_convention"] == MATERIAL
        assert m["generator"] == "scripts/testing/freeze_object_permanence_optical.py"
        for key in ("generator_sha256", "reference_sha256"):
            assert re.fullmatch("[0-9a-f]{64}", m[key])
        assert m["frames"] == 120
        with RigidTrajectory(RECORDINGS, clip) as trajectory:
            source = trajectory.manifest["source"]
            assert m["source_script_sha256"] == source["script_sha256"]
            assert m["source_task_manifest_sha256"] == source["task_manifest_sha256"]
            assert m["source_preparation_recipe"] == source["preparation_recipe"]
            assert m["source_use_raytracing"] == source["use_raytracing"]
            for key in ("exporter_sha256", "geometry_exporter_sha256", "optical_policy_sha256"):
                assert re.fullmatch("[0-9a-f]{64}", source[key])
            assert trajectory.transforms([0, 44, 119]).dtype == np.float64


@pytest.mark.parametrize("task,member", [("G01", 0), ("G01", 1), ("G03", 1)])
def test_optical_oracle_encoding_is_bounded_and_keeps_geometric_support(task, member):
    from scripts.testing.check_object_permanence_optical import load
    from tests.helpers.reference.object_permanence_recordings import FIXTURES

    directory = (
        FIXTURES
        / f"semantics/object_permanence/{task.lower()}/sampled_optical_rays_v1"
        / ("nominal" if member == 0 else "member1")
    )
    metadata, source = load(directory)
    assert all(p.stat().st_size <= 1000 * 1024 for p in directory.rglob("*") if p.is_file())
    assert source["ids"].dtype == np.int16 and source["path_depth"].dtype == np.float32
    np.testing.assert_array_equal(source["path_depth"] > 0, source["masks"])
    for column, name in enumerate(metadata["object_order"]):
        label = metadata["object_ids"][name]
        assert np.all(source["masks"][:, column][source["ids"] == label])
