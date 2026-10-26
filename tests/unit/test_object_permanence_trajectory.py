"""Rigid recordings preserve instance identity, sharing and caller ownership."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.export.object_permanence import array_digest, write_recording
from stable_envs.envs.object_permanence import RigidTrajectory


@pytest.fixture
def recording():
    mesh = {
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        "normals": np.array([[0, 0, 1]] * 3, np.float32),
        "triangles": np.array([[0, 1, 2]], np.int32),
    }
    key = array_digest(mesh)
    poses = np.zeros((4, 2, 7), np.float32)
    poses[..., 3] = 1
    poses[:, 0, 0] = np.arange(4)
    poses[:, 1, 1] = -np.arange(4)
    objects = [
        {
            "name": name,
            "id": (i + 1) * 10,
            "asset": key,
            "node": node,
            "pose": [9, 0, 0, 1, 0, 0, 0],
            "rgba": color,
            "shape": "triangle",
        }
        for i, (name, node, color) in enumerate(
            [
                ("moving-a", 0, [1, 0, 0, 1]),
                ("part-of-a", 0, [0, 1, 0, 1]),
                ("moving-b", 1, [0, 0, 1, 1]),
                ("static", -1, [1, 1, 1, 1]),
            ]
        )
    ]
    return {
        "meshes": {key: mesh},
        "poses": poses,
        "frames": np.arange(11, 15, dtype=np.int64),
        "objects": objects,
        "dynamic_nodes": ["a", "b"],
        "fps": 24,
        "camera": {"world": np.eye(4).tolist(), "projection": np.eye(4).tolist(), "width": 16, "height": 9},
    }


def write(tmp_path, recording, clip="example"):
    return write_recording(tmp_path, clip, recording, {"license_text": "Synthetic test data."})


def edit_manifest(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_multi_objects_share_assets_and_nodes_without_sharing_returned_memory(tmp_path, recording):
    path = write(tmp_path, recording)
    write(tmp_path, recording, "second")
    assert len(list((tmp_path / "assets").rglob("*.npz"))) == 1
    with RigidTrajectory(tmp_path, "example") as trajectory:
        assert isinstance(trajectory._poses, np.memmap) and not trajectory._poses.flags.writeable
        indices = np.array([3, 0, 3])
        selected = trajectory.poses(indices)
        assert selected.shape == (3, 2, 7) and selected.dtype == np.float32
        np.testing.assert_array_equal(selected, recording["poses"][indices])
        np.testing.assert_array_equal(trajectory.frames, [11, 12, 13, 14])
        expanded = trajectory.object_poses(indices)
        np.testing.assert_array_equal(expanded[:, 0], expanded[:, 1])
        np.testing.assert_array_equal(expanded[:, 2], selected[:, 1])
        np.testing.assert_array_equal(expanded[:, 3, 0], [9, 9, 9])
        selected.fill(0)
        expanded.fill(0)
        assert trajectory.poses(indices)[0, 0, 3] == 1
        description = trajectory.manifest
        description["objects"][0]["node"] = -1
        assert trajectory.manifest["objects"][0]["node"] == 0
        payload = trajectory.triangles()
        assert payload["vert"].shape == (12, 3)
        assert payload["face"].shape == (4, 3)
        np.testing.assert_array_equal(
            payload["vert_geom"][payload["face"]], np.repeat(np.arange(4)[:, None], 3, axis=1)
        )
        payload["vert"].fill(7)
        assert trajectory.triangles()["vert"][0, 0] == 0
        key = next(iter(recording["meshes"]))
        trajectory.mesh(key)["vertices"].fill(8)
        assert trajectory.mesh(key)["vertices"][0, 0] == 0
    with pytest.raises(RuntimeError):
        trajectory.poses([0])
    with pytest.raises(FileExistsError):
        write(tmp_path, recording)
    assert path.exists()


def test_v2_recording_marks_projection_and_keeps_the_existing_pose_layout(tmp_path, recording):
    from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera

    recording["camera"].update(orthographic_camera())
    path = write(tmp_path, recording)
    metadata = json.loads(path.read_text())
    assert metadata["schema"] == "stable-envs/object-permanence-rigid-v2"
    assert metadata["visibility_contract"] == "stable-envs/opaque-pixel-rays-v2"
    with RigidTrajectory(tmp_path, "example") as clip:
        np.testing.assert_array_equal(clip.poses([0, 3]), recording["poses"][[0, 3]])
    del metadata["camera"]["projection_type"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="explicit projection type"):
        RigidTrajectory(tmp_path, "example")


def test_v4_material_policy_is_explicit_and_preserves_optional_scale_layout(tmp_path, recording):
    recording["material_convention"] = "stable-envs/constant-alpha-surfaces-v1"
    recording["objects"][0]["rgba"][3] = 0.25
    recording["scales"] = np.ones((4, 2, 3), np.float32)
    path = write(tmp_path, recording)
    with RigidTrajectory(tmp_path, "example") as clip:
        assert clip.manifest["schema"] == "stable-envs/object-permanence-rigid-v4"
        assert clip.manifest["visibility_contract"] == "stable-envs/transparent-pixel-rays-v1"
        assert clip.has_scales
        np.testing.assert_array_equal(clip.triangles()["rgba"][:, 3], [0.25, 1, 1, 1])
    edit_manifest(path, lambda m: m.pop("material_convention"))
    with pytest.raises(ValueError, match="v4 requires"):
        RigidTrajectory(tmp_path, "example")


@pytest.mark.parametrize("orthographic", [False, True])
def test_optional_relative_scales_preserve_poses_nodes_and_caller_ownership(tmp_path, recording, orthographic):
    if orthographic:
        from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera

        recording["camera"].update(orthographic_camera())
    recording["scales"] = np.ones((4, 2, 3), np.float32)
    recording["scales"][1:, 0] = [2, 0.5, 1e-4]
    path = write(tmp_path, recording)
    assert json.loads(path.read_text())["schema"] == "stable-envs/object-permanence-rigid-v3"
    with RigidTrajectory(tmp_path, "example") as clip:
        assert clip.has_scales and isinstance(clip._scales, np.memmap)
        assert not clip._scales.flags.writeable
        indices = [3, 0, 3]
        selected = clip.scales(indices)
        assert selected.dtype == np.float32 and selected.shape == (3, 2, 3)
        np.testing.assert_array_equal(selected, recording["scales"][indices])
        expanded = clip.object_scales(indices)
        np.testing.assert_array_equal(expanded[:, 0], expanded[:, 1])
        np.testing.assert_array_equal(expanded[:, 2], selected[:, 1])
        np.testing.assert_array_equal(expanded[:, 3], 1)
        np.testing.assert_array_equal(clip.poses(indices), recording["poses"][indices])
        selected.fill(9)
        expanded.fill(9)
        assert clip.scales([3])[0, 0, 0] == 2
    edit_manifest(path, lambda m: m.pop("scale_convention"))
    with pytest.raises(ValueError, match="scale convention"):
        RigidTrajectory(tmp_path, "example")
    with pytest.raises(RuntimeError):
        clip.scales([0])


@pytest.mark.parametrize("filename", ["poses.npy", "frames.npy", "asset"])
def test_payload_corruption_is_rejected(tmp_path, recording, filename):
    write(tmp_path, recording)
    path = next((tmp_path / "assets").rglob("*.npz")) if filename == "asset" else tmp_path / "clips/example" / filename
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="hash"):
        RigidTrajectory(tmp_path, "example")


def test_valid_file_hash_cannot_hide_invalid_quaternions(tmp_path, recording):
    path = write(tmp_path, recording)
    filename = path.parent / "poses.npy"
    rows = np.load(filename)
    rows[2, 0, 3:] = 0
    np.save(filename, rows)
    edit_manifest(
        path, lambda m: m["arrays"]["poses"].update(sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    )
    with pytest.raises(ValueError, match="quaternions"):
        RigidTrajectory(tmp_path, "example")


def test_invalid_identifier_does_not_traverse_paths(tmp_path):
    for name in ("../example", "a/b", "", 12):
        with pytest.raises(ValueError, match="identifier"):
            RigidTrajectory(tmp_path, name)


def test_recorded_frames_load_without_physics_or_rendering_engines():
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    code = """
import sys
sys.modules['mujoco'] = None
sys.modules['bpy'] = None
sys.modules['nvdiffrast'] = None
from stable_envs.envs.object_permanence import RigidTrajectory
with RigidTrajectory(sys.argv[1], 'g34-nominal') as clip:
    assert len(clip) == 120
    assert clip.poses([0, 44, 119]).shape == (3, 1, 7)
    assert clip.frames[44] == 45
assert all(sys.modules[name] is None for name in ('mujoco', 'bpy', 'nvdiffrast'))
"""
    subprocess.run([sys.executable, "-c", code, str(RECORDINGS)], check=True, timeout=60)
