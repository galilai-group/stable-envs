"""Rigid recordings preserve instance identity, sharing and caller ownership."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from scripts.export.object_permanence import array_digest, organize, write_recording
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


@pytest.mark.parametrize("alpha", [-0.1, 1.1, np.nan])
def test_v4_alpha_values_are_validated(tmp_path, recording, alpha):
    recording["material_convention"] = "stable-envs/constant-alpha-surfaces-v1"
    recording["objects"][0]["rgba"][3] = alpha
    write(tmp_path, recording)
    with pytest.raises(ValueError):
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


@pytest.mark.parametrize("schema", ["v1", "v2", "v3"])
def test_missing_optional_scales_default_to_one(tmp_path, recording, schema):
    if schema == "v2":
        from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera

        recording["camera"].update(orthographic_camera())
    path = write(tmp_path, recording)
    if schema == "v3":

        def upgrade(m):
            m.update(
                schema="stable-envs/object-permanence-rigid-v3",
                scale_convention="stable-envs/asset-relative-positive-diagonal-v1",
                visibility_contract="stable-envs/opaque-pixel-rays-v1",
            )
            m["camera"]["projection_type"] = "perspective"

        edit_manifest(path, upgrade)
    with RigidTrajectory(tmp_path, "example") as clip:
        assert not clip.has_scales
        np.testing.assert_array_equal(clip.scales([3, 0]), np.ones((2, 2, 3), np.float32))
        np.testing.assert_array_equal(clip.object_scales([3, 0]), np.ones((2, 4, 3), np.float32))
        with pytest.raises(ValueError):
            clip.scales([-1])


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_even_hash_valid_scale_arrays_must_be_positive_and_finite(tmp_path, recording, bad):
    recording["scales"] = np.ones((4, 2, 3), np.float32)
    recording["scales"][2, 0, 1] = bad
    write(tmp_path, recording)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        RigidTrajectory(tmp_path, "example")


@pytest.mark.parametrize(
    "bad", [np.ones((4, 2, 3), np.float64), np.ones((4, 1, 3), np.float32), np.full((4, 2, 3), 2, np.float32)]
)
def test_scale_dtype_shape_and_asset_reference_are_versioned(tmp_path, recording, bad):
    recording["scales"] = bad
    path = write(tmp_path, recording)
    with pytest.raises(ValueError, match="scales"):
        RigidTrajectory(tmp_path, "example")
    edit_manifest(path, lambda m: m.update(schema="stable-envs/object-permanence-rigid-v1"))
    with pytest.raises(ValueError, match="require rigid-scene v3"):
        RigidTrajectory(tmp_path, "example")


def test_scale_file_hash_and_identity_bind_animation(tmp_path, recording):
    recording["scales"] = np.ones((4, 2, 3), np.float32)
    path = write(tmp_path, recording)
    with RigidTrajectory(tmp_path, "example") as clip:
        identity = clip.identity
    filename = path.parent / "scales.npy"
    changed = recording["scales"].copy()
    changed[2, 1] = [2, 3, 4]
    np.save(filename, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="hash"):
        RigidTrajectory(tmp_path, "example")
    edit_manifest(
        path, lambda m: m["arrays"]["scales"].update(sha256=hashlib.sha256(filename.read_bytes()).hexdigest())
    )
    with RigidTrajectory(tmp_path, "example") as clip:
        assert clip.identity != identity


@pytest.mark.parametrize("indices", [[-1], [4], [], [[0]], [0.0], [True]])
def test_bad_frame_selection_is_rejected(tmp_path, recording, indices):
    write(tmp_path, recording)
    with RigidTrajectory(tmp_path, "example") as trajectory, pytest.raises(ValueError):
        trajectory.poses(indices)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema="future-format"),
        lambda m: m["conventions"].update(quaternion="xyzw"),
        lambda m: m["objects"][0].update(node=2),
        lambda m: m["objects"][0].update(id=20),
        lambda m: m["objects"][0].update(pose=[0] * 7),
        lambda m: m["objects"][0].update(asset="0" * 64),
        lambda m: m.update(fps=0),
        lambda m: m["arrays"]["poses"].update(dtype="float64"),
    ],
)
def test_corrupt_contract_is_rejected(tmp_path, recording, change):
    path = write(tmp_path, recording)
    edit_manifest(path, change)
    with pytest.raises(ValueError):
        RigidTrajectory(tmp_path, "example")


@pytest.mark.parametrize("filename", ["poses.npy", "frames.npy", "asset"])
def test_payload_corruption_is_rejected(tmp_path, recording, filename):
    write(tmp_path, recording)
    path = next((tmp_path / "assets").rglob("*.npz")) if filename == "asset" else tmp_path / "clips/example" / filename
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="hash"):
        RigidTrajectory(tmp_path, "example")


def test_identity_binds_camera_and_metadata_and_content_hash_binds_shape(tmp_path, recording):
    path = write(tmp_path, recording)
    with RigidTrajectory(tmp_path, "example") as trajectory:
        before = trajectory.identity
    edit_manifest(path, lambda m: m["camera"]["world"][0].__setitem__(3, 1))
    with RigidTrajectory(tmp_path, "example") as trajectory:
        assert before != trajectory.identity
    a = np.arange(6, dtype=np.float32)
    assert array_digest({"a": a}) != array_digest({"a": a.reshape(2, 3)})
    assert array_digest({"a": a}) != array_digest({"b": a})


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


def test_concurrent_clip_writers_publish_one_complete_shared_asset(tmp_path, recording):
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda n: write(tmp_path, recording, f"clip{n}"), range(4)))
    assert len(list((tmp_path / "assets").rglob("*.npz"))) == 1
    assert not list((tmp_path / "assets").rglob(".mesh-*"))
    for index in range(4):
        with RigidTrajectory(tmp_path, f"clip{index}") as trajectory:
            np.testing.assert_array_equal(trajectory.poses([3]), recording["poses"][3:4])
    assert len(list((tmp_path / "licenses").glob("*.txt"))) == 1


def test_organize_migrates_flat_assets_and_licenses_idempotently(tmp_path, recording):
    paths = [write(tmp_path, recording, clip) for clip in ("one", "two")]
    mesh = next((tmp_path / "assets").rglob("*.npz"))
    mesh.rename(tmp_path / "assets" / mesh.name)
    for path in paths:

        def legacy(m):
            m.pop("license_ref")
            for description in m["assets"].values():
                description.pop("file")

        edit_manifest(path, legacy)
        (path.parent / "SOURCE_LICENSE.txt").write_text("Synthetic test data.")
    (tmp_path / "index.json").write_text("{}")
    payloads = {p: p.read_bytes() for p in tmp_path.rglob("*.npy")}
    with RigidTrajectory(tmp_path, "one") as trajectory:
        identity = trajectory.identity
    organize(tmp_path)
    assert not (tmp_path / "index.json").exists()
    assert len(list((tmp_path / "assets/triangle").glob("*.npz"))) == 1
    assert not list((tmp_path / "assets").glob("*.npz"))
    assert not list((tmp_path / "clips").rglob("SOURCE_LICENSE.txt"))
    assert len(list((tmp_path / "licenses").glob("*.txt"))) == 1
    with RigidTrajectory(tmp_path, "one") as trajectory:
        assert identity != trajectory.identity
    first = {p: p.read_bytes() for p in paths}
    organize(tmp_path)
    assert all(p.read_bytes() == data for p, data in first.items())
    assert all(p.read_bytes() == data for p, data in payloads.items())


def test_shared_license_integrity_is_verified(tmp_path, recording):
    write(tmp_path, recording)
    next((tmp_path / "licenses").glob("*.txt")).write_text("changed")
    with pytest.raises(ValueError, match="license hash"):
        RigidTrajectory(tmp_path, "example")
