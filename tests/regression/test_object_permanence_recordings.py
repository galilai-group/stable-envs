"""Offline recorded poses, including hidden motion, match independent Blender."""

import hashlib

import numpy as np
import pytest

from stable_envs.envs.object_permanence import RigidTrajectory
from tests.helpers.reference.object_permanence_recordings import RAY_TASKS, RECORDED_TASKS, RECORDINGS, load_geometry


@pytest.mark.parametrize("task", RECORDED_TASKS)
def test_all_sampled_source_poses_including_occluded_frames(task):
    manifest, oracle = load_geometry(task)
    assert manifest["source_commit"] == "824baddcdad087810d35638bf8f273b884f74975"
    with RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip:
        np.testing.assert_array_equal(clip.frames, oracle["source_frames"])
        metadata = clip.manifest
        assert metadata["source"]["commit"] == manifest["source_commit"]
        assert metadata["source"]["script"] == manifest["source_script"]
        assert metadata["source"]["script_sha256"] == manifest["source_script_sha256"]
        assert metadata["fps"] == manifest["fps"]
        ids = {obj["name"]: obj["id"] for obj in metadata["objects"]}
        assert ids == manifest["object_ids"]
        if task == "g34":
            poses = clip.poses(np.arange(len(clip)))
            expected = np.concatenate([oracle["positions"], oracle["quaternions"]], -1)[:, None]
        else:
            poses = clip.object_poses(np.arange(len(clip)))
            assert [obj["name"] for obj in metadata["objects"]] == manifest["object_order"]
            expected = oracle["poses"]
        np.testing.assert_allclose(poses[..., :3], expected[..., :3], atol=3e-7, rtol=0)
        signs = np.where((poses[..., 3:] * expected[..., 3:]).sum(-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(poses[..., 3:], expected[..., 3:] * signs, atol=1e-6, rtol=0)


def test_g62_reference_depth_encoding_is_lossless_and_covers_both_visibility_boundaries():
    manifest, oracle = load_geometry("g62")
    assert oracle["depth"].shape == oracle["object_ids"].shape == (120, 180, 320)
    assert oracle["depth"].dtype == np.float32
    assert np.isfinite(oracle["depth"]).all() and (oracle["depth"] >= 0).all()
    np.testing.assert_array_equal(oracle["depth"][0].view(np.uint32), oracle["depth_base"].view(np.uint32))
    for name in manifest["targets"]:
        counts = (oracle["object_ids"] == manifest["object_ids"][name]).sum(axis=(1, 2))
        assert counts[0] > 0 and counts[-1] > 0
        hidden = np.flatnonzero(counts == 0)
        assert len(hidden) == 28
        assert counts[hidden[0] - 1] > 0 and counts[hidden[-1] + 1] > 0


def test_g34_expansion_preserves_original_21_samples_bit_for_bit():
    from tests.helpers.reference.occlusion_ball import LEGACY_GEOMETRY_FRAMES

    _, oracle = load_geometry("g34")
    assert oracle["depth"].shape == (120, 180, 320)
    selected = np.searchsorted(oracle["geometry_frames"], LEGACY_GEOMETRY_FRAMES)
    digest = hashlib.sha256(oracle["depth"][selected].tobytes() + oracle["object_ids"][selected].tobytes()).hexdigest()
    assert digest == "ca579ee8d589a4d817804d34a56d8e9056df5b1cace8452cef246f8c0cdb3456"


@pytest.mark.parametrize("task", ["g37", "g38", "g64"])
def test_first_batch_lossless_depth_parts_and_actual_source_visibility(task):
    from scripts.testing.freeze_object_permanence import REFERENCE_CASES

    manifest, oracle = load_geometry(task)
    assert oracle["depth"].shape == oracle["object_ids"].shape == (120, 180, 320)
    assert oracle["depth"].dtype == np.float32
    assert np.isfinite(oracle["depth"]).all() and (oracle["depth"] >= 0).all()
    assert len(manifest["depth_parts"]) == 4
    np.testing.assert_array_equal(oracle["depth"][0].view(np.uint32), oracle["depth_base"].view(np.uint32))
    assert manifest["targets"] == list(REFERENCE_CASES[task.upper()][2])
    for name in manifest["targets"]:
        counts = (oracle["object_ids"] == manifest["object_ids"][name]).sum(axis=(1, 2))
        np.testing.assert_array_equal(counts, manifest["target_visible_pixels"][name])
        assert counts[0] > 0 and counts[-1] > 0
        assert counts.min() < counts.max()
        if task == "g64":
            assert counts.min() > 0  # Board window preserves partial visibility.
        else:
            assert (counts == 0).any()  # Actual complete occlusion occurs.


def test_every_audited_nominal_entry_has_a_declared_independent_source_gate():
    from scripts.export.object_permanence import AUDITED_TASKS, CONTRACT_ONLY_TASKS, DEFERRED_TASKS

    assert {task.lower() for task in AUDITED_TASKS if task not in set(DEFERRED_TASKS) | CONTRACT_ONLY_TASKS} == set(
        RECORDED_TASKS
    )


@pytest.mark.parametrize("task", ["g27", "g55"])
def test_second_batch_reference_visibility_and_lossless_depth_chunks(task):
    from scripts.testing.freeze_object_permanence import REFERENCE_CASES

    manifest, oracle = load_geometry(task)
    assert oracle["depth"].shape == oracle["object_ids"].shape == (120, 180, 320)
    assert oracle["depth"].dtype == np.float32
    assert np.isfinite(oracle["depth"]).all() and (oracle["depth"] >= 0).all()
    assert all(0 < p["stop"] - p["start"] <= 30 for p in manifest["depth_parts"])
    np.testing.assert_array_equal(oracle["depth"][0].view(np.uint32), oracle["depth_base"].view(np.uint32))
    assert manifest["targets"] == list(REFERENCE_CASES[task.upper()][2])
    for name in manifest["targets"]:
        counts = (oracle["object_ids"] == manifest["object_ids"][name]).sum(axis=(1, 2))
        np.testing.assert_array_equal(counts, manifest["target_visible_pixels"][name])
        assert counts[0] > 0 and counts[-1] > 0
        assert counts.min() < counts.max()


def test_support_removal_precedes_the_recorded_ball_drop():
    with RigidTrajectory(RECORDINGS, "g27-nominal") as clip:
        objects = clip.manifest["objects"]
        poses = clip.object_poses(np.arange(len(clip)))
        by_name = {obj["name"]: i for i, obj in enumerate(objects)}
        support = poses[:, by_name["support_plate"], 0]
        ball = poses[:, by_name["target_orange_ball"], 2]
        moving_support = np.flatnonzero(support != support[0])
        falling_ball = np.flatnonzero(ball < ball[0])
        assert len(moving_support) and len(falling_ball)
        assert moving_support[0] < falling_ball[0]
        assert support[-1] < support[0] and ball[-1] < ball[0]


@pytest.mark.parametrize("task", RAY_TASKS)
def test_versioned_source_poses_hidden_motion_and_reference_identity(task):
    from tests.helpers.reference.object_permanence_recordings import load_ray_geometry

    manifest, reference = load_ray_geometry(task)
    assert reference["depth"].shape == reference["object_ids"].shape == (120, 180, 320)
    assert reference["depth"].dtype == np.float32 and np.isfinite(reference["depth"]).all()
    np.testing.assert_array_equal(reference["depth"][0], reference["depth_base"])
    with RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip:
        np.testing.assert_array_equal(clip.frames, reference["source_frames"])
        assert manifest["source_script_sha256"] == clip.manifest["source"]["script_sha256"]
        assert manifest.get("source_execution_context") == clip.manifest["source"].get("execution_context")
        assert manifest["object_ids"] == {o["name"]: o["id"] for o in clip.manifest["objects"]}
        actual = clip.object_poses(np.arange(120))
        expected = reference["poses"]
        np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=3e-7, rtol=0)
        signs = np.where((actual[..., 3:] * expected[..., 3:]).sum(-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(actual[..., 3:], expected[..., 3:] * signs, atol=1e-6, rtol=0)
    for name in manifest["targets"]:
        visible = (reference["object_ids"] == manifest["object_ids"][name]).sum((1, 2))
        np.testing.assert_array_equal(visible, manifest["target_visible_pixels"][name])
        assert visible.max() > 0


def test_versioned_contract_registry_never_erases_legacy_bvh_deferrals():
    from scripts.export.object_permanence import AUDITED_TASKS, DEFERRED_TASKS, RAY_CONTRACT_TASKS
    from tests.helpers.reference.object_permanence_recordings import RAY_TASKS

    assert set(DEFERRED_TASKS) == {"G63", "G66", "G80", "G81"}
    assert set(t.upper() for t in RAY_TASKS) == RAY_CONTRACT_TASKS
    assert RAY_CONTRACT_TASKS <= set(AUDITED_TASKS)
