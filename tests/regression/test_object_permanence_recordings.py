"""Offline recorded poses, including hidden motion, match independent Blender."""

import hashlib

import numpy as np
import pytest

from stable_envs.envs.object_permanence import RigidTrajectory
from tests.helpers.reference.object_permanence_recordings import RECORDINGS, load_geometry


@pytest.mark.parametrize("task", ["g34", "g62"])
def test_all_sampled_source_poses_including_occluded_frames(task):
    manifest, oracle = load_geometry(task)
    assert manifest["source_commit"] == "824baddcdad087810d35638bf8f273b884f74975"
    with RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip:
        np.testing.assert_array_equal(clip.frames, oracle["source_frames"])
        metadata = clip.manifest
        assert metadata["source"]["commit"] == manifest["source_commit"]
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
