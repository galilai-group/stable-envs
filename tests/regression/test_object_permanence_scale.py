"""The scale recording and independent source oracle describe the same samples."""

import numpy as np
import pytest

from stable_envs.envs.object_permanence import RigidTrajectory
from tests.helpers.reference.object_permanence_recordings import RECORDINGS, load_ray_geometry


def test_g146_scale_only_motion_preserves_source_origin_and_all_frame_ratios():
    oracle, reference = load_ray_geometry("g146")
    with RigidTrajectory(RECORDINGS, "g146-nominal") as clip:
        metadata = clip.manifest
        assert metadata["schema"] == "stable-envs/object-permanence-rigid-v3"
        assert metadata["scale_convention"] == oracle["scale_convention"]
        assert metadata["source"]["preparation_recipe"] == oracle["source_preparation_recipe"]
        assert oracle["source_scale_space"] == "absolute-world-matrix-decomposition"
        assert reference["source_scales"].shape == (120, 8, 3)
        assert reference["source_scales"].dtype == np.float64
        indices = np.arange(120)
        np.testing.assert_allclose(
            clip.object_scales(indices), reference["source_scales"] / reference["source_scales"][:1], atol=2e-6, rtol=0
        )
        rod = next(i for i, obj in enumerate(metadata["objects"]) if obj["name"] == "string_rod")
        node = metadata["objects"][rod]["node"]
        assert node >= 0 and metadata["dynamic_nodes"][node] == "string_rod"
        poses, scales = clip.poses(indices)[:, node], clip.scales(indices)[:, node]
        np.testing.assert_array_equal(poses, np.repeat(poses[:1], 120, axis=0))
        np.testing.assert_array_equal(scales[:30], np.ones((30, 3), np.float32))
        assert scales[30, 2] < 1 and scales[37, 2] > 0
        np.testing.assert_allclose(scales[37:, 2], 1e-4, atol=0, rtol=1e-7)
        np.testing.assert_array_equal(scales[:, :2], np.ones((120, 2), np.float32))


def test_old_recordings_keep_unit_relative_scale_without_optional_files():
    for task in ("g34", "g62", "g09", "g46"):
        with RigidTrajectory(RECORDINGS, task + "-nominal") as clip:
            assert not clip.has_scales and "scales" not in clip.manifest["arrays"]
            assert clip.manifest["schema"] != "stable-envs/object-permanence-rigid-v3"
            np.testing.assert_array_equal(clip.object_scales([0, 44, 119]), 1)


@pytest.mark.parametrize("field", ["policy", "space", "shape", "values"])
def test_source_scale_provenance_must_match_before_rendering(monkeypatch, field):
    from scripts.testing import check_object_permanence_rendering as checker

    metadata, reference = load_ray_geometry("g146")
    metadata, reference = dict(metadata), dict(reference)
    if field == "policy":
        metadata["scale_convention"] = "unknown-policy"
    elif field == "space":
        metadata["source_scale_space"] = "candidate-local-scale"
    elif field == "shape":
        reference["source_scales"] = reference["source_scales"][:, :1]
    else:
        reference["source_scales"] = reference["source_scales"].copy()
        reference["source_scales"][1, 0, 0] = 0
    monkeypatch.setattr(checker, "load_geometry_directory", lambda _: (metadata, reference))
    with pytest.raises(ValueError, match="scale"):
        checker.check(RECORDINGS, "unused", "G146", visibility_contract=metadata["visibility_contract"])
