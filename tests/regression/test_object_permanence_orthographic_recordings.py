"""All V2 source poses and task objects survive orthographic recording."""

from copy import deepcopy

import numpy as np
import pytest

from stable_envs.envs.object_permanence import RigidTrajectory
from tests.helpers.reference.object_permanence_orthographic import (
    ORTHOGRAPHIC_TASKS,
    load_orthographic_geometry,
)
from tests.helpers.reference.object_permanence_recordings import RECORDINGS


@pytest.mark.parametrize("task", ORTHOGRAPHIC_TASKS)
def test_all_orthographic_source_poses_provenance_and_observable_probes(task):
    from scripts.testing.freeze_object_permanence import REFERENCE_CASES

    manifest, reference = load_orthographic_geometry(task)
    assert reference["depth"].shape == reference["object_ids"].shape == (120, 180, 320)
    assert reference["depth"].dtype == np.float32 and np.isfinite(reference["depth"]).all()
    np.testing.assert_array_equal(reference["depth"][0], reference["depth_base"])
    assert manifest["source_semantic_targets"] == list(REFERENCE_CASES[task.upper()][2])
    observed = []
    for name in manifest["source_semantic_targets"]:
        visible = (reference["object_ids"] == manifest["object_ids"][name]).sum((1, 2))
        if visible.max():
            observed.append(name)
            np.testing.assert_array_equal(visible, manifest["target_visible_pixels"][name])
    assert manifest["targets"] == observed and observed
    with RigidTrajectory(RECORDINGS, task + "-nominal") as clip:
        candidate = clip.manifest
        assert candidate["schema"] == "stable-envs/object-permanence-rigid-v2"
        assert candidate["visibility_contract"] == manifest["visibility_contract"]
        assert candidate["source"]["preparation_recipe"] == manifest["source_preparation_recipe"]
        assert candidate["source"]["script_sha256"] == manifest["source_script_sha256"]
        assert candidate["source"].get("execution_context") == manifest.get("source_execution_context")
        assert manifest["object_ids"] == {o["name"]: o["id"] for o in candidate["objects"]}
        assert set(manifest["source_semantic_targets"]) <= set(manifest["object_ids"])
        np.testing.assert_array_equal(clip.frames, reference["source_frames"])
        actual = clip.object_poses(np.arange(120))
        expected = reference["poses"]
        np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=3e-7, rtol=0)
        signs = np.where((actual[..., 3:] * expected[..., 3:]).sum(-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(actual[..., 3:], expected[..., 3:] * signs, atol=1e-6, rtol=0)


def test_registered_orthographic_cases_and_default_v1_are_disjoint():
    from scripts.export.object_permanence import AUDITED_TASKS, CONTRACT_ONLY_TASKS
    from scripts.export.object_permanence import ORTHOGRAPHIC_TASKS as EXPORT_TASKS
    from tests.helpers.reference.object_permanence_recordings import RAY_TASKS

    expected = {t.upper() for t in ORTHOGRAPHIC_TASKS}
    assert expected == EXPORT_TASKS
    assert expected <= set(AUDITED_TASKS) and expected <= CONTRACT_ONLY_TASKS
    assert not (expected & {t.upper() for t in RAY_TASKS})


@pytest.mark.parametrize("change", ["recipe", "missing-recipe", "camera"])
def test_reference_definition_mismatch_fails_before_constructing_a_renderer(monkeypatch, change):
    from scripts.testing import check_object_permanence_rendering as checker
    from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT_V2

    manifest, arrays = load_orthographic_geometry("g09")
    mismatched = deepcopy(manifest)
    if change == "recipe":
        mismatched["source_preparation_recipe"]["animate"] = "different_animation"
    elif change == "missing-recipe":
        mismatched.pop("source_preparation_recipe")
    else:
        mismatched["camera"]["projection_type"] = "perspective"
    monkeypatch.setattr(checker, "load_geometry_directory", lambda _: (mismatched, arrays))
    with pytest.raises(ValueError, match="recipes differ|cameras differ"):
        checker.check(RECORDINGS, "unused", "g09", visibility_contract=CONTRACT_V2)
