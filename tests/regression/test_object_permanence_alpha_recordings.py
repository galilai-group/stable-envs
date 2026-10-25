"""Pinned alpha recordings keep source identity, layers and hidden trajectories."""

from pathlib import Path

import numpy as np
import pytest

from scripts.export.object_permanence import ALPHA_TASKS
from scripts.testing.check_object_permanence_alpha import load
from stable_envs.envs.object_permanence import RigidTrajectory
from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS


@pytest.mark.parametrize("task", sorted(ALPHA_TASKS))
def test_alpha_reference_pose_material_policy_and_every_source_frame(task):
    metadata, source = load(FIXTURES / f"semantics/object_permanence/{task.lower()}/transparent_pixel_rays_v1/nominal")
    with RigidTrajectory(RECORDINGS, task.lower() + "-nominal") as clip:
        m = clip.manifest
        assert m["schema"] == "stable-envs/object-permanence-rigid-v4"
        assert m["material_convention"] == metadata["material_convention"]
        assert m["visibility_contract"] == metadata["visibility_contract"]
        assert m["source"]["script_sha256"] == metadata["source_script_sha256"]
        assert m["source"]["preparation_recipe"] == metadata["source_preparation_recipe"]
        np.testing.assert_array_equal(clip.frames, source["source_frames"])
        np.testing.assert_array_equal(metadata["alphas"], [o["rgba"][3] for o in m["objects"]])
        assert metadata["object_order"] == [o["name"] for o in m["objects"]]
        p = clip.object_poses(np.arange(len(clip)))
        expected = source["poses"]
        np.testing.assert_allclose(p[..., :3], expected[..., :3], atol=3e-7, rtol=0)
        signs = np.where((p[..., 3:] * expected[..., 3:]).sum(-1, keepdims=True) < 0, -1, 1)
        np.testing.assert_allclose(p[..., 3:], expected[..., 3:] * signs, atol=1e-6, rtol=0)
        ids = source["layer_ids"]
        alpha = np.array([0, *metadata["alphas"]], np.float32)
        for row in range(len(clip)):
            opaque = (ids[row] > 0) & (alpha[ids[row]] == 1)
            assert not (np.cumsum(opaque, -1) - opaque > 0)[ids[row] > 0].any()
        assert (ids > 0).sum(-1).max() == metadata["observed_layers_max"] <= metadata["max_layers"]
        assert len(clip) == 120


def test_alpha_oracle_implementation_has_no_candidate_imports():
    import ast

    for name in ("object_permanence_alpha_reference.py", "freeze_object_permanence_alpha.py"):
        tree = ast.parse((Path("scripts/testing") / name).read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(
            n and (n.startswith("stable_envs") or n.startswith("scripts.export") or n.startswith("torch"))
            for n in imports
        )
