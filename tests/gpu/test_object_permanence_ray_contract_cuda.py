"""Canonical visibility: all-frame source fidelity plus generic adversarial geometry."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("ray visibility requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence.rendering.visibility_cuda import ContractRenderer
from stable_envs.render.nvdiff import Camera, Scene
from tests.helpers.reference.object_permanence_recordings import RAY_TASKS, RECORDINGS
from tests.unit.test_object_permanence_ray_contract import camera, quad


@pytest.mark.parametrize("nearer", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tilted", [False, True])
def test_semantic_ties_winding_retriangulation_nearer_surfaces_and_partial_cuda_blocks(nearer, reverse, tilted):
    points = quad().astype(np.float32)
    if tilted:
        points[:, 2] += points[:, 0] * 0.5 + points[:, 1] * 0.25
    vertices = np.concatenate((points, points))
    vertices[4:, 2] -= np.spacing(np.float32(3)) if nearer else 0
    faces = np.array([[0, 1, 2], [0, 2, 3], [7, 6, 5], [7, 5, 4]], np.int32)
    if reverse:
        faces = faces[::-1].copy()
    with Scene.from_triangles(
        vertices, [[0, 0, 1]] * 8, faces, [0] * 4 + [1] * 4, [[1, 0, 0, 1], [0, 1, 0, 1]]
    ) as scene:
        with ContractRenderer(
            scene=scene,
            res=(31, 47),
            textures=False,
            mipmap=False,
            skybox=False,
            shading="tiny",
            coverage_mode="refined",
            geometry_precision="float64",
            camera_metadata=camera(),
            semantic_ids=torch.tensor([20, 5], device="cuda"),
        ) as r:
            pos = torch.zeros(2, 2, 3, device="cuda")
            rot = torch.eye(3, dtype=torch.float64, device="cuda").expand(2, 2, 3, 3)
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(camera()["projection"]))
            output = r.render_many(pos, rot, cam)
            # Public IDs here are geometry indices; semantic IDs choose winner.
            assert output["ids"][0, 15, 23] == (1 if nearer else 2)
            assert output["depth"][0, 15, 23] == 3
            for mode, field in output.items():
                assert torch.equal(field, r.render(pos, rot, cam, mode=mode))
                assert torch.equal(field[:1], r.render(pos[:1], rot[:1], cam, mode=mode))
            buffers = {k: torch.empty_like(v) for k, v in output.items()}
            assert r.render_many(pos, rot, cam, out=buffers) is buffers
            assert all(torch.equal(buffers[k], output[k]) for k in output)
            bad = pos.clone()
            bad[..., 2] = 10
            with pytest.raises(ValueError, match="depth-plane crossing"):
                r.render_many(bad, rot, cam)


@pytest.mark.parametrize("task", RAY_TASKS)
@pytest.mark.parametrize("cache_static", [True, False])
def test_versioned_all_frame_source_depth_and_all_object_masks(task, cache_static):
    from scripts.testing.check_object_permanence_rendering import check
    from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT
    from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS

    oracle = FIXTURES / f"semantics/object_permanence/{task}/opaque_pixel_rays_v1/nominal"
    result = check(RECORDINGS, oracle, task, visibility_contract=CONTRACT, cache_static=cache_static)
    assert result["frames"] == 120 and result["certification_eligible"]
    assert result["passed"], result["failures"]


@pytest.mark.parametrize("task", RAY_TASKS)
@pytest.mark.parametrize("size", [90, 180])
def test_ray_static_cache_is_bit_identical_for_all_fields_and_shuffled_frames(task, size):
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT

    frames = np.arange(120)[::-1].copy()
    with RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip:
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as cached,
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT, cache_static=False) as full,
        ):
            expected = full.render_many(frames)
            actual = cached.render_many(frames)
            assert all(torch.equal(actual[m], expected[m]) for m in actual)
            assert cached.renderer._ray_coverage.builds == 1
            indices = np.array([44, 0, 119, 44, 67])
            selected = cached.render_many(indices)
            assert all(torch.equal(selected[m], expected[m][119 - indices]) for m in selected)
            assert cached.renderer._ray_coverage.builds == 1
            selected["ids"].zero_()
            assert torch.equal(cached.render(indices, mode="ids"), expected["ids"][119 - indices])


def test_versioned_contract_cannot_be_mixed_with_legacy_or_camera_diagnostics():
    from scripts.testing.check_object_permanence_rendering import check
    from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT
    from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS

    new = FIXTURES / "semantics/object_permanence/g63/opaque_pixel_rays_v1/nominal"
    old = FIXTURES / "semantics/object_permanence/g62/nominal"
    with pytest.raises(ValueError, match="contracts differ"):
        check(RECORDINGS, new, "g63")
    with pytest.raises(ValueError, match="contracts differ"):
        check(RECORDINGS, old, "g62", visibility_contract=CONTRACT)
    with pytest.raises(ValueError, match="diagnostic camera"):
        check(RECORDINGS, new, "g63", visibility_contract=CONTRACT, camera_probe={})
