"""Positive affine transforms preserve normals, visibility and static cache reuse."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("scale rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.testing.object_permanence_ray_reference import draw
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT, CONTRACT_V2
from stable_envs.envs.object_permanence.rendering.visibility_cuda import ContractRenderer
from stable_envs.render.nvdiff import Camera, Scene
from tests.helpers.reference.object_permanence_recordings import RECORDINGS
from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera
from tests.unit.test_object_permanence_ray_contract import camera, quad


@pytest.mark.parametrize("orthographic", [False, True])
def test_anisotropic_scale_rotation_normals_and_nearly_flat_surfaces(orthographic):
    metadata = orthographic_camera() if orthographic else camera()
    contract = CONTRACT_V2 if orthographic else CONTRACT
    moving = quad(0).astype(np.float32)
    moving[:, 2] = moving[:, 0] * 0.3 + moving[:, 1] * 0.2
    vertices = np.concatenate((quad(-6).astype(np.float32), moving))
    normal = np.array([-0.3, -0.2, 1], np.float32)
    normal /= np.linalg.norm(normal)
    normals = np.concatenate((np.tile([0, 0, 1], (4, 1)), np.tile(normal, (4, 1))))
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.int32)
    with Scene.from_triangles(
        vertices, normals, faces, [0] * 4 + [1] * 4, [[0.2, 0.4, 0.6, 1], [0.7, 0.3, 0.1, 1]]
    ) as scene:
        options = dict(
            scene=scene,
            res=(31, 47),
            textures=False,
            mipmap=False,
            skybox=False,
            shading="tiny",
            coverage_mode="refined",
            geometry_precision="float64",
            camera_metadata=metadata,
            semantic_ids=torch.tensor([1, 2], device="cuda"),
            visibility_contract=contract,
            affine_transforms=True,
        )
        with (
            ContractRenderer(**options, static_geoms=torch.tensor([True, False], device="cuda")) as cached,
            ContractRenderer(**options) as full,
        ):
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(metadata["projection"]))
            c, s = np.cos(0.37), np.sin(0.37)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            scale = np.array([[1, 1, 1], [0.5, 2, 1e-4], [2, 0.5, 1.7], [0.2, 0.3, 0.4]])
            affine = np.broadcast_to(np.eye(3), (4, 2, 3, 3)).copy()
            affine[:, 1] = rotation[None] * scale[:, None, :]
            positions = np.zeros((4, 2, 3), np.float32)
            positions[:, 1] = [0.11, -0.07, -3]
            pos = torch.tensor(positions, device="cuda")
            matrices = torch.tensor(affine, device="cuda")
            expected = full.render_many(pos, matrices, cam)
            _, nrm, _ = cached._clip(pos, matrices, np.eye(4)[None], np.asarray(metadata["projection"]))
            independent = normals[None] @ np.linalg.inv(affine[:, 1])
            independent = independent / np.linalg.norm(independent, axis=-1, keepdims=True)
            np.testing.assert_allclose(nrm[:, 4:].cpu(), independent[:, 4:], atol=1e-7, rtol=0)
            for row in range(4):
                world = moving.astype(np.float64) @ affine[row, 1].T + positions[row, 1]
                depth, ids = draw(
                    {"background": vertices[:4][faces[:2]], "moving": world[faces[2:] - 4]},
                    metadata,
                    31,
                    47,
                    visibility_contract=contract,
                )
                np.testing.assert_array_equal(expected["ids"][row].cpu(), ids)
                np.testing.assert_allclose(expected["depth"][row].cpu(), depth, atol=5e-7, rtol=0)
            for indices in ([0, 1, 2, 3], [3], [2, 1, 3], [1, 3, 2, 0, 1, 3, 2]):
                actual = cached.render_many(pos[indices], matrices[indices], cam)
                assert all(torch.equal(actual[k], expected[k][indices]) for k in actual)
                assert cached._ray_coverage.builds == 1 and cached._ray_coverage.fallbacks == 0


@pytest.mark.parametrize("size", [90, 180])
def test_g146_scale_only_node_static_cache_and_all_field_chunk_parity(size):
    with RigidTrajectory(RECORDINGS, "g146-nominal") as clip:
        assert clip.has_scales and clip.manifest["dynamic_nodes"] == [
            "hanging_ball",
            "release_hook_support",
            "string_rod",
        ]
        with pytest.raises(ValueError, match="explicit visibility contract"):
            RigidSceneRenderer(clip, size=size)
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as cached,
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT, cache_static=False) as full,
        ):
            expected = full.render_many(np.arange(120))
            for chunk in (4, 1, 3, 7, 4):
                cached.chunk_size = chunk
                indices = np.array([30, 33, 36, 37, 44, 0, 119, 44])
                actual = cached.render_many(indices)
                assert all(torch.equal(actual[k], expected[k][indices]) for k in actual)
                assert cached.renderer._ray_coverage.builds == 1
                assert cached.renderer._ray_coverage.fallbacks == 0
            for key in expected:
                assert torch.equal(cached.render(indices, mode=key), expected[key][indices])
