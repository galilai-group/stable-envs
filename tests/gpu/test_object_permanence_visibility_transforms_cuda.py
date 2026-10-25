"""Fused transforms preserve exact geometry, shading and static-cache inputs."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("visibility transforms require CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence.rendering.visibility_cuda import ContractRenderer
from stable_envs.render.nvdiff import Scene
from tests.unit.test_object_permanence_ray_contract import camera


@pytest.mark.parametrize("batch", [1, 3, 8])
@pytest.mark.parametrize("affine", [False, True])
def test_fused_geometry_matches_original_torch_bits_for_random_transforms(batch, affine):
    rng = np.random.default_rng(63)
    vertices = rng.normal(size=(63, 3)).astype(np.float32)
    normals = rng.normal(size=(63, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    faces = np.arange(63, dtype=np.int32).reshape(-1, 3)
    with Scene.from_triangles(vertices, normals, faces, np.repeat(np.arange(3), 21), [[1, 0, 0, 1]] * 3) as scene:
        with ContractRenderer(
            scene=scene,
            res=(16, 24),
            camera_metadata=camera(),
            semantic_ids=torch.arange(3, device="cuda"),
            geometry_precision="float64",
            coverage_mode="refined",
            textures=False,
            mipmap=False,
            skybox=False,
            affine_transforms=affine,
        ) as renderer:
            positions = torch.tensor(rng.normal(size=(batch, 3, 3)), dtype=torch.float32, device="cuda")
            rotations = torch.tensor(rng.normal(size=(batch, 3, 3, 3)), dtype=torch.float64, device="cuda")
            # Well-conditioned arbitrary linear transforms cover normal matrices,
            # negative components, ties in canonical scaling and partial blocks.
            rotations += 3 * torch.eye(3, device="cuda", dtype=torch.float64)
            view = torch.eye(4, dtype=torch.float64, device="cuda")[None]
            view[:, :3, :3] = torch.tensor([[0.9, -0.2, 0.1], [0.3, 0.8, -0.1], [-0.1, 0.2, 1.1]], device="cuda")
            view[:, :3, 3] = torch.tensor([0.5, -0.25, 0.1], device="cuda")
            projection = torch.tensor(camera()["projection"], device="cuda")
            actual = renderer._clip(positions, rotations, view, projection)
            world, planes = renderer.vertices, renderer.world_planes
            assert getattr(scene, "S4_64", None) is None
            expected = renderer._clip_reference(positions, rotations, view, projection)
            for name, left, right in zip(("clip", "normal", "shading_position"), actual, expected, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
            torch.testing.assert_close(world, renderer.vertices, rtol=0, atol=0, msg="world vertices")
            torch.testing.assert_close(planes, renderer.world_planes, rtol=0, atol=0)


@pytest.mark.parametrize("task", ["g98", "g132", "g103", "g09", "g146", "g90"])
@pytest.mark.parametrize("size", [90, 180])
def test_fused_and_reference_rendering_preserve_all_fields_and_static_cache(task, size):
    from stable_envs.envs.object_permanence import RigidTrajectory
    from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
    from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT, CONTRACT_V2
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS

    with RigidTrajectory(RECORDINGS, task + "-nominal") as clip:
        contract = clip.manifest.get("visibility_contract") or (
            CONTRACT_V2 if clip.manifest["camera"].get("projection_type") == "orthographic" else CONTRACT
        )
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract=contract) as fused,
            RigidSceneRenderer(clip, size=size, visibility_contract=contract) as reference,
        ):
            reference.renderer.fuse_geometry = False
            modes = ("rgb", "depth", "ids")
            if task == "g90":
                modes += ("masks", "weights", "layer_depth", "layer_ids")
            frames = np.arange(120) if task in ("g98", "g132") else np.array([0, 44, 119, 44, 23, 88, 1])
            for start in range(0, len(frames), 4):
                rows = frames[start : start + 4]
                actual = fused.render_many(rows, modes=modes)
                expected = reference.render_many(rows, modes=modes)
                assert all(torch.equal(actual[k], expected[k]) for k in modes), (task, size, start)
            left = getattr(fused.renderer, "_layer_cache", fused.renderer._ray_coverage)
            right = getattr(reference.renderer, "_layer_cache", reference.renderer._ray_coverage)
            assert left.builds == right.builds == 1
            if task in ("g98", "g132"):
                frames = [119, 44, 0, 44, 67]
                expected = reference.render_many(frames, modes=modes)
                for chunk in (1, 3, 5):
                    fused.chunk_size = chunk
                    actual = fused.render_many(frames, modes=modes)
                    assert all(torch.equal(actual[k], expected[k]) for k in modes)
                assert left.builds == 1 and left.fallbacks == 0
