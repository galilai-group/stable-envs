"""Exact alpha layers and simplified compositing match independent geometry."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("alpha tests require CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.testing.object_permanence_alpha_reference import draw
from stable_envs.envs.object_permanence.rendering.transparency_cuda import AlphaRenderer
from stable_envs.render.nvdiff import QUANT, Camera, Scene, shade_tiny
from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera
from tests.unit.test_object_permanence_ray_contract import camera, quad


@pytest.mark.parametrize("orthographic,coplanar,alpha", [(False, True, 0.25), (True, False, 1.0)])
def test_gpu_alpha_shared_edges_overlapping_masks_ties_and_composition(orthographic, coplanar, alpha):
    metadata = orthographic_camera() if orthographic else camera()
    points = np.concatenate((quad(-2), quad(-2 if coplanar else -3), quad(-4))).astype(np.float32)
    faces = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + i * 4 for i in range(3)]).astype(np.int32)
    rgba = np.array([[1, 0, 0, alpha], [0, 1, 0, 0.4 if coplanar else 1], [0, 0, 1, 1]], np.float32)
    with Scene.from_triangles(points, [[0, 0, 1]] * 12, faces, [0] * 4 + [1] * 4 + [2] * 4, rgba) as scene:
        with AlphaRenderer(
            scene=scene,
            res=(31, 47),
            textures=False,
            mipmap=False,
            skybox=False,
            shading="tiny",
            coverage_mode="refined",
            geometry_precision="float64",
            camera_metadata=metadata,
            semantic_ids=torch.tensor([1, 2, 3], device="cuda"),
            max_layers=4,
        ) as r:
            assert r.visibility_contract == "stable-envs/transparent-pixel-rays-v1"
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.array(metadata["projection"]))
            positions = torch.zeros(2, 3, 3, device="cuda")
            matrices = torch.eye(3, dtype=torch.float64, device="cuda").expand(2, 3, 3, 3)
            modes = ("rgb", "depth", "ids", "layer_depth", "layer_ids", "masks", "weights")
            actual = r.render_many(positions, matrices, cam, modes=modes)
            expected = draw(
                {n: points[faces[i * 2 : i * 2 + 2]] for i, n in enumerate(("a", "b", "c"))},
                dict(zip(("a", "b", "c"), rgba[:, 3])),
                metadata,
                31,
                47,
                max_layers=4,
            )
            np.testing.assert_array_equal(actual["layer_ids"][0].cpu(), expected["layer_ids"])
            np.testing.assert_allclose(actual["layer_depth"][0].cpu(), expected["layer_depth"], atol=1e-7, rtol=0)
            for index in range(3):
                np.testing.assert_array_equal(
                    actual["masks"][0, index].cpu(), (expected["layer_ids"] == index + 1).any(-1)
                )
            colors = shade_tiny(torch.tensor([[0.0, 0.0, 1.0]], device="cuda"), scene.mat_rgb, r.tiny).clamp(0, 1)
            rgb = torch.zeros(3, device="cuda")
            t = 1.0
            expected_weights = np.zeros(3, np.float32)
            weight_t = np.float32(1)
            for label in expected["layer_ids"][15, 23]:
                if not label:
                    break
                a = float(rgba[label - 1, 3])
                expected_weights[label - 1] += weight_t * rgba[label - 1, 3]
                weight_t *= np.float32(1) - rgba[label - 1, 3]
                rgb += t * a * colors[label - 1]
                t *= 1 - a
            rgb += t * torch.tensor(r.background, device="cuda")
            assert torch.equal(actual["rgb"][0, 15, 23], QUANT(rgb))
            np.testing.assert_array_equal(actual["weights"][0, :, 15, 23].cpu(), expected_weights)
            buffers = {key: torch.empty_like(value) for key, value in actual.items()}
            assert r.render_many(positions, matrices, cam, modes=modes, out=buffers) is buffers
            assert all(torch.equal(buffers[k], v) for k, v in actual.items())
            for mode in modes:
                assert torch.equal(actual[mode], r.render_many(positions, matrices, cam, modes=(mode,))[mode])
