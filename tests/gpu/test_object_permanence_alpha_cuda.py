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


@pytest.mark.parametrize("orthographic", [False, True])
@pytest.mark.parametrize("coplanar", [False, True])
@pytest.mark.parametrize("alpha", [0, 0.25, 1])
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


@pytest.mark.parametrize("moving_alpha", [0.4, 1.0])
def test_static_alpha_lists_merge_interleaved_moving_objects_and_invalidate_exactly(moving_alpha):
    points = np.concatenate((quad(-2), quad(-3), quad(-5))).astype(np.float32)
    faces = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + i * 4 for i in range(3)]).astype(np.int32)
    with Scene.from_triangles(
        points,
        [[0, 0, 1]] * 12,
        faces,
        [0] * 4 + [1] * 4 + [2] * 4,
        [[1, 0, 0, 0.2], [0, 1, 0, moving_alpha], [0, 0, 1, 1]],
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
            camera_metadata=camera(),
            semantic_ids=torch.tensor([1, 2, 3], device="cuda"),
            max_layers=4,
        )
        with (
            AlphaRenderer(**options, static_geoms=torch.tensor([True, False, True], device="cuda")) as cached,
            AlphaRenderer(**options) as full,
        ):
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(camera()["projection"]))
            positions = torch.zeros(4, 3, 3, device="cuda")
            positions[:, 1, 2] = torch.tensor([2.0, 0.0, -3.0, 1.0], device="cuda")
            matrices = torch.eye(3, dtype=torch.float64, device="cuda").expand(4, 3, 3, 3)
            modes = ("rgb", "depth", "ids", "masks", "weights", "layer_depth", "layer_ids")
            expected = full.render_many(positions, matrices, cam, modes=modes)
            for indices in ([0, 1, 2, 3], [1], [3, 0, 2], [2, 1, 1, 3, 0]):
                out = cached.render_many(positions[indices], matrices[indices], cam, modes=modes)
                assert all(torch.equal(out[k], expected[k][indices]) for k in out)
                assert cached._layer_cache.builds == 1 and cached._layer_cache.fallbacks == 0
                out["layer_depth"].zero_()
            scene.alpha[0] = 0.3
            actual = cached.render_many(positions, matrices, cam, modes=modes)
            other = full.render_many(positions, matrices, cam, modes=modes)
            assert all(torch.equal(actual[k], other[k]) for k in actual)
            assert cached._layer_cache.builds == 2
            varied = positions.clone()
            varied[:, 0, 0] = torch.tensor([0, 0.1, 0, 0.1], device="cuda")
            actual = cached.render_many(varied, matrices, cam, modes=modes)
            other = full.render_many(varied, matrices, cam, modes=modes)
            assert all(torch.equal(actual[k], other[k]) for k in actual)
            assert cached._layer_cache.fallbacks == 1 and cached._layer_cache.builds == 2
            cached.render_many(positions, matrices, cam, modes=modes)
            assert cached._layer_cache.builds == 2


def test_alpha_capacity_overflow_is_reported_in_full_and_cached_paths():
    points = np.concatenate((quad(-2), quad(-3), quad(-4))).astype(np.float32)
    faces = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + i * 4 for i in range(3)]).astype(np.int32)
    with Scene.from_triangles(
        points, [[0, 0, 1]] * 12, faces, [0] * 4 + [1] * 4 + [2] * 4, [[1, 0, 0, 0.2], [0, 1, 0, 0.4], [0, 0, 1, 1]]
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
            camera_metadata=camera(),
            semantic_ids=torch.tensor([1, 2, 3], device="cuda"),
            max_layers=2,
        )
        cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
        cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(camera()["projection"]))
        pos = torch.zeros(1, 3, 3, device="cuda")
        rot = torch.eye(3, dtype=torch.float64, device="cuda").expand(1, 3, 3, 3)
        for mask in (None, torch.tensor([True, False, True], device="cuda")):
            with AlphaRenderer(**options, static_geoms=mask) as r:
                with pytest.raises(ValueError, match="capacity"):
                    r.render_many(pos, rot, cam, modes=("ids",))


def test_thick_glass_counts_front_and_back_and_zero_alpha_is_empty():
    points = np.concatenate((quad(-2), quad(-2.5), quad(-3))).astype(np.float32)
    faces = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + i * 4 for i in range(3)]).astype(np.int32)
    with Scene.from_triangles(
        points, [[0, 0, 1]] * 12, faces, [0] * 8 + [1] * 4, [[1, 0, 0, 0.2], [0, 1, 0, 1]]
    ) as scene:
        with AlphaRenderer(
            scene=scene,
            res=(31, 47),
            textures=False,
            mipmap=False,
            skybox=False,
            shading="tiny",
            coverage_mode="refined",
            geometry_precision="float64",
            camera_metadata=camera(),
            semantic_ids=torch.tensor([1, 2], device="cuda"),
            max_layers=4,
        ) as r:
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(camera()["projection"]))
            pos = torch.zeros(1, 2, 3, device="cuda")
            rot = torch.eye(3, dtype=torch.float64, device="cuda").expand(1, 2, 3, 3)
            out = r.render_many(pos, rot, cam, modes=("rgb", "layer_ids", "weights", "masks"))
            np.testing.assert_array_equal(out["layer_ids"][0, 15, 23].cpu(), [1, 1, 2, 0])
            np.testing.assert_allclose(out["weights"][0, :, 15, 23].cpu(), [0.36, 0.64], atol=1e-7, rtol=0)
            scene.alpha.zero_()
            pos[..., 2] = 10  # Omitted alpha-zero surfaces may lie behind the camera.
            out = r.render_many(pos, rot, cam, modes=("rgb", "depth", "ids", "masks", "weights", "layer_ids"))
            assert not any(out[k].any() for k in out if k != "rgb")
            bg = QUANT(torch.tensor(r.background, dtype=torch.float32, device="cuda"))
            assert torch.equal(out["rgb"], bg.expand_as(out["rgb"]))
