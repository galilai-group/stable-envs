"""V2 orthographic coverage shares opaque depth/IDs and exact cache semantics."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("orthographic rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.testing.object_permanence_ray_reference import draw
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT, CONTRACT_V2
from stable_envs.envs.object_permanence.rendering.visibility_cuda import ContractRenderer
from stable_envs.render.nvdiff import Camera, Scene
from tests.unit.test_object_permanence_orthographic_contract import orthographic_camera
from tests.unit.test_object_permanence_ray_contract import camera, quad


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.21, -0.13)])
@pytest.mark.parametrize("clipping", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.37])
def test_orthographic_pixel_centers_axial_depth_interval_clipping_and_cache(shift, clipping, angle):
    metadata = orthographic_camera(near=0.5 if clipping else 0.1, far=1.5 if clipping else 10, shift=shift)
    points = quad()
    if clipping:
        points[:, 2] = -(1 + points[:, 0])
    world = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    world[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    world[:3, 3] = [2, -1, 4]
    metadata["world"] = world.tolist()
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    source_points = points @ world[:3, :3].T + world[:3, 3]
    expected_depth, expected_ids = draw(
        {"object": source_points[faces]}, metadata, 31, 47, visibility_contract=CONTRACT_V2
    )
    with Scene.from_triangles(points.astype(np.float32), [[0, 0, 1]] * 4, faces, [0] * 4, [[1, 0, 0, 1]]) as scene:
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
            semantic_ids=torch.tensor([1], device="cuda"),
            visibility_contract=CONTRACT_V2,
        )
        with (
            ContractRenderer(**options, static_geoms=torch.tensor([True], device="cuda")) as cached,
            ContractRenderer(**options) as full,
        ):
            cam = Camera(world[:3, 3], world[:3, :3], 90, metadata["near"], metadata["far"])
            cam.matrices = lambda **kwargs: (np.linalg.inv(world)[None], np.asarray(metadata["projection"]))
            assert cached.origin.shape == (31, 47, 3) and cached.rays.shape == (3,)
            for batch in (4, 1, 3, 7, 1):
                pos = torch.tensor(world[:3, 3], dtype=torch.float64, device="cuda").expand(batch, 1, 3)
                rot = torch.tensor(world[:3, :3], dtype=torch.float64, device="cuda").expand(batch, 1, 3, 3)
                actual = cached.render_many(pos, rot, cam)
                other = full.render_many(pos, rot, cam)
                assert all(torch.equal(actual[k], other[k]) for k in actual)
                np.testing.assert_array_equal(actual["ids"][0].cpu(), expected_ids)
                np.testing.assert_allclose(actual["depth"][0].cpu(), expected_depth, atol=1e-7, rtol=0)
                assert cached._ray_coverage.builds == 1


def test_v2_perspective_uses_the_same_layout_and_pixels_as_v1():
    points = quad().astype(np.float32)
    with Scene.from_triangles(points, [[0, 0, 1]] * 4, [[0, 1, 2], [0, 2, 3]], [0] * 4, [[0, 1, 0, 1]]) as scene:
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
            semantic_ids=torch.tensor([1], device="cuda"),
        )
        with (
            ContractRenderer(**options, visibility_contract=CONTRACT) as old,
            ContractRenderer(**options, visibility_contract=CONTRACT_V2) as new,
        ):
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(camera()["projection"]))
            pos = torch.zeros(3, 1, 3, device="cuda")
            rot = torch.eye(3, dtype=torch.float64, device="cuda").expand(3, 1, 3, 3)
            expected = old.render_many(pos, rot, cam)
            actual = new.render_many(pos, rot, cam)
            assert all(torch.equal(actual[k], expected[k]) for k in actual)
            assert new.origin.shape == (3,) and new.rays.shape == (31, 47, 3)


def test_per_pixel_orthographic_origin_is_part_of_the_exact_seed_key():
    from stable_envs.envs.object_permanence.rendering.visibility_cache import StaticRayCoverage
    from tests.gpu.test_object_permanence_visibility_cache_cuda import inputs, visibility

    data = inputs([True, False, True, False])
    grid = data["rays"].clone()
    grid[..., 2] = 0
    data["origin"] = grid
    data["rays"] = torch.tensor([0.0, 0.0, -1.0], device="cuda", dtype=torch.float64)
    p = torch.tensor(orthographic_camera()["projection"], device="cuda", dtype=torch.float64)
    data["positions"] = (
        torch.cat((data["vertices"], torch.ones_like(data["vertices"][..., :1])), -1) @ p.T
    ).contiguous()
    faces = torch.arange(12, device="cuda", dtype=torch.int32).reshape(4, 3)
    cache = StaticRayCoverage(faces, torch.tensor([True, False, True, False], device="cuda"))
    visibility(data, faces, cache)
    value = data["origin"][8, 12, 0]
    value.copy_(torch.nextafter(value, torch.full_like(value, float("inf"))))
    actual = visibility(data, faces, cache)
    expected = visibility(data, faces)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert cache.builds == 2 and cache.fallbacks == 0


@pytest.mark.parametrize("nearer", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_orthographic_semantic_ties_keep_a_strictly_nearer_object(nearer, reverse):
    points = quad().astype(np.float32)
    vertices = np.concatenate((points, points))
    if nearer:
        vertices[4:, 2] += np.spacing(np.float32(3))
    faces = np.array([[0, 1, 2], [0, 2, 3], [7, 6, 5], [7, 5, 4]], np.int32)
    if reverse:
        faces = faces[::-1].copy()
    metadata = orthographic_camera()
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
            camera_metadata=metadata,
            semantic_ids=torch.tensor([5, 20], device="cuda"),
            visibility_contract=CONTRACT_V2,
        ) as renderer:
            cam = Camera(np.zeros(3), np.eye(3), 90, 0.1, 10)
            cam.matrices = lambda **kwargs: (np.eye(4)[None], np.asarray(metadata["projection"]))
            pos = torch.zeros(2, 2, 3, device="cuda")
            rot = torch.eye(3, dtype=torch.float64, device="cuda").expand(2, 2, 3, 3)
            actual = renderer.render_many(pos, rot, cam)
            # Public IDs are geometry indices; metadata IDs determine ties.
            assert actual["ids"][0, 15, 23] == (2 if nearer else 1)
            expected = np.float32(3) - np.spacing(np.float32(3)) if nearer else 3
            assert actual["depth"][0, 15, 23] == expected
            for mode in actual:
                assert torch.equal(actual[mode], renderer.render(pos, rot, cam, mode=mode))
