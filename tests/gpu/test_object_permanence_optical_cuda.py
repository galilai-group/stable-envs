"""Arbitrary-ray BVH queries, optical laws and full/cached recorded paths."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("optical replay requires CUDA", allow_module_level=True)

from scripts.testing.object_permanence_optical_reference import uniform
from stable_envs.envs.object_permanence.rendering.optical_scene import dielectric, random_event
from stable_envs.envs.object_permanence.rendering.optics import snell
from stable_envs.envs.object_permanence.rendering.optics_cuda import TriangleQuery
from stable_envs.envs.object_permanence.rendering.visibility import local_planes


def test_counter_stream_matches_independent_uint32_reference_and_order():
    ray = np.array([0, 1, 8, 999, 460799, 10000000], np.int64)
    tensor = torch.tensor(ray, device="cuda")
    for event in (0, 1, 31, 127):
        np.testing.assert_array_equal(random_event(tensor, event).cpu(), uniform(ray, event))
        np.testing.assert_array_equal(random_event(tensor.flip(0), event).flip(0).cpu(), uniform(ray, event))


@pytest.mark.parametrize("n1,n2,angle", [(1.0, 1.45, 0.0), (1.0, 1.45, 60.0), (1.45, 1.0, 60.0), (1.45, 1.45, 90.0)])
def test_cuda_snell_and_fresnel_choices_match_analytic_equations(n1, n2, angle):
    a = np.deg2rad(angle)
    d = np.array([[np.sin(a), 0, -np.cos(a)]], np.float64)
    n = np.array([[0, 0, 1]], np.float64)
    reflected, refracted, f = snell(d, n, n1, n2)
    x = torch.tensor(d, device="cuda")
    normal = torch.tensor(n, device="cuda")
    for choice in (0.0, 0.999999):
        actual, reflection = dielectric(
            x,
            normal,
            torch.tensor([n1], dtype=torch.float64, device="cuda"),
            torch.tensor([n2], dtype=torch.float64, device="cuda"),
            torch.tensor([choice], device="cuda"),
        )
        expected = reflected if choice < f[0] else refracted
        np.testing.assert_allclose(actual.cpu(), expected, atol=2e-15, rtol=0)
        assert reflection.item() == bool(choice < f[0])


def test_bvh_arbitrary_parallel_inside_miss_and_shared_edge_rays():
    square = np.array([[[-1, -1, -2], [1, -1, -2], [1, 1, -2]], [[-1, -1, -2], [1, 1, -2], [-1, 1, -2]]], np.float64)
    tri = [square, square.copy()]
    query = TriangleQuery(tri, semantic_ids=[20, 10])
    points = np.array([[0, 0, 0], [0, 0, -3], [2, 0, 0], [-1, -1, 0], [0, 0, 0], [0, 0, -2]], np.float64)
    directions = np.array([[0, 0, -1], [0, 0, 1], [0, 0, -1], [0, 0, -1], [1, 0, 0], [0, 0, -1]], np.float64)
    planes = np.concatenate([local_planes(t.reshape(-1, 3), np.arange(t.size // 3).reshape(-1, 3)) for t in tri])
    t, face, normal = query(
        torch.tensor(points, device="cuda"),
        torch.tensor(directions, device="cuda"),
        torch.zeros(2, 3, device="cuda", dtype=torch.float64),
        torch.eye(3, device="cuda", dtype=torch.float64).expand(2, 3, 3).contiguous(),
        torch.ones(2, device="cuda", dtype=torch.bool),
        torch.tensor(planes, device="cuda"),
    )
    np.testing.assert_array_equal(face.cpu(), [2, 2, -1, 2, -1, -1])
    np.testing.assert_array_equal(t.cpu(), [2, 1, np.inf, 2, np.inf, np.inf])
    np.testing.assert_array_equal(normal[[0, 1, 3]].cpu(), [[0, 0, 1]] * 3)


def test_optical_cached_resolve_keeps_semantic_ties_and_blocked_paths():
    import triton

    from stable_envs.envs.object_permanence.rendering.optical_cache_cuda import composite

    device = "cuda"
    n = 5
    ray = torch.arange(n, device=device)
    travel = torch.full((n,), 0.25, device=device, dtype=torch.float64)
    st = torch.tensor([2.0, 2.0, float("inf"), float("inf"), 2.0], device=device, dtype=torch.float64)
    sf = torch.tensor([0, 1, -1, -1, 0], device=device)
    dt = torch.tensor([2.0, 2.0, 1.0, float("inf"), 1.0], device=device, dtype=torch.float64)
    df = torch.tensor([1, 0, 1, -1, 1], device=device)
    normal = torch.tensor([[0.0, 0.0, 1.0]] * n, device=device, dtype=torch.float64)
    geometry = torch.tensor([0, 1], device=device)
    semantics = torch.tensor([20, 10], device=device, dtype=torch.int32)
    labels = semantics.to(torch.int16)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device, dtype=torch.float64)
    axis = torch.ones(n, device=device, dtype=torch.float64)
    alive = torch.tensor([True, True, True, True, False], device=device)
    rgb = torch.full((n, 3), 0.125, device=device, dtype=torch.float64)
    depth = torch.zeros(n, device=device, dtype=torch.float64)
    ids = torch.zeros(n, device=device, dtype=torch.int16)
    paths = torch.full((2 * n,), float("inf"), device=device, dtype=torch.float64)
    composite[(triton.cdiv(n, 128),)](
        ray,
        travel,
        st,
        sf,
        dt,
        df,
        normal,
        geometry,
        semantics,
        labels,
        colors,
        axis,
        alive,
        rgb,
        depth,
        ids,
        paths,
        n,
        n,
        1,
        True,
        True,
        128,
        enable_fp_fusion=False,
    )
    np.testing.assert_array_equal(ids.cpu(), [10, 10, 10, 0, 0])
    np.testing.assert_array_equal(alive.cpu(), [False, True, False, True, False])
    np.testing.assert_array_equal(depth.cpu(), [2, 2, 1, 0, 0])
    np.testing.assert_array_equal(paths.reshape(2, n)[1, :3].cpu(), [2.25, 2.25, 1.25])
    np.testing.assert_array_equal(rgb[[1, 3, 4]].cpu(), [[0.125] * 3] * 3)
