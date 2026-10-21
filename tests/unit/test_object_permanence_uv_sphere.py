"""Faceted hits agree with independent triangle intersections, including fallback."""

import numpy as np
import pytest
import torch

from stable_envs.render.nvdiff import _sphere
from stable_envs.render.primitives import INF, SPHERE
from stable_envs.render.raycast3d import intersect
from stable_envs.render.uv_sphere import refine_uv_sphere


def _triangles(origin, direction, radius, slices, stacks):
    vertices, _, faces = _sphere([radius] * 3, slices, stacks)
    triangles = vertices[faces]
    a, b, c = (triangles[:, i] for i in range(3))
    e1, e2 = b - a, c - a
    cross = np.cross(direction[:, None], e2[None])
    determinant = (cross * e1[None]).sum(-1)
    inv = np.divide(1, determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > 1e-15)
    delta = origin[:, None] - a[None]
    u = (delta * cross).sum(-1) * inv
    q = np.cross(delta, e1[None])
    v = (direction[:, None] * q).sum(-1) * inv
    t = (e2[None] * q).sum(-1) * inv
    valid = (np.abs(determinant) > 1e-15) & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 0)
    return np.where(valid, t, INF).min(-1)


@pytest.mark.parametrize("iterations", [0, 8])
@pytest.mark.parametrize("slices,stacks", [(48, 24), (12, 7)])
def test_uv_sphere_matches_triangles_for_grazing_polar_and_inside_rays(iterations, slices, stacks, device="cpu"):
    rng = np.random.default_rng(3)
    direction = rng.normal(size=(160, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    origin = rng.normal(size=(160, 3))
    origin /= np.linalg.norm(origin, axis=-1, keepdims=True)
    origin *= rng.uniform(0.02, 0.3, size=(160, 1))
    direction[:80] = -origin[:80] + rng.uniform(-0.06, 0.06, size=(80, 3))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    o, d = torch.from_numpy(origin).to(device), torch.from_numpy(direction).to(device)
    analytic, _ = intersect(SPHERE, o.new_tensor([0.1, 0, 0]), o.new_zeros(3), torch.eye(3, device=device), o, d)
    selected = analytic < INF
    actual, normal = refine_uv_sphere(
        o[selected],
        d[selected],
        analytic[selected],
        radius=0.1,
        slices=slices,
        stacks=stacks,
        max_iterations=iterations,
    )
    selected = selected.cpu().numpy()
    expected = _triangles(origin[selected], direction[selected], 0.1, slices, stacks)
    np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(torch.linalg.vector_norm(normal, dim=-1).cpu(), 1, atol=1e-12, rtol=0)
