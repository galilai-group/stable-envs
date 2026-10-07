from types import SimpleNamespace

import pytest
import torch

from stable_envs.render import cpu3d
from stable_envs.render import raycast3d as rays


@pytest.mark.parametrize("kind", [rays.SPHERE, rays.CAPSULE, rays.CYLINDER, rays.BOX, rays.ELLIPSOID, rays.PLANE])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_occlusion_distances_match_primary_intersection(kind, dtype):
    generator = torch.Generator().manual_seed(710)
    origins = torch.randn(3, 257, 3, generator=generator, dtype=dtype)
    directions = torch.randn(3, 257, 3, generator=generator, dtype=dtype)
    directions /= directions.norm(dim=-1, keepdim=True)
    positions = torch.randn(3, 1, 3, generator=generator, dtype=dtype)
    rotations = torch.linalg.qr(torch.randn(3, 1, 3, 3, generator=generator, dtype=dtype))[0]
    size = torch.tensor([0.8, 0.6, 0.4], dtype=dtype)
    expected, _ = rays.intersect(kind, size, positions, rotations, origins, directions)
    actual = rays._intersect_distance(kind, size, positions, rotations, origins, directions)
    assert torch.equal(actual, expected)
    assert (expected < rays.INF).any()
    assert (expected == rays.INF).any()


@pytest.mark.parametrize("scale", [0.0, 0.7, 1.0, 1.3])
def test_compact_shadow_rays_match_independent_full_intersections(scale):
    generator = torch.Generator().manual_seed(810)
    kinds = [rays.SPHERE, rays.CAPSULE, rays.CYLINDER, rays.BOX, rays.ELLIPSOID]
    sizes = torch.tensor([[0.4, 0.6, 0.5]]).expand(5, -1)
    scene = SimpleNamespace(kinds=kinds, sizes=sizes)
    positions = torch.randn(4, 5, 3, generator=generator)
    rotations = torch.linalg.qr(torch.randn(4, 5, 3, 3, generator=generator))[0]
    # Deliberately shuffled worlds with no rays in world 2.
    env = torch.randint(0, 3, (1031,), generator=generator)
    env[env == 2] = 3
    origins = torch.randn(1031, 3, generator=generator)
    directions = torch.randn(1031, 3, generator=generator)
    directions *= scale / directions.norm(dim=-1, keepdim=True)
    limits = torch.rand(1031, generator=generator) * 5
    limits[::11] = 0
    expected = torch.zeros(1031, dtype=torch.bool)
    for k, kind in enumerate(kinds):
        distance, _ = rays.intersect(kind, sizes[k], positions[env, k], rotations[env, k], origins, directions)
        expected |= (distance > 0) & (distance < limits)
    actual = cpu3d._any_blocking_flat(scene, positions, rotations, env, origins, directions, limits)
    assert torch.equal(actual, expected)


def test_empty_compact_shadow_rays():
    scene = SimpleNamespace(kinds=[rays.SPHERE], sizes=torch.ones(1, 3))
    result = cpu3d._any_blocking_flat(
        scene,
        torch.zeros(1, 1, 3),
        torch.eye(3).reshape(1, 1, 3, 3),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, 3),
        torch.empty(0, 3),
        torch.empty(0),
    )
    assert result.shape == (0,)
    assert result.dtype == torch.bool
