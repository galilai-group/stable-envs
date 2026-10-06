"""The CPU memory-layout optimization preserves the previous bound predicate."""

import pytest
import torch

from stable_envs.render import raycast3d as rays
from stable_envs.render.primitives import dot3


def previous_bound(pos, rad, origins, directions, dd):
    """Frozen pre-optimization packed-vector expression, independent of dispatch."""
    offset = pos - origins
    cross = torch.cross(offset.expand_as(directions), directions, dim=-1)
    radius = rad * rays._BOUND_SLACK_REL + rays._BOUND_SLACK_ABS
    squared = radius * radius + rays._BOUND_SLACK_NOISE * dot3(offset, offset)
    return (dot3(cross, cross) <= squared * dd) & (dot3(offset, directions) + torch.sqrt(squared * dd) >= 0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("distance", [0.001, 1.0, 100.0, 100000.0])
@pytest.mark.parametrize("strided", [False, True])
def test_tangent_masks_match_previous_bound(dtype, distance, strided):
    generator = torch.Generator().manual_seed(1209)
    axis = torch.randn(4, 1, 3, generator=generator, dtype=dtype)
    axis /= axis.norm(dim=-1, keepdim=True)
    side = torch.randn(4, 1, 3, generator=generator, dtype=dtype)
    side -= (side * axis).sum(-1, keepdim=True) * axis
    side /= side.norm(dim=-1, keepdim=True)
    pos = axis * distance
    origins = torch.zeros(4, 1026, 3, dtype=dtype)
    rad = torch.full((4, 1), distance * 0.07, dtype=dtype)
    padded = rad * rays._BOUND_SLACK_REL + rays._BOUND_SLACK_ABS
    effective = torch.sqrt(padded * padded + rays._BOUND_SLACK_NOISE * dot3(pos, pos))
    # Rays on both sides of the inflated bound, only a few ulps from tangency.
    perturb = 1 + torch.linspace(-32, 32, 1026, dtype=dtype) * torch.finfo(dtype).eps
    sine = (effective / distance) * perturb
    directions = axis * torch.sqrt(1 - sine * sine).unsqueeze(-1) + side * sine.unsqueeze(-1)
    directions[:, ::19] = 0  # Degenerate directions must retain legacy behavior.
    directions[:, 1::19] *= -1  # Include bound spheres behind the ray.
    if strided:
        origins, directions = origins[:, ::2], directions[:, ::2]
    dd = dot3(directions, directions)
    expected = previous_bound(pos, rad, origins, directions, dd)
    actual = rays._rays_near(pos, rad, origins, directions, dd)
    assert bool(expected.any()) and bool((~expected).any())
    assert torch.equal(actual, expected)
