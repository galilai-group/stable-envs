import torch

from stable_envs.render.texture_mip import mip_pyramid, sample_plane_filtered, sample_skybox_filtered


def test_minified_checker_averages_without_aliasing():
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    texture = ((x + y) % 2).float()[..., None].expand(-1, -1, 3)
    uv = torch.tensor([0.0, 0.07, 0.5, 0.9])
    zero = torch.zeros_like(uv)
    footprint = torch.full_like(uv, 0.25)
    result = sample_plane_filtered(mip_pyramid(texture), uv, uv, footprint, zero, zero, footprint)
    torch.testing.assert_close(result, torch.full_like(result, 0.5))


def test_magnification_preserves_texel_and_repeat_seam():
    texture = torch.tensor([[[0.0], [1.0]]])
    u = torch.tensor([0.25, 0.75, 0.0, 1.0, -0.25, 1.25])
    zero = torch.zeros_like(u)
    result = sample_plane_filtered(mip_pyramid(texture), u, zero, zero, zero, zero, zero)
    torch.testing.assert_close(result[:, 0], torch.tensor([0.0, 1.0, 0.5, 0.5, 1.0, 0.0]))


def test_trilinear_blends_neighboring_levels():
    texture = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]])
    u = torch.tensor([0.25])
    zero = torch.zeros_like(u)
    # A sqrt(2)-texel footprint lies halfway between base and first mip.
    derivative = torch.full_like(u, 2**-0.5)
    result = sample_plane_filtered(mip_pyramid(texture), u, u, derivative, zero, zero, derivative)
    torch.testing.assert_close(result, torch.tensor([[0.25]]))


def test_sky_faces_keep_their_colors_at_large_footprints():
    colors = torch.eye(6)
    sky = colors[:, None, None, :].expand(6, 32, 32, 6).reshape(192, 32, 6)
    # MuJoCo's world axes map to cube +X,-X,+Y,-Y,+Z,-Z.
    directions = torch.tensor(
        [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    result = sample_skybox_filtered(sky, 32, directions)
    torch.testing.assert_close(result.reshape(6, 6), colors)


def test_sky_minification_is_independent_of_ray_length():
    torch.manual_seed(9)
    sky = torch.rand(96, 16, 3)
    directions = torch.tensor([[[-0.3, -1.0, 0.2], [0.3, -1.0, 0.2]], [[-0.3, -1.0, -0.2], [0.3, -1.0, -0.2]]])
    expected = sample_skybox_filtered(sky, 16, directions)
    lengths = torch.tensor([[2.0, 5.0], [3.0, 7.0]])[..., None]
    torch.testing.assert_close(sample_skybox_filtered(sky, 16, directions * lengths), expected)


def test_single_pixel_sky_and_non_square_texture():
    sky = torch.ones(48, 8, 3)
    directions = torch.tensor([[[0.0, -1.0, 0.0]]])
    torch.testing.assert_close(sample_skybox_filtered(sky, 8, directions), torch.ones(1, 1, 3))
    levels = mip_pyramid(torch.ones(3, 7, 3))
    assert levels.sizes.tolist() == [[7, 3], [3, 1], [1, 1]]
    assert torch.equal(levels.texels, torch.ones_like(levels.texels))


def test_packed_mips_match_independent_grid_sample():
    import torch.nn.functional as functional

    from stable_envs.render.texture_mip import _trilinear

    generator = torch.Generator().manual_seed(17)
    texture = torch.rand((9, 13, 3), generator=generator)
    u, v = (torch.rand(51, generator=generator) * 4 - 2 for _ in range(2))
    pyramid = mip_pyramid(texture)
    levels = [texture.permute(2, 0, 1)[None]]
    while max(levels[-1].shape[-2:]) > 1:
        h, w = levels[-1].shape[-2:]
        levels.append(functional.interpolate(levels[-1], size=(max(1, h // 2), max(1, w // 2)), mode="area"))
    lod = torch.linspace(-1, len(levels) + 1, 51)
    low = lod.clamp(0, len(levels) - 1).floor().long()
    weight = lod.clamp(0, len(levels) - 1) - low
    for wrap in (False, True):
        sampled = []
        for level in levels:
            h, w = level.shape[-2:]
            x, y = u, v
            if wrap:
                level = functional.pad(level, (1, 1, 1, 1), mode="circular")
                x, y = ((u % 1) * w + 1) / (w + 2), ((v % 1) * h + 1) / (h + 2)
            grid = torch.stack((2 * x - 1, 2 * y - 1), -1)[None, None]
            sampled.append(functional.grid_sample(level, grid, padding_mode="border", align_corners=False)[0, :, 0].T)
        samples = torch.stack(sampled)
        index = torch.arange(len(u))
        expected = samples[low, index] * (1 - weight[:, None])
        expected += samples[(low + 1).clamp(max=len(levels) - 1), index] * weight[:, None]
        torch.testing.assert_close(_trilinear(pyramid, u, v, lod, wrap), expected, rtol=0, atol=2e-6)


def test_source_plane_grid_finite_extent_spacing_and_infinite_padding():
    import numpy as np

    from stable_envs.render.texture_mip import plane_grid

    np.testing.assert_array_equal(plane_grid(4, 1, 100, 1), [-4, -2, 0, 2, 4])
    grid = plane_grid(250, 0.1, 100, 1)
    assert len(grid) <= 201
    assert grid[0] == -250 and grid[-1] == 250
    assert np.diff(grid).max() > 2.5
    infinite = plane_grid(0, 0.1, 100, 2)
    assert len(infinite) == 201
    np.testing.assert_allclose(infinite, np.arange(-100, 101) * 1.06)
