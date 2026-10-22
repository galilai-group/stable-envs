"""Viewport bounds are geometry contracts, independent of G34 and CUDA."""

import numpy as np
import pytest

from stable_envs.envs.object_permanence.rendering.viewports import CropCamera, PixelCenterRenderer, crop_spheres
from stable_envs.render.nvdiff import Camera


def camera():
    return Camera(np.zeros(3), np.eye(3), 60.0, 0.1, 100.0)


def test_multiple_off_axis_spheres_fit_and_crop_preserves_pixel_coordinates():
    cam = camera()
    centers = np.array([[[1.1, 0.3, -4.0], [1.5, -0.1, -4.5]], [[-1.3, 0.4, -4.2], [-1.1, 0.1, -4.8]]])
    radii = np.array([0.14, 0.2])
    h, w, k = 120, 200, 36
    proj, xs, ys = crop_spheres(cam, centers, radii, image_size=(h, w), window_size=k)
    _, full = cam.matrices(w / h)
    # Independently project a dense surface sample, rather than duplicating the
    # tangent-bound formula. Check every object, translated away from image center.
    rng = np.random.default_rng(71)
    directions = rng.normal(size=(8192, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    for b in range(2):
        points = (centers[b, :, None] + radii[:, None, None] * directions).reshape(-1, 3)
        homogeneous = np.column_stack([points, np.ones(len(points))])
        clip = homogeneous @ full.T
        pixels = clip[:, :2] / clip[:, 3:4] * [w / 2, -h / 2] + [w / 2, h / 2]
        local = homogeneous @ proj[b].T
        local_pixels = local[:, :2] / local[:, 3:4] * [k / 2, -k / 2] + [k / 2, k / 2]
        np.testing.assert_allclose(local_pixels, pixels - [xs[b], ys[b]], atol=1e-12)
        assert (local_pixels >= 0.5).all() and (local_pixels <= k - 0.5).all()
        single, sx, sy = crop_spheres(cam, centers[b : b + 1], radii, image_size=(h, w), window_size=k)
        assert np.array_equal(single[0], proj[b]) and sx[0] == xs[b] and sy[0] == ys[b]


@pytest.mark.parametrize(
    "center,radius,window",
    [([0, 0, -0.15], 0.1, 24), ([0, 0, -99.95], 0.1, 24), ([4, 0, -2], 0.1, 24), ([0, 0, -2], 0.8, 8)],
)
def test_invalid_bounds_fail_instead_of_clipping(center, radius, window):
    with pytest.raises(ValueError, match="bounds"):
        crop_spheres(camera(), [[center]], [radius], image_size=(120, 200), window_size=window)


def test_prepared_camera_returns_borrowed_matrices_without_numpy_conversion():
    import torch

    cam = CropCamera(np.zeros(3), np.eye(3), 60, 0.1, 100)
    cam.view = torch.eye(4)[None]
    cam.projection = torch.eye(4)[None].expand(3, -1, -1)
    view, projection = cam.matrices()
    assert view is cam.view and projection is cam.projection


@pytest.mark.parametrize("scale", [0, 2, -1, True])
def test_sampling_rejects_invalid_scale_before_opening_a_cuda_context(scale):
    with pytest.raises(ValueError, match="odd integer"):
        PixelCenterRenderer(coverage_scale=scale, mipmap=False)
