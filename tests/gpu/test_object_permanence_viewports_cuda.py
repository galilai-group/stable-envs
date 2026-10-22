"""The family viewport helpers also work on a non-G34 multi-object scene."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("viewports need CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence.rendering.viewports import CropCamera, PixelCenterRenderer, crop_spheres
from stable_envs.render import nvdiff as ND


@pytest.mark.parametrize("coverage", ["scaled", "refined"])
def test_two_moving_boxes_crop_matches_full_frame_and_preserves_object_ids(coverage):
    v, n, f = ND._box([0.1, 0.1, 0.1])
    vertices, normals = np.tile(v, (2, 1)), np.tile(n, (2, 1))
    faces = np.concatenate([f, f + len(v)])
    owners = np.repeat(np.arange(2), len(v))
    poses = np.array([[[-0.2, 0.0, -3.5], [0.2, 0.0, -3.8]], [[-0.15, 0.1, -3.6], [0.25, 0.1, -3.9]]])
    camera = ND.Camera(np.zeros(3), np.eye(3), 60, 0.1, 100)
    projection, xs, ys = crop_spheres(camera, poses, np.full(2, np.sqrt(0.03)), image_size=(90, 160), window_size=24)
    crop = CropCamera(camera.pos, camera.mat, camera.fovy, camera.znear, camera.zfar)
    crop.projection = projection
    with ND.Scene.from_triangles(vertices, normals, faces, owners, [[1, 0, 0, 1], [0, 1, 0, 1]]) as scene:
        with PixelCenterRenderer(
            scene=scene,
            res=(90, 160),
            shading="tiny",
            textures=False,
            mipmap=False,
            skybox=False,
            coverage_mode=coverage,
        ) as renderer:
            positions = torch.as_tensor(poses, dtype=torch.float32, device="cuda")
            rotations = torch.eye(3, device="cuda").expand(2, 2, 3, 3)
            for mode in ("rgb", "depth", "ids"):
                renderer.height, renderer.width = 90, 160
                full = renderer.render(positions, rotations, camera, mode=mode)
                renderer.height = renderer.width = 24
                windows = renderer.render(positions, rotations, crop, mode=mode)
                for b in range(2):
                    expected = full[b, ys[b] : ys[b] + 24, xs[b] : xs[b] + 24]
                    if mode == "depth":
                        torch.testing.assert_close(windows[b], expected, rtol=0, atol=2e-6)
                    else:
                        assert torch.equal(windows[b], expected)
                if mode == "ids":
                    assert windows.unique().tolist() == [0, 1, 2]
