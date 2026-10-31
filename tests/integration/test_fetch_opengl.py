import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.fetch import opengl as fetch_opengl

pytestmark = pytest.mark.egl


@pytest.fixture(scope="module")
def gl_device() -> str:
    if os.environ.get("MUJOCO_GL") not in ("egl", "glfw"):
        pytest.skip("Requires an explicit EGL GPU or GLFW software-GL test environment")
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("cv2")
    return "cuda" if os.environ["MUJOCO_GL"] == "egl" else "cpu"


class CpuReadback:
    def __init__(self, renderer, size: int, images: np.ndarray, start: int, total: int):
        self.renderer, self.size, self.images = renderer, size, images

    def render(self, index: int) -> None:
        import cv2

        self.images[index] = cv2.resize(self.renderer.render(), (self.size, self.size))


@pytest.mark.parametrize("scene", ["Reach", "Push", "Slide", "PickAndPlace"])
def test_fetch_resize_preserves_pixels_and_output_ownership(scene: str, gl_device: str, record_property) -> None:
    with se.make(f"Fetch{scene}", num_envs=2, device=gl_device, backend="opengl", size=64) as env:
        env.reset(seed=17)
        env.step(torch.zeros((2, 3, 4), device=gl_device), frames=None, states=None)
        state = env.get_state()
        for index, size in enumerate((1, 63, 64, 96, 480, 513, 64)):
            with patch.object(fetch_opengl, "_Readback", CpuReadback):
                expected = env.render(state, size=size)
            actual = env.render(state, size=size)
            error = actual.to(torch.int16) - expected.to(torch.int16)
            maximum = error.abs().max().item()
            mse = (error.double() / 255).square().mean().item()
            record_property(f"resize_{index}", json.dumps(dict(size=size, max_channel=maximum, mse=mse)))
            # Hardware interpolation and OpenCV use different coefficient/rounding precision.
            assert maximum <= 2
            assert mse <= (1 / 255) ** 2
            out = torch.empty_like(actual)
            returned = env.render(state, size=size, out=out)
            assert returned.data_ptr() == out.data_ptr()
            torch.testing.assert_close(actual, returned, rtol=0, atol=0)
            saved = actual.clone()
            env.render(state, size=32)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_fetch_resize_reads_only_output_pixels_and_restores_context(gl_device: str, monkeypatch) -> None:
    from OpenGL import GL

    reads = []
    read_pixels = GL.glReadPixels

    def record_read(x, y, width, height, *args):
        reads.append((width, height))
        return read_pixels(x, y, width, height, *args)

    with (
        se.make("FetchPush", num_envs=3, device=gl_device, backend="opengl", size=64) as push,
        se.make("FetchReach", num_envs=1, device=gl_device, backend="opengl", size=64) as reach,
    ):
        push.reset(seed=17)
        reach.reset(seed=42)
        expected = push.render()
        reach.render()
        monkeypatch.setattr(GL, "glReadPixels", record_read)
        get_integer = GL.glGetIntegerv
        monkeypatch.setattr(
            GL, "glGetIntegerv", lambda key: 128 if key == GL.GL_MAX_RENDERBUFFER_SIZE else get_integer(key)
        )
        torch.testing.assert_close(push.render(), expected, rtol=0, atol=0)
        assert reads == [(64, 128), (64, 64)]
        renderer = fetch_opengl._CACHE[("Push", 480)][2]
        assert GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING) == renderer._mjr_context.offFBO
        assert GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING) == renderer._mjr_context.offFBO
