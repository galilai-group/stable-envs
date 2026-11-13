"""Fetch ``backend="raster"`` on CUDA: the CUDA backend of the shared MuJoCo rasterizer.

It is compared with MuJoCo's NVIDIA OpenGL frames (``backend="opengl"``, the Gymnasium Robotics renderer) and with
the CPU backend, which reads the same packed scene and per-frame inputs.
"""

import numpy as np
import pytest
import torch

import stable_envs as se

pytestmark = [pytest.mark.gpu]

SCENES = [pytest.param(s, marks=pytest.mark.family("fetch")) for s in ("Reach", "Push", "Slide", "PickAndPlace")]
VISUAL = ["table.color", "background.color", "light.intensity", "camera.angle_delta", "rendering.transparent_arm"]
#: Largest mean and per-frame MSE (0-255 scale) against backend="opengl".
MEAN_MSE, FRAME_MSE = 2.0, 5.0
#: Largest mean frame MSE between the CUDA and CPU backends.
CPU_MSE = 3.0


@pytest.fixture(scope="module")
def cuda_raster():
    if not torch.cuda.is_available():
        pytest.skip("the CUDA backend draws on CUDA")
    pytest.importorskip("gymnasium_robotics")
    pytest.importorskip("nvdiffrast.torch")
    pytest.importorskip("triton")
    pytest.importorskip("numba")


def _mse(a, b):
    return ((a.double() - b.double()) ** 2).reshape(len(a), -1).mean(1)


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("scene", SCENES)
def test_frames_follow_opengl_and_the_cpu_backend(cuda_raster, scene, varied):
    with se.make(f"Fetch{scene}", num_envs=4, device="cuda") as env:
        options = {"variation": VISUAL + ([] if scene == "Reach" else ["object.color"])} if varied else None
        env.reset(seed=3, options=options)
        generator = torch.Generator(device="cuda").manual_seed(3)
        for _ in range(4):
            env.step(torch.rand(4, 4, generator=generator, device="cuda") * 2 - 1)
        rows = env._pack_render_state(env.get_state())
        expected = env._render(rows, backend="opengl")
        actual = env._render(rows, backend="raster")
        assert torch.equal(env._render(rows, backend="raster"), actual)
    assert actual.device.type == "cuda" and actual.shape == expected.shape
    mse = _mse(actual, expected)
    assert float(mse.mean()) < MEAN_MSE and float(mse.max()) < FRAME_MSE
    if not varied:
        with se.make(f"Fetch{scene}", num_envs=4, device="cpu") as env:
            cpu = env._render(rows.cpu(), backend="raster")
            for lane in range(4):
                alone = env._render(rows[lane : lane + 1].cpu(), backend="raster")
                assert torch.equal(alone, cpu[lane : lane + 1])
        assert float(_mse(actual.cpu(), cpu).mean()) < CPU_MSE


def test_transparent_arm_and_flat_sky(cuda_raster):
    with se.make("FetchPush", num_envs=2, device="cuda") as env:
        rows = env._pack_render_state(env.get_state())
        for variation in (
            {"rendering.transparent_arm": np.asarray([1], np.float32)},
            {"sky.color": np.asarray([0.3, 0.3, 0.6], np.float32)},
        ):
            expected = env._render(rows, variation=variation, backend="opengl")
            actual = env._render(rows, variation=variation, backend="raster")
            assert float(_mse(actual, expected).mean()) < MEAN_MSE, variation
