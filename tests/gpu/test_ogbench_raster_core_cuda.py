"""Cube/Scene ``backend="raster"`` on CUDA: the CUDA backend of the shared MuJoCo rasterizer.

It is compared with MuJoCo's NVIDIA OpenGL frames (``backend="opengl"``) and with the CPU backend, which reads the
same packed scene and per-frame inputs.
"""

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

pytestmark = [pytest.mark.gpu]

NAMES = [pytest.param(n, marks=pytest.mark.family("ogbench")) for n in ("OGBCube", "OGBScene")]
#: Largest mean frame MSE (0-255 scale) against backend="opengl" over the checked frames.
MEAN_MSE = {"OGBCube": 5.0, "OGBScene": 12.0}
#: Largest MSE of one frame.
FRAME_MSE = {"OGBCube": 10.0, "OGBScene": 20.0}
#: Largest mean frame MSE between the CUDA and CPU backends (sample positions, centroid rule, float32).
CPU_MSE = 6.0


@pytest.fixture(scope="module")
def cuda_raster():
    if not torch.cuda.is_available():
        pytest.skip("the CUDA backend draws on CUDA")
    pytest.importorskip("nvdiffrast.torch")
    pytest.importorskip("triton")
    pytest.importorskip("numba")


def _states(env, n, varied):
    rows = torch.as_tensor(C.sample_render_rows(env, n, rng=np.random.default_rng(11)), device="cuda")
    appearance = env._stored_appearance(n, env.size, 1, "cuda")
    state = env._rows_to_render_state(rows, None, env.size, 1, "cuda", appearance)
    if varied:
        look = state["appearance"]
        look["agent.color"][1] = look["agent.color"].new_tensor([0.2, 0.7, 0.3])
        look["cube.color"][2] = look["cube.color"].new_tensor([[0.1, 0.4, 0.9]])
        look["floor.color"][3] = look["floor.color"].new_tensor([[0.3, 0.2, 0.1], [0.5, 0.45, 0.4]])
        look["light.intensity"][4] = 0.35
        look["camera.angle_delta"][5] = look["camera.angle_delta"].new_tensor([[3.0, -2.0]])
        look["cube.size"][6] = 0.028
        if env.CELL == "Scene":
            look["lock_color"][7] = look["lock_color"].new_tensor([[0.1, 0.8, 0.2], [0.9, 0.9, 0.1]])
    return state


def _mse(a, b):
    return ((a.double() - b.double()) ** 2).reshape(len(a), -1).mean(1)


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_frames_follow_opengl_and_the_cpu_backend(cuda_raster, name, varied):
    with se.make(name, num_envs=1, device="cuda") as env:
        state = _states(env, 8, varied)
        expected = env.render(state, backend="opengl")
        actual = env.render(state, backend="raster")
    assert actual.device.type == "cuda" and actual.dtype == torch.uint8 and actual.shape == expected.shape
    mse = _mse(actual, expected)
    assert float(mse.mean()) < MEAN_MSE[name]
    assert float(mse.max()) < FRAME_MSE[name]
    with se.make(name, num_envs=1, device="cpu") as env:
        cpu = env.render(state.cpu(), backend="raster")
    assert float(_mse(actual.cpu(), cpu).mean()) < CPU_MSE


@pytest.mark.parametrize("name", NAMES)
def test_batch_chunks_and_rows_alone_draw_the_same_frames(cuda_raster, name):
    with se.make(name, num_envs=1, device="cuda") as env:
        state = _states(env, 8, True)
        batch = env.render(state, backend="raster")
        assert torch.equal(env.render(state, backend="raster", chunk_size=3), batch)
        for row in (0, 3, 6):
            assert torch.equal(env.render(state[row : row + 1], backend="raster"), batch[row : row + 1])
        assert torch.equal(env.render(state, backend="raster"), batch)


@pytest.mark.parametrize("name", NAMES)
def test_size_and_supersample_follow_opengl(cuda_raster, name):
    with se.make(name, num_envs=1, device="cuda") as env:
        state = _states(env, 2, False)
        for size, ss in ((64, 1), (63, 1), (96, 2)):
            expected = env.render(state, backend="opengl", size=size, supersample=ss)
            actual = env.render(state, backend="raster", size=size, supersample=ss)
            assert actual.shape == (2, size, size, 3)
            assert float(_mse(actual, expected).mean()) < 2 * MEAN_MSE[name]


@pytest.mark.parametrize("name", NAMES)
def test_live_lanes_draw_like_their_render_state(cuda_raster, name):
    with se.make(name, num_envs=4, device="cuda", backend="raster") as env:
        env.reset(seed=3)
        live = env.render()
        assert torch.equal(live, env.render(env.to_render_state()))
