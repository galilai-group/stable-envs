"""backend="nvdiff" draws Cube/Scene frames close to MuJoCo's NVIDIA OpenGL frames, for default and varied looks."""

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


@pytest.fixture(scope="module")
def nvdiff():
    if not torch.cuda.is_available():
        pytest.skip("backend='nvdiff' draws on CUDA")
    pytest.importorskip("nvdiffrast.torch")
    pytest.importorskip("triton")


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
        look["camera.angle_delta"][5] = look["camera.angle_delta"].new_tensor([[4.0, -3.0]])
        look["cube.size"][6] = 0.028
        if env.CELL == "Scene":
            look["lock_color"][7] = look["lock_color"].new_tensor([[0.1, 0.8, 0.2], [0.9, 0.9, 0.1]])
    return state


@pytest.mark.parametrize("varied", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_frames_follow_opengl_frames(nvdiff, name, varied):
    with se.make(name, num_envs=1, device="cuda") as env:
        state = _states(env, 8, varied)
        expected = env.render(state, backend="opengl").double()
        actual = env.render(state, backend="nvdiff").double()
    assert actual.shape == expected.shape and actual.dtype == torch.float64
    mse = ((actual - expected) ** 2).reshape(len(actual), -1).mean(1)
    assert float(mse.mean()) < MEAN_MSE[name]
    assert float(mse.max()) < FRAME_MSE[name]


@pytest.mark.parametrize("name", NAMES)
def test_batch_chunks_and_rows_alone_draw_the_same_frames(nvdiff, name):
    with se.make(name, num_envs=1, device="cuda") as env:
        state = _states(env, 8, True)
        batch = env.render(state, backend="nvdiff")
        assert torch.equal(env.render(state, backend="nvdiff", chunk_size=3), batch)
        for row in (0, 3, 6):
            assert torch.equal(env.render(state[row : row + 1], backend="nvdiff"), batch[row : row + 1])
        assert torch.equal(env.render(state, backend="nvdiff"), batch)


@pytest.mark.parametrize("name", NAMES)
def test_size_and_supersample_follow_opengl(nvdiff, name):
    with se.make(name, num_envs=1, device="cuda") as env:
        state = _states(env, 2, False)
        for size, ss in ((64, 1), (96, 2)):
            expected = env.render(state, backend="opengl", size=size, supersample=ss).double()
            actual = env.render(state, backend="nvdiff", size=size, supersample=ss).double()
            assert actual.shape == (2, size, size, 3)
            assert float(((actual - expected) ** 2).mean()) < 2 * MEAN_MSE[name]


@pytest.mark.parametrize("name", NAMES)
def test_cpu_device_is_rejected(name):
    with se.make(name, num_envs=1, device="cpu") as env:
        with pytest.raises(ValueError, match="nvdiff"):
            env.render(backend="nvdiff")
