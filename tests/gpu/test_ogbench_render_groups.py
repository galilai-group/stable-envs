"""Cube size, camera angle, light intensity and Scene lock color are per-world inputs of one MJWarp context."""

import pytest
import torch

from tests.helpers import ogbench_golden as golden

pytestmark = [pytest.mark.gpu, pytest.mark.regression]

CASES = (("Cube", "mjwarp"), ("Scene", "mjwarp"), ("Cube", "mjwarp_fidelity"))


def mixed_batch(family, backend):
    state, _ = golden.load(golden.FIXTURES / f"{family}-{backend}.npz")
    state = state[:6].clone()
    state["appearance", "cube.size"][1] = 0.013
    state["appearance", "light.intensity"][2] = 0.35
    state["appearance", "camera.angle_delta"][3] = state["appearance", "camera.angle_delta"].new_tensor([[4.0, -3.0]])
    if family == "Scene":
        state["appearance", "lock_color"][4] = state["appearance", "lock_color"].new_tensor([[0.1, 0.8, 0.2]] * 2)
    state["appearance", "cube.size"][5] = 0.027
    state["appearance", "camera.angle_delta"][5] = state["appearance", "camera.angle_delta"].new_tensor([[-2.0, 1.0]])
    return state


@pytest.fixture(scope="module")
def cuda_env():
    if not torch.cuda.is_available():
        pytest.skip("MJWarp rendering requires CUDA")


@pytest.mark.parametrize("family,backend", CASES)
def test_mixed_configuration_batch_equals_rendering_each_row_alone(cuda_env, family, backend):
    from stable_envs.envs.ogbench import OGBCube, OGBScene
    from stable_envs.envs.ogbench import manipulation as core

    state = mixed_batch(family, backend)
    cls = OGBCube if family == "Cube" else OGBScene
    with cls(device="cuda", size=64, seed=17) as env:
        batch = env.render(state, backend=backend)
        alone = torch.cat([env.render(state[i : i + 1], backend=backend) for i in range(len(state))])
        chunked = env.render(state, backend=backend, chunk_size=4)
        defaults = state.clone()
        for name in ("cube.size", "light.intensity", "camera.angle_delta") + (
            ("lock_color",) if family == "Scene" else ()
        ):
            defaults["appearance", name][:] = state["appearance", name][0]
        unvaried = env.render(defaults, backend=backend)
    assert torch.equal(batch, alone)
    assert torch.equal(batch, chunked)
    assert torch.equal(batch[0], unvaried[0])
    varied_rows = [1, 2, 3, 5] + ([4] if family == "Scene" else [])
    for varied in varied_rows:
        assert not torch.equal(batch[varied], unvaried[varied]), (
            f"row {varied}'s configuration did not reach the frame"
        )
    base = f"stable-envs-{family}"
    tokens = {key[1] for key in core._CTX_CACHE.keys() if key[1] == base or key[1].startswith(base + "/")}
    assert tokens <= {base, *(f"{base}/fidelity/{part}" for part in ("glass", "opaque", "static"))}, sorted(tokens)
