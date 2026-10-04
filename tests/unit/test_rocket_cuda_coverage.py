"""Regression for thin fin coverage and near-plane ground triangles."""

import numpy as np
import pytest
import torch

from stable_envs import make

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rocket coverage requires CUDA")


@pytest.mark.parametrize("scenario", ["thin_fin", "near_contact"])
def test_reached_source_pixels_and_batch_partition(scenario):
    pytest.importorskip("PyFlyt")
    pytest.importorskip("nvdiffrast.torch")
    torch.set_num_threads(1)
    config = dict(render_mode="rgb_array", render_resolution=(96, 96))
    with make("PyFlytRocketLanding", num_envs=2, device="cuda", size=96, simulator_config=config) as env:
        env.reset(seed=17)
        actions = np.zeros((2, 7), np.float32)
        count = 302
        if scenario == "thin_fin":
            actions[:] = [0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1]
            count = 3
        for _ in range(count):
            env.step(actions, states=None, frames=None)
        projected = env.to_render_state()
        source = np.stack([engine.env.render() for engine in env._simulator.envs])
        ours = env.render(projected)
        mse = np.mean(((ours.cpu().numpy().astype(np.float64) - source) / 255) ** 2, axis=(1, 2, 3))
        assert np.all(mse <= 0.001), mse
        assert torch.equal(ours, env.render(projected, chunk_size=1))
        for lane in range(2):
            assert torch.equal(ours[lane : lane + 1], env.render(projected[lane : lane + 1]))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_noncurrent_cuda_device_and_context_restoration():
    pytest.importorskip("PyFlyt")
    pytest.importorskip("nvdiffrast.torch")
    torch.set_num_threads(1)
    config = dict(render_mode="rgb_array", render_resolution=(96, 96))
    with torch.cuda.device(0):
        with make("PyFlytRocketLanding", device="cuda:1", size=96, simulator_config=config) as env:
            env.reset(seed=17)
            for _ in range(3):
                env.step(np.array([[0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1]], np.float32), states=None, frames=None)
            source = env._simulator.envs[0].env.render()
            ours = env.render()
            assert ours.device == torch.device("cuda:1")
            assert torch.cuda.current_device() == 0
            mse = np.mean(((ours[0].cpu().numpy().astype(np.float64) - source) / 255) ** 2)
            assert mse <= 0.001, mse
