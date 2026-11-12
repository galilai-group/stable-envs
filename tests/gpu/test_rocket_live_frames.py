"""CUDA Rocket step frames drawn from the live lanes equal ``render`` of the step States."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.regression]

if not torch.cuda.is_available():
    pytest.skip("Rocket CUDA frames require a CUDA device", allow_module_level=True)

pytest.importorskip("PyFlyt")
pytest.importorskip("nvdiffrast")

import stable_envs as se  # noqa: E402
from stable_envs.envs.rocket.simulation import execution  # noqa: E402
from stable_envs.envs.rocket.simulation.process import RocketProcessAdapter  # noqa: E402


@pytest.mark.parametrize("processes", [0, 2])
def test_live_frames_equal_state_frames(monkeypatch, processes):
    with monkeypatch.context() as patch:
        patch.setattr(execution, "process_count", lambda num_envs, cpus, **_: processes)
        env = se.make("PyFlytRocketLanding", num_envs=4, size=64, device="cuda", seed=3)
    try:
        assert isinstance(env._simulator, RocketProcessAdapter) == bool(processes)
        if processes:
            assert env._simulator.live_frames
        rng = np.random.default_rng(5)
        actions = torch.as_tensor(rng.uniform(env.action_low, env.action_high, (4, 12, 7)), device="cuda")
        env.reset(seed=7)
        for chunk in (actions[:, :1], actions[:, 1:12]):
            out = env.step(chunk, states=slice(None), frames=slice(None))
            assert out["frames"].device.type == "cuda"
            for t in range(chunk.shape[1]):
                torch.testing.assert_close(out["frames"][:, t], env.render(out["states"][:, t]), rtol=0, atol=0)
        torch.testing.assert_close(env._render_live(), env.render(env.get_state()), rtol=0, atol=0)
    finally:
        env.close()
