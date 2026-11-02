"""Public DMC sequences only export Warp State where captures need it."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("dm_control")

from stable_envs.env import Env
from stable_envs.envs.dmcontrol._delegated import DelegatedSemantics


class SpyAdapter:
    def __init__(self):
        self.position = torch.zeros((3, 1), dtype=torch.float64)
        self.calls = []
        self.exports = 0

    def get_state(self):
        self.exports += 1
        return TensorDict({"position": self.position.clone()}, [3], device="cpu")

    def step(self, actions, **kwargs):
        self.calls.append(kwargs)
        self.position.add_(actions)
        out = TensorDict(
            {
                "reward": self.position[:, 0].clone(),
                "terminated": torch.zeros(3, dtype=torch.bool),
                "truncated": torch.zeros(3, dtype=torch.bool),
            },
            [3],
            device="cpu",
        )
        if kwargs.get("capture_state", True):
            out["state"] = self.get_state()
        return out


class SequenceHarness(DelegatedSemantics):
    """Use public sequence routing with the real DMC dispatch and a counted adapter."""

    step = Env.step
    num_envs = 3
    device = torch.device("cpu")
    has_dynamics = True
    action_spec = SimpleNamespace(shape=(1,), dtype=torch.float64)
    frame_spec = SimpleNamespace(shape=(1, 1, 3), dtype=torch.uint8)

    def __init__(self, backend):
        self.simulation_backend = backend
        self.execution_mode = {"native": "DELEGATED_CPU", "mjx": "JAX_MJX", "mjwarp": "MUJOCO_WARP"}[backend]
        self._simulator = SpyAdapter()

    def _check_open(self, method):
        pass

    def render(self, state):
        return state["position"].to(torch.uint8).reshape(3, 1, 1, 1).expand(3, 1, 1, 3).clone()


def _actions():
    return torch.arange(1, 5, dtype=torch.float64).reshape(1, 4, 1).expand(3, 4, 1)


@pytest.mark.parametrize(
    "states,frames,exports,state_values,frame_values",
    [
        (None, None, 1, None, None),
        (slice(None, None, 2), None, 3, [1, 6], None),
        (slice(None, None, -1), None, 4, [10, 6, 3, 1], None),
        (None, slice(None, None, 2), 3, None, [1, 6]),
        (slice(None, None, 2), slice(None, None, 2), 3, [1, 6], [1, 6]),
        (slice(0, 0), slice(0, 0), 1, [], []),
    ],
)
def test_warp_exports_only_requested_snapshots(states, frames, exports, state_values, frame_values):
    env = SequenceHarness("mjwarp")
    result = env.step(_actions(), states=states, frames=frames)

    assert env._simulator.calls == [{"capture_state": False}] * 4
    assert env._simulator.exports == exports
    assert torch.equal(result["state", "position"], torch.full((3, 1), 10.0, dtype=torch.float64))
    assert torch.equal(result["reward"], torch.tensor([[1.0, 3.0, 6.0, 10.0]]).expand(3, 4))
    assert bool(result["valid"].all())
    if state_values is None:
        assert "states" not in result
    else:
        expected = torch.tensor(state_values, dtype=torch.float64).reshape(1, -1, 1).expand(3, -1, 1)
        assert torch.equal(result["states", "position"], expected)
        assert bool(result["state_valid"].all())
    if frame_values is None:
        assert "frames" not in result
    else:
        expected = torch.tensor(frame_values, dtype=torch.uint8).reshape(1, -1, 1, 1, 1).expand(3, -1, 1, 1, 3)
        assert torch.equal(result["frames"], expected)
        assert bool(result["frame_valid"].all())

    # Later buffer reuse must not change any retained final, selected, or task output.
    frozen = result.clone()
    env.step(_actions(), states=None, frames=None)
    for key, value in frozen.items(True, True):
        assert torch.equal(result[key], value), key


@pytest.mark.parametrize("backend", ["native", "mjx"])
def test_existing_backends_keep_adapter_step_signature(backend):
    env = SequenceHarness(backend)
    result = env.step(_actions(), states=None, frames=None)
    assert env._simulator.calls == [{}] * 4
    assert env._simulator.exports == 4
    assert torch.equal(result["state", "position"], torch.full((3, 1), 10.0, dtype=torch.float64))
