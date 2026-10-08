"""Real replay States through the shared slice API; no renderer acceptance claim."""

import numpy as np
import pytest
import torch

from stable_envs.env import Env
from stable_envs.envs.pusht.simulation.history import stack_states
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter
from stable_envs.spaces import Dict


class ReplayStepHarness(Env):
    action_dim = 2
    _stack_states = staticmethod(stack_states)

    def __init__(self, discrete, device="cpu"):
        super().__init__(num_envs=2, device=device, seed=17)
        self.action_dtype = torch.int64 if discrete else torch.float64
        self.adapter = PushTReplayAdapter(2, device=device, seed=17, discrete=discrete, config={"resolution": 32})

    def make_variation_space(self):
        return Dict({})

    def get_state(self):
        return self.adapter.get_state()

    def _step(self, actions, *, active=None):
        return self.adapter.step(actions, active=active)

    def _close(self):
        self.adapter.close()


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize(
    "selection", [None, slice(None), slice(-1, None), slice(None, None, -1), slice(None, None, 2), slice(9, 10)]
)
def test_real_history_slice_roundtrip(discrete, device, sequence, selection):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA State storage requires Oscar")
    with ReplayStepHarness(discrete, device) as env:
        dtype = np.int64 if discrete else np.float64
        action = np.array([4, 3] if discrete else [0.2, -0.3], dtype=dtype)
        for lane, prefix in zip(env.adapter.envs, (3, 1), strict=True):
            for _ in range(prefix):
                lane.step(action)
        initial = env.get_state()
        count = 4 if sequence else 1
        actions = np.broadcast_to(action, (2, count, 2)).copy()
        expected = [env.adapter.step(actions[:, t])["state"] for t in range(count)]
        env.adapter.set_state(initial)
        output = env.step(actions if sequence else actions[:, 0], states=selection, frames=None)
        assert (output["state"] == expected[-1]).all()
        assert output["valid"].all()
        if selection is None:
            assert "states" not in output.keys()
            return
        indices = list(range(count)[selection])
        selected = output["states"]
        assert selected.batch_size == torch.Size([2, len(indices)])
        for index, time in enumerate(indices):
            state = selected[:, index]
            # Every selected State is independently restorable, even with
            # padding to a longer selected sibling's history.
            env.adapter.set_state(state)
            assert (env.get_state() == expected[time]).all()
            lengths = state["history", "lengths"]
            valid = state["history", "valid"]
            assert torch.equal(valid, torch.arange(valid.shape[1], device=device)[None] < lengths[:, None])
            assert not state["history", "actions"][~valid].any()
        frozen = output.clone()
        env.adapter.step(actions[:, 0])
        assert (frozen == output).all()


def test_collation_owns_storage_and_does_not_pad_inputs_in_place():
    with ReplayStepHarness(False) as env:
        short = env.get_state()
        long = env.adapter.step(np.zeros((2, 2), np.float64))["state"]
        result = stack_states([long, short])
        assert short["history", "actions"].shape == (2, 0, 2)
        assert result["history", "actions"].shape == (2, 2, 1, 2)
        assert result["history", "lengths"].tolist() == [[1, 0], [1, 0]]
        saved = short.clone()
        result["current", "observation"].fill_(0)
        assert (saved == short).all()
