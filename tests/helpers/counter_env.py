"""A three-lane counting Env, plain and pipelined, for exercising public stepping."""

import torch
from tensordict import TensorDict

import stable_envs as snv
from stable_envs.spaces import Dict


class Counter(snv.Env):
    action_dim = 1
    action_dtype = torch.int64
    state_dim = 1
    frame_shape = (2, 2)

    def __init__(self, device="cpu"):
        super().__init__(num_envs=3, device=device)
        self.position = torch.zeros(3, 1, dtype=torch.int64, device=self.device)
        self.limit = torch.tensor([1, 3, 20], device=self.device)
        self.calls = torch.zeros(3, dtype=torch.int64, device=self.device)
        self.render_calls = 0

    def make_variation_space(self):
        return Dict({})

    def get_state(self):
        return TensorDict({"position": self.position.clone()}, [3], device=self.device)

    def _step(self, actions, *, active=None):
        active = torch.ones(3, dtype=torch.bool, device=self.device) if active is None else active
        self.position[active] += torch.as_tensor(actions, device=self.device)[active]
        self.calls += active
        return self._step_result(
            self.get_state(), reward=self.position[:, 0].float(), terminated=self.position[:, 0] >= self.limit
        )

    def render(self, state=None):
        self.render_calls += 1
        state = self.get_state() if state is None else state
        return state["position"].to(torch.uint8).reshape(3, 1, 1, 1).expand(3, 2, 2, 3).clone()


class PipelinedCounter(Counter):
    """Complete immediately to stress snapshot ownership under maximum lookahead."""

    pending = None
    aborted = False
    overlaps = 0

    def _step_pipeline(self):
        return self

    def submit_step(self, actions, *, active=None, capture_state=True):
        assert self.pending is None
        self.pending = self._step(actions, active=active)
        if not capture_state:
            self.pending = self.pending.exclude("state")

    def receive_step(self):
        result, self.pending = self.pending, None
        assert result is not None
        return result

    def abort_step(self):
        self.aborted = True
        self.pending = None

    def render(self, state=None):
        self.overlaps += self.pending is not None
        return super().render(state)
