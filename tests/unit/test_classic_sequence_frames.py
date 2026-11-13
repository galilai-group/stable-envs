"""A classic-control step draws the frames of an action sequence in one call; each is the render of its State."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygame")
torch = pytest.importorskip("torch")

import stable_envs as snv

pytestmark = pytest.mark.unit

ENVS = ["CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"]


@pytest.mark.parametrize("per_lane", [False, True], ids=["shared", "per-lane"])
@pytest.mark.parametrize("lanes", [1, 3, 9])
@pytest.mark.parametrize("name", ENVS)
def test_sequence_frames_are_the_renders_of_the_states(name, lanes, per_lane):
    with snv.make(name, num_envs=lanes, seed=5, device="cpu") as env:
        env.reset(seed=5)
        if per_lane:
            colour = torch.tensor([[255, 255, 255], [10, 200, 30], [0, 0, 90]] * 3, dtype=torch.float32)[:lanes]
            env.set_variation({"background.color": colour})
        rng = np.random.default_rng(1)
        actions = torch.stack([env.action_spec.sample(lanes, rng=rng) for _ in range(12)], 1)
        out = env.step(actions, states=slice(None), frames=slice(None))
        assert out["frames"].shape == (lanes, 12, *env.frame_spec.shape)
        want = torch.stack([env.render(out["states"][:, t]) for t in range(12)], 1)
        assert torch.equal(out["frames"], want)
        every_other = env.step(actions[:, :5], states=slice(None), frames=slice(0, None, 2))
        want = torch.stack([env.render(every_other["states"][:, t]) for t in (0, 2, 4)], 1)
        assert torch.equal(every_other["frames"], want)
