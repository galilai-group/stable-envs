"""Frames from Craftax ``step`` equal frames rendered from the returned States."""

from __future__ import annotations

import pytest
import torch

import stable_envs as se

pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, actions  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", NAMES)
def test_step_frames_equal_frames_rendered_from_the_states(name):
    with se.make(name, num_envs=3, device="cpu") as env:
        env.reset(seed=7)
        sequence = torch.as_tensor(actions(env, 5, 7)).permute(1, 0, 2)
        out = env.step(sequence, states=slice(None), frames=slice(None))
        for t in range(sequence.shape[1]):
            assert torch.equal(out["frames"][:, t], env.render(out["states"][:, t])), t
