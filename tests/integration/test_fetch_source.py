"""The Fetch tasks against gymnasium-robotics' own Fetch*-v4 envs at their registered defaults."""

import numpy as np
import pytest

gymnasium_robotics = pytest.importorskip("gymnasium_robotics")

import gymnasium as gym  # noqa: E402

import stable_envs as se  # noqa: E402
from tests.helpers.reference.fetch_compare import full_compare  # noqa: E402


@pytest.mark.parametrize("scene", ["Reach", "Push", "Slide", "PickAndPlace"])
def test_default_episode_matches_gymnasium_robotics(scene):
    gym.register_envs(gymnasium_robotics)
    seed, lanes = 31, 3
    refs = [gym.make(f"Fetch{scene}-v4") for _ in range(lanes)]
    try:
        with se.make(f"Fetch{scene}", num_envs=lanes, device="cpu") as env:
            state = env.reset(seed=seed)
            for lane, ref in enumerate(refs):
                ref.reset(seed=seed + lane)
            full_compare(state.cpu(), refs)
            for action in np.random.default_rng(seed).uniform(-1, 1, (10, lanes, 4)).astype(np.float32):
                state = env.step(action, frames=None)["state"]
                for ref, a in zip(refs, action, strict=True):
                    ref.step(a)
                full_compare(state.cpu(), refs)
    finally:
        for ref in refs:
            ref.close()
