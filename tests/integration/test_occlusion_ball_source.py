"""The adapter calls MuJoCo correctly and reproduces independent Blender poses."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

import stable_envs as se
from tests.helpers.reference.occlusion_ball import NativeReference, load_oracle
from tests.helpers.reference.semantics import assert_same_state


def test_adapter_matches_direct_mujoco_from_the_same_checkpoint():
    """Exact engine continuation; oracle serialization/scheduling is independent."""
    with se.MuJoCoOcclusionBall(num_envs=3) as env:
        reference = NativeReference(3)
        reference.set_state(env.get_state())
        # The authored schedule is independently transcribed in the reference.
        # Blender-vs-float64 rounding is tested against frozen poses below.
        state = env.get_state()
        for frame_index in range(1, 120):
            actual = env.step(np.zeros((3, 1)), states=None, frames=None)
            expected = reference.step()
            assert_same_state(actual["state"], expected)
            assert not bool(actual["terminated"].any())
            assert bool(actual["truncated"].all()) == (frame_index == 119)
            assert bool((actual["reward"] == 0).all())
            state = actual["state"]
        reference.set_state(state)
        assert_same_state(reference.get_state(), state)


def test_nominal_rollout_matches_frozen_blender_trajectory():
    """Original-source gate: all 120 source frames, including hidden motion."""
    _, oracle = load_oracle()
    with se.MuJoCoOcclusionBall() as env:
        states = [env.reset(seed=0)]
        trajectory = env.step(np.zeros((1, 119, 1)), frames=None)["states"]
        states.extend(trajectory[:, t] for t in range(119))
        offset = env._simulator.pose_offset
        for i, state in enumerate(states):
            pose = state["physics", "integration"][0, offset:].numpy()
            np.testing.assert_allclose(pose[:3], oracle["positions"][i], rtol=0, atol=3e-7)
            # q and -q encode the same orientation. Blender's matrix extraction
            # changes sign across its Euler branch; compare rotation, not sign.
            expected = oracle["quaternions"][i]
            actual = pose[3:]
            if np.dot(actual, expected) < 0:
                actual = -actual
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            assert state["lifecycle", "elapsed_steps"].item() == i
