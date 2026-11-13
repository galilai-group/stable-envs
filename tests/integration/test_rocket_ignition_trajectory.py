"""Flame color through ignition against the pinned source pixel env."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")

import stable_envs as se
from stable_envs.envs.rocket.legacy import FLAME_RGBA, IGNITION_RGBA
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.integration, pytest.mark.reference, pytest.mark.swm]

SIZE = 96
SEED = 17
COAST, BURN = 2, 10
ACTIONS = [np.zeros(7)] * COAST + [np.array([0.15, -0.1, 0.2, 1.0, 0.5, 0.1, -0.1])] * BURN


def source_flame(source):
    visuals = {v[1]: np.array(v[7]) for v in source.env.getVisualShapeData(source.env.drones[0].Id)}
    return visuals[9]


def test_flame_colour_and_cpu_frames_follow_the_rendering_source():
    from tests.helpers.reference.rocket_semantics import RocketLandingEnv as Oracle

    source = Oracle(render_mode="rgb_array", render_resolution=(SIZE, SIZE))
    ours = se.make("PyFlytRocketLanding", num_envs=1, size=SIZE, simulation_backend="pybullet")
    restored = se.make("PyFlytRocketLanding", num_envs=1, size=SIZE, simulation_backend="pybullet")
    try:
        source.reset(seed=SEED)
        ours.reset(seed=SEED)
        flames = []
        for action in ACTIONS:
            source.step(action.copy())
            state = ours.step(action[None].copy(), frames=None)["state"]
            np.testing.assert_array_equal(state["current", "wrapper", "state"][0].numpy(), source.state)
            flame = state["render", "rgba"][0, 5].numpy()
            np.testing.assert_array_equal(flame, source_flame(source))
            np.testing.assert_array_equal(ours.render().cpu().numpy()[0], source.render())
            flames.append(flame)
        np.testing.assert_allclose(flames[0], FLAME_RGBA, rtol=0, atol=1e-6)
        np.testing.assert_allclose(flames[-1], IGNITION_RGBA, rtol=0, atol=1e-6)
        saved = ours.get_state()
        restored.set_state(saved)
        assert_same_state(restored.get_state(), saved)
        assert torch.equal(restored.render(), ours.render())
    finally:
        restored.close()
        ours.close()
        source.close()
        if source.modified_pad_urdf:
            Path(source.modified_pad_urdf).unlink(missing_ok=True)
