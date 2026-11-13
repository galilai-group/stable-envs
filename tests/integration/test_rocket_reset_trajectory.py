"""Reset frame and engine fields after a varied reset against the pinned source pixel env."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")

import stable_envs as se

pytestmark = [pytest.mark.integration, pytest.mark.reference, pytest.mark.swm]

SIZE = 96
OPTIONS = {
    "variation": [
        "environment.start_height_ratio",
        "environment.start_horizontal_offset",
        "environment.start_tilt",
    ]
}
ACTIONS = [np.zeros(7)] * 2 + [np.array([0.15, -0.1, 0.2, 1.0, 0.5, 0.1, -0.1])] * 6


@pytest.mark.parametrize("seed", [1000000, 1000003])
def test_reset_frame_and_engine_follow_the_rendering_source(seed):
    from tests.helpers.reference.rocket_semantics import RocketLandingEnv as Oracle
    from tests.helpers.reference.rocket_semantics import capture, differences

    source = Oracle(render_mode="rgb_array", render_resolution=(SIZE, SIZE))
    ours = se.make("PyFlytRocketLanding", num_envs=1, size=SIZE, simulation_backend="pybullet")
    try:
        source.reset(seed=seed, options=OPTIONS)
        ours.reset(seed=seed, options=OPTIONS)

        def check():
            a, b = capture(source), capture(ours._simulator.envs[0].env)
            # The source keeps its rendered goal image in State; without a render mode it stays zero.
            del a["wrapper/current_goal"], b["wrapper/current_goal"]
            assert not differences(a, b)
            np.testing.assert_array_equal(ours.render().cpu().numpy()[0], source.render())

        check()
        for action in ACTIONS:
            source.step(action.copy())
            ours.step(action[None].copy(), frames=None)
            check()
    finally:
        ours.close()
        source.close()
        if source.modified_pad_urdf:
            Path(source.modified_pad_urdf).unlink(missing_ok=True)
