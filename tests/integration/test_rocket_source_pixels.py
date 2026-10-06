import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.pixels import check_online_pixels
from tests.helpers.reference.rocket_pixels import SCENARIOS, rocket_trace

pytestmark = [pytest.mark.integration, pytest.mark.reference]

SELECTIONS = (None, slice(9, 10), slice(None), slice(None, None, -2))


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("horizon", [1, 8])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_source_rollout_pixels(scenario, dtype, horizon, device, tmp_path, record_property):
    pytest.importorskip("PyFlyt")
    pytest.importorskip("pybullet")
    pytest.importorskip("stable_worldmodel")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA source-pixel tests run on Oscar")
    with rocket_trace(scenario, dtype, horizon) as (config, trace):
        with se.make("PyFlytRocketLanding", num_envs=2, device=device, size=96, simulator_config=config) as env:
            errors = []
            # Preserve all four historical selectors, and exercise independent State/frame selections.
            for i, (states, frames) in enumerate([(s, s) for s in SELECTIONS] + [(None, slice(-1, None))]):
                errors.extend(
                    check_online_pixels(
                        env,
                        trace,
                        states=states,
                        frames=frames,
                        mse_max=0.001,
                        failure_dir=tmp_path / f"selection-{i}",
                    ).reshape(-1)
                )
            record_property("source_device", "cpu")
            record_property("mse_max", float(max(errors)))
            record_property("frames", len(errors))
