from collections.abc import Callable
from pathlib import Path

import pytest

import stable_envs as se
from tests.helpers.reference.pixels import check_online_pixels
from tests.helpers.reference.profiles import Registration
from tests.helpers.reference.pusht_pixels import compare_padded_states, pusht_trace

pytestmark = [pytest.mark.integration, pytest.mark.reference]


@pytest.mark.parametrize("scenario", ["reached", "contact", "terminal"])
@pytest.mark.parametrize("horizon", [1, 8])
def test_source_rollout_pixels(
    source_case: tuple[Registration, int, str],
    source_identity: dict,
    scenario: str,
    horizon: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    case, seed, dtype = source_case
    if case.family != "pusht":
        pytest.skip("PushT source pixels belong to the PushT profile")
    trace = pusht_trace(case, seed, dtype, horizon, scenario)
    device = request.config.getoption("--contract-device")
    with se.make(case.name, num_envs=2, size=64, device=device) as env:
        errors = []
        for i, (states, frames) in enumerate(
            [
                (None, None),
                (slice(None), slice(20, 30)),
                (None, slice(-1, None)),
                (slice(None), slice(None)),
                (slice(None, None, -2), slice(None, None, -2)),
            ]
        ):
            errors.extend(
                check_online_pixels(
                    env,
                    trace,
                    states=states,
                    frames=frames,
                    mse_max=0.001,
                    failure_dir=tmp_path / f"selection-{i}",
                    compare_states=compare_padded_states,
                ).reshape(-1)
            )
        record_property("source_render_device", "cpu")
        record_property("candidate_raster_device", "cpu")
        record_property("candidate_output_device", device)
        record_property("oracle_mode", "independent-source-diagnostics-witnessed-replay-State")
        record_property("mse_max", float(max(errors)))
        record_property("frames", len(errors))
