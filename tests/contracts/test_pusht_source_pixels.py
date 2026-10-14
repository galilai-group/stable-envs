from collections.abc import Callable
from pathlib import Path

import pytest

import stable_envs as se
from tests.helpers.reference.pixels import check_online_selections
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
        check_online_selections(env, trace, tmp_path, record_property, compare_states=compare_padded_states)
        record_property("source_render_device", "cpu")
        record_property("candidate_raster_device", "cpu")
        record_property("candidate_output_device", device)
        record_property("oracle_mode", "independent-source-diagnostics-witnessed-replay-State")
