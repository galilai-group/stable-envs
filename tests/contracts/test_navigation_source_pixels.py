from collections.abc import Callable
from pathlib import Path

import pytest

import stable_envs as se
from tests.helpers.reference.navigation_pixels import navigation_trace
from tests.helpers.reference.pixels import check_online_selections
from tests.helpers.reference.profiles import Registration

pytestmark = [pytest.mark.integration, pytest.mark.reference]


@pytest.mark.parametrize("scenario", ["reached", "terminal"])
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
    if case.family != "navigation":
        pytest.skip("Navigation source pixels belong to the navigation profile")
    trace = navigation_trace(case, seed, dtype, horizon, scenario)
    device = request.config.getoption("--contract-device")
    with se.make(case.name, num_envs=2, device=device) as env:
        check_online_selections(env, trace, tmp_path, record_property)
        record_property("source_device", "cpu")
