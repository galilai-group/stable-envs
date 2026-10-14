from collections.abc import Callable
from pathlib import Path

import pytest

import stable_envs as se
from tests.helpers.reference.dmc_pixels import dmc_trace
from tests.helpers.reference.pixels import check_online_selections
from tests.helpers.reference.profiles import Registration

pytestmark = [pytest.mark.integration, pytest.mark.reference]


@pytest.mark.parametrize("horizon", [1, 8])
def test_source_rollout_pixels(
    source_case: tuple[Registration, int, str],
    source_identity: dict,
    horizon: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
) -> None:
    case, seed, dtype = source_case
    if case.family != "dmc":
        pytest.skip("DM Control source pixels belong to the dmc profile")
    device = request.config.getoption("--contract-device")
    trace, identity = dmc_trace(case, seed, dtype, horizon, device, source_identity["root"])
    record_property("source_physics_device", "cpu")
    record_property("source_render_device", device)
    for key, value in identity.items():
        record_property("source_gl_" + key, value)
    with se.make(case.name, num_envs=2, device=device, size=64) as env:
        check_online_selections(env, trace, tmp_path, record_property)
