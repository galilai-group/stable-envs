import json
from collections.abc import Callable
from pathlib import Path

import pytest

import stable_envs as se
from tests.helpers.reference.fetch_pixels import fetch_mutation_witness, fetch_trace
from tests.helpers.reference.pixels import check_online_pixels
from tests.helpers.reference.profiles import Registration

pytestmark = [pytest.mark.integration, pytest.mark.reference]


@pytest.mark.parametrize("scenario", ["reached", "horizon"])
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
    if case.family != "fetch":
        pytest.skip("Fetch source pixels belong to the Fetch profile")
    device = request.config.getoption("--contract-device")
    trace, identities = fetch_trace(case, seed, dtype, horizon, scenario, device, source_identity["root"])
    with se.make(case.name, num_envs=2, size=64, device=device, backend="opengl") as env:
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
                ).reshape(-1)
            )
        record_property("source_physics_device", "cpu")
        record_property("source_render_device", device)
        record_property("source_gl", json.dumps(identities, sort_keys=True))
        record_property("candidate_backend", "opengl")
        record_property(
            "oracle_mode",
            "disposable-forward-source-branch" if case.domain == "PickAndPlace" else "continuous-forward-source",
        )
        record_property("mse_max", float(max(errors)))
        record_property("frames", len(errors))


def test_source_render_changes_continuation(
    source_case: tuple[Registration, int, str],
    source_identity: dict,
    request: pytest.FixtureRequest,
    record_property: Callable[[str, object], None],
) -> None:
    case, seed, dtype = source_case
    if case.family != "fetch" or case.domain != "PickAndPlace":
        pytest.skip("Rejected continuous-render source witness belongs to PickAndPlace")
    witness = fetch_mutation_witness(
        case, seed, dtype, request.config.getoption("--contract-device"), source_identity["root"]
    )
    record_property("rejected_continuous_source_render", json.dumps(witness, sort_keys=True))
