"""Coverage index over the accepted matrix, without importing environments.

References preserve each registration's original scope. Presence in this index
means a gate is required, not that it has run against the current checkout.
"""

import json
from pathlib import Path

from tests.helpers.reference.profiles import PROFILES

ROOT = Path(__file__).resolve().parents[3]
EXCLUDED = frozenset(
    {"swm/CraftaxClassicPixels-v1", "swm/CraftaxClassicSymbolic-v1", "swm/CraftaxPixels-v1", "swm/CraftaxSymbolic-v1"}
)
SHARED_CONTRACTS = [
    "tests/contracts/test_contract_state.py",
    "tests/contracts/test_simulation.py",
    "tests/contracts/test_task.py",
]
ONLINE = {
    "classic_control": "benchmarks/online/classic_source_pipeline.py",
    "dmcontrol": "benchmarks/online/dmc_source_transport_gate.py",
    "fetch": "benchmarks/online/fetch_source_transport_gate.py",
    "navigation": "benchmarks/online/navigation_source_pipeline.py",
    "ogbench": "benchmarks/online/ogbench_source_transport_gate.py",
}


def inventory():
    matrix = json.loads((ROOT / "tests/fixtures/semantic_recovery_matrix.json").read_text())
    profiles = {case.registration: profile for profile, cases in PROFILES.items() for case in cases}
    rows = []
    for row in matrix["rows"]:
        registration = row["historical_id"]
        if registration in EXCLUDED:
            continue
        recorded = [value.strip() for value in row["test"].split(";")]
        # Historical acceptance drivers outside tests/ were retired; the shared contract
        # profiles cover every registration here and replace them as the required gates.
        historical = [path for path in recorded if not path.startswith("tests/")]
        gates = [path for path in recorded if path.startswith("tests/")]
        if historical:
            gates += SHARED_CONTRACTS
        if profiles.get(registration) == "native":
            gates.append("tests/contracts/test_native_acceptance.py")
        online = [ONLINE[row["family"]]] if row["family"] in ONLINE else []
        if row["family"] == "rocket":
            online.append("tests/integration/test_rocket_source_pixels.py")
        profile = profiles.get(registration)
        if profile in {"native", "navigation", "dmc", "fetch", "ogbench", "pusht"}:
            online.append(f"tests/contracts/test_{profile}_source_pixels.py")
        rows.append(
            dict(
                registration=registration,
                classes=row["stable_envs_classes"],
                family=row["family"],
                source_scope=row["verification_scope"],
                source_audit=row.get("audit_report"),
                historical_gates=historical,
                required_gates={
                    "state": gates,
                    "simulation": gates,
                    "task": gates,
                    "render": ["tests/gpu/test_conformance.py"],
                    "online": online,
                },
                render_scope=row["renderer_regression"],
                execution_mode=row["execution_mode"],
                remaining_limits=row["remaining_issues"],
                source_regression_profile=profiles.get(registration),
                migration="shared regression subset"
                if registration in profiles
                else "existing family drivers retained",
                current_run_status="NOT_RUN",
            )
        )
    return dict(
        schema="stable-envs-test-inventory/v1",
        excluded=sorted(EXCLUDED),
        registrations=rows,
        scope="Coverage inventory, not current acceptance or device/performance certification",
    )
