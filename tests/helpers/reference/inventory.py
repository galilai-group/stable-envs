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
ONLINE = {
    "classic_control": "scripts/benchmarks/online_stepping/classic_source_pipeline.py",
    "dmcontrol": "scripts/benchmarks/online_stepping/dmc_source_transport_gate.py",
    "fetch": "scripts/benchmarks/online_stepping/fetch_source_transport_gate.py",
    "navigation": "scripts/benchmarks/online_stepping/navigation_source_pipeline.py",
    "ogbench": "scripts/benchmarks/online_stepping/ogbench_source_transport_gate.py",
    "pusht": "scripts/research_replay/pusht_public_gate.py",
    "rocket": "scripts/research_rocket_production/online_source_gate.py",
}


def inventory():
    matrix = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    profiles = {case.registration: profile for profile, cases in PROFILES.items() for case in cases}
    rows = []
    for row in matrix["rows"]:
        registration = row["historical_id"]
        if registration in EXCLUDED:
            continue
        gates = [value.strip() for value in row["test"].split(";")]
        if profiles.get(registration) == "native":
            gates.append("tests/contracts/test_native_acceptance.py")
        online = [ONLINE[row["family"]]]
        if row["family"] == "rocket":
            online.append("tests/integration/test_rocket_source_pixels.py")
        if profiles.get(registration) == "native":
            online.append("tests/contracts/test_native_source_pixels.py")
        rows.append(
            dict(
                registration=registration,
                classes=row["stable_envs_classes"],
                family=row["family"],
                source_scope=row["verification_scope"],
                source_audit=row.get("audit_report"),
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
