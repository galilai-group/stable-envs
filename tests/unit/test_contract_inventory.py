"""Inventory completeness does not depend on which extras are installed."""

from pathlib import Path

from tests.helpers.reference.crosswalk import SWM_IDS
from tests.helpers.reference.inventory import EXCLUDED, inventory


def test_each_noncraftax_registration_keeps_its_own_required_gates():
    rows = inventory()["registrations"]
    assert len(rows) == 38
    assert {row["registration"] for row in rows} == SWM_IDS.keys() - EXCLUDED
    root = Path(__file__).resolve().parents[2]
    for row in rows:
        assert row["source_scope"] and row["classes"]
        assert set(row["required_gates"]) == {"state", "simulation", "task", "render", "online"}
        assert row["current_run_status"] == "NOT_RUN"
        for paths in row["required_gates"].values():
            assert paths and all((root / path).is_file() for path in paths)


def test_shared_profiles_cover_each_registration_once():
    from tests.contracts.conftest import case_map
    from tests.helpers.reference.profiles import PROFILES

    registrations = [case.registration for cases in PROFILES.values() for case in cases]
    assert len(registrations) == len(set(registrations)) == 38
    assert set(registrations) == SWM_IDS.keys() - EXCLUDED
    expected_counts = dict(native=56, fetch=192, dmc=192, navigation=48, ogbench=48, pusht=24, rocket=16)
    for device in ("cpu", "cuda"):
        for profile, expected in expected_counts.items():
            cases = list(case_map(profile, device).values())
            assert len(cases) == len(set(cases)) == expected
            assert {case.device for case in cases} == {device}
