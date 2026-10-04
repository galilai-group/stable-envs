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
