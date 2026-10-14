"""The shared source profiles cover every non-Craftax registration once."""

from pathlib import Path

from tests.helpers.reference.crosswalk import SWM_IDS

CRAFTAX = frozenset(
    {"swm/CraftaxClassicPixels-v1", "swm/CraftaxClassicSymbolic-v1", "swm/CraftaxPixels-v1", "swm/CraftaxSymbolic-v1"}
)


def test_shared_profiles_cover_each_registration_once():
    from tests.contracts.conftest import case_map
    from tests.helpers.reference.profiles import PROFILES

    registrations = [case.registration for cases in PROFILES.values() for case in cases]
    assert len(registrations) == len(set(registrations)) == 38
    assert set(registrations) == SWM_IDS.keys() - CRAFTAX
    expected_counts = dict(native=128, fetch=396, dmc=288, navigation=96, ogbench=72, pusht=60, rocket=16)
    for device in ("cpu", "cuda"):
        for profile, expected in expected_counts.items():
            cases = list(case_map(profile, device).values())
            assert len(cases) == len(set(cases)) == expected
            assert {case.device for case in cases} == {device}


def test_source_helpers_do_not_depend_on_research_drivers():
    """Reference helpers do not import scripts or unit tests, and do not execute code dynamically."""
    import ast

    root = Path(__file__).resolve().parents[1] / "helpers" / "reference"
    for path in root.rglob("*.py"):
        if path.name == "craftax_source.py":  # Craftax is outside the 38 shared profiles.
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            assert not any(m == "scripts" or m.startswith(("scripts.", "tests.unit.")) for m in modules), path
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in {"spec_from_file_location", "exec_module", "run_path", "exec", "eval"}, path
