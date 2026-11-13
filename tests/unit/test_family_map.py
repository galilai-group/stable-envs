"""ci/families.toml names real extras, groups and paths, and gives every test file one owner."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

from tests.helpers.families import CONFIG, FAMILIES

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_every_test_file_matches_at_most_one_family():
    owners = {}
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        owners[path.relative_to(ROOT).as_posix()] = [
            family
            for family, spec in FAMILIES.items()
            if any(fnmatch.fnmatchcase(path.name, pattern) for pattern in spec.get("tests", ()))
        ]
    assert {path: found for path, found in owners.items() if len(found) > 1} == {}


def test_every_family_test_glob_matches_a_file():
    names = [path.name for path in (ROOT / "tests").rglob("test_*.py")]
    stale = [
        (family, pattern)
        for family, spec in FAMILIES.items()
        for pattern in spec.get("tests", ())
        if not fnmatch.filter(names, pattern)
    ]
    assert stale == []


def test_profiles_install_declared_extras_and_groups():
    extras = PYPROJECT["project"]["optional-dependencies"]
    groups = PYPROJECT["dependency-groups"]
    for family, spec in FAMILIES.items():
        assert set(spec.get("extras", ())) <= set(extras), family
        assert set(spec.get("groups", ())) <= set(groups), family


def test_every_source_path_glob_matches_a_file():
    files = [path.name for path in ROOT.iterdir() if path.is_file()]
    for top in ("stable_envs", "tests", "ci", "scripts", "benchmarks", ".github"):
        files += [path.relative_to(ROOT).as_posix() for path in (ROOT / top).rglob("*") if path.is_file()]
    globs = [*CONFIG["shared"]["paths"], *(p for spec in FAMILIES.values() for p in spec.get("paths", ()))]
    assert [glob for glob in globs if not fnmatch.filter(files, glob)] == []


def test_the_swm_pin_is_defined_once_as_the_families_toml_commit():
    """ci/families.toml [swm] names the stable-worldmodel commit; saved.py holds the one test constant for it."""
    files = [
        ROOT / "tests/helpers/reference/saved.py",
        *sorted((ROOT / "tests/helpers/reference/gates").glob("*.py")),
    ]
    found = {
        path.relative_to(ROOT).as_posix(): value
        for path in files
        for value in re.findall(r'^(?:SWM_PIN|PIN) = "([0-9a-f]{40})"', path.read_text(), re.M)
    }
    assert found == {"tests/helpers/reference/saved.py": CONFIG["swm"]["commit"]}


def test_ci_runs_every_shard_of_every_profile_and_the_shards_partition_a_selection():
    import importlib.util

    from tests._runner.shard import UNKNOWN, split

    spec = importlib.util.spec_from_file_location("plan", ROOT / "scripts" / "ci" / "plan.py")
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    for tier in ("pr", "nightly"):
        jobs = plan.matrix(list(FAMILIES), tier)["include"]
        for family, entry in FAMILIES.items():
            count = int(entry.get("shards", {}).get(tier, 1))
            assert [job["shard"] for job in jobs if job["profile"] == family] == [
                f"{i}/{count}" for i in range(1, count + 1)
            ]
    ids = [f"tests/unit/test_x.py::test_{i}" for i in range(40)]
    durations = {nodeid: float(i % 7) for i, nodeid in enumerate(ids[::2])}
    for count in (1, 3, 8):
        shard = split(reversed(ids), count, durations)
        assert shard == split(ids, count, durations) and set(shard) == set(ids)
        loads = [sum(durations.get(n, UNKNOWN) for n in ids if shard[n] == k) for k in range(count)]
        assert max(loads) - min(loads) <= max(durations.values())
