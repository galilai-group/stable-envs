"""The pin-* dependency groups and profiles.PINS state the same reference versions."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from tests.helpers.reference.profiles import PINS

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
GROUPS = PYPROJECT["dependency-groups"]
GROUP_OF_PROFILE = {"native": "pin-classic"}
# Extras whose code checks exact versions at import or construction; every other extra states a range.
EXACT_EXTRAS = {"pusht", "rocket", "ogbench-manipulation", "ogbench-manipulation-gpu"}
# Packages that speed up stable-envs without being part of a source reference; pinned, but not in PINS.
ACCELERATORS = {"numba", "mujoco-mjx"}


def _exact(group: str) -> dict[str, str]:
    pins = {}
    for entry in GROUPS[group]:
        req = Requirement(entry)
        (spec,) = req.specifier
        assert spec.operator == "==", entry
        assert str(req.marker) == 'python_version >= "3.12"', entry
        pins[canonicalize_name(req.name)] = spec.version
    return pins


@pytest.mark.parametrize("profile", sorted(PINS))
def test_each_reference_profile_has_a_pin_group_with_its_exact_versions(profile):
    group = GROUP_OF_PROFILE.get(profile, f"pin-{profile}")
    expected = {canonicalize_name(name): version for name, version in PINS[profile].items()}
    assert {name: v for name, v in _exact(group).items() if name not in ACCELERATORS} == expected


def test_every_pin_group_is_mutually_exclusive_with_the_others():
    pins = {name for name in GROUPS if name.startswith("pin-")}
    sets = [{entry.get("group") for entry in row} for row in PYPROJECT["tool"]["uv"]["conflicts"]]
    assert any(pins <= s for s in sets), "add every pin-* group to the one mutually exclusive conflict set"


def test_user_extras_state_ranges_not_reference_pins():
    extras = PYPROJECT["project"]["optional-dependencies"]
    pinned = {name: [req for req in reqs if "==" in req] for name, reqs in extras.items() if name not in EXACT_EXTRAS}
    assert {name: reqs for name, reqs in pinned.items() if reqs} == {}
