"""Every registered env has an owning family, a source comparison and, where its source varies, a variation map.

Waivers live only in ``tests/exemptions.toml``, and that file is checked here for stale entries.
"""

from __future__ import annotations

import fnmatch
import re

import pytest

import stable_envs as se
from tests.helpers import families
from tests.helpers.exemptions import CHECKS, ENTRIES, owners, param, source_owners
from tests.helpers.reference.cases import SOURCE_GATES
from tests.helpers.reference.crosswalk import swm_ids_for_class
from tests.helpers.reference.variation import variation_map

pytestmark = pytest.mark.unit


def envs(check):
    return [param(name, se.registry.REGISTRY[name], check) for name in se.list_envs()]


def _defines(nodeid: str) -> bool:
    path, _, function = nodeid.partition("::")
    file = families.ROOT / path
    return file.is_file() and re.search(rf"^def {re.escape(function)}\(", file.read_text(), re.M) is not None


def _module_owners(key: str) -> list[str]:
    for package in (key, key.split("_")[0]):
        if (families.ROOT / "stable_envs" / "envs" / package).is_dir():
            return source_owners(f"stable_envs/envs/{package}/__init__.py")
    return []


@pytest.mark.parametrize("name", envs("owner"))
def test_env_has_one_owning_family(name):
    found = owners(se.registry.REGISTRY[name])
    assert len(found) == 1, f"{name}: ci/families.toml must give its module exactly one family, found {found}"


@pytest.mark.parametrize("name", envs("source"))
def test_env_is_compared_with_its_source(name):
    gate = SOURCE_GATES.get(name)
    assert gate, f"{name}: no source comparison in tests/helpers/reference/cases.py::SOURCE_GATES"
    assert _defines(gate), f"{name}: its source gate {gate} does not exist"
    owner = families.family_of_file(gate.partition("::")[0].rsplit("/", 1)[-1])
    family = owners(se.registry.REGISTRY[name])
    assert [owner] == family, f"{name}: its source gate {gate} runs in profile {owner}, not {family}"


@pytest.mark.parametrize("name", envs("variation_map"))
def test_env_with_a_varying_source_has_a_variation_map(name):
    cls = se.registry.REGISTRY[name]
    swm = swm_ids_for_class(name) or str(getattr(cls, "reference_id", "") or "").startswith("swm/")
    assert not swm or variation_map(name) is not None, (
        f"{name}: its stable-worldmodel source has a variation space and no VariationMap describes it "
        f"(tests/helpers/reference/variation_maps)"
    )


@pytest.mark.parametrize("family", [pytest.param(f, marks=pytest.mark.family(f)) for f in families.FAMILIES])
def test_an_installed_family_registers_its_envs(family):
    if families.unmet(family):
        pytest.skip(f"family {family} not installed")
    failed = {module: error for module, error in se.import_errors().items() if family in _module_owners(module)}
    assert failed == {}, f"family {family} is installed but these env modules did not import: {failed}"


@pytest.mark.parametrize(
    "entry",
    [pytest.param(e, marks=pytest.mark.family(e.get("family", "core")), id=f"{e['env']}") for e in ENTRIES],
)
def test_an_exemption_names_live_envs_and_known_checks(entry):
    assert set(entry) == {"env", "family", "checks", "reason"}, entry
    assert set(entry["checks"]) <= set(CHECKS), f"unknown check in {entry['checks']}; known: {CHECKS}"
    assert entry["family"] in families.FAMILIES and entry["reason"].strip()
    if families.unmet(entry["family"]):
        pytest.skip(f"family {entry['family']} not installed")
    matched = [name for name in se.list_envs() if fnmatch.fnmatchcase(name, entry["env"])]
    assert matched, f"stale exemption: no registered env matches {entry['env']}"
    for name in matched:
        assert owners(se.registry.REGISTRY[name]) == [entry["family"]], f"{name} belongs to another family"
        for check in entry["checks"]:
            waived = [e for e in ENTRIES if check in e["checks"] and fnmatch.fnmatchcase(name, e["env"])]
            assert len(waived) == 1, f"{name}/{check} is waived by more than one entry"
