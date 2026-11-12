"""Every registered env has an owning family, source comparisons and, where its source varies, a variation map.

Source comparisons come in two layers. ``ORIGINAL_GATES`` compare with the upstream library an env reproduces
(or a corpus recorded from it), import no stable_worldmodel and run on pull requests. ``SWM_GATES`` compare with
live stable-worldmodel and carry the ``swm`` mark. Every test file runs in a profile that installs
what it imports. Waivers live only in ``tests/exemptions.toml``, and
that file is checked here for stale entries.
"""

from __future__ import annotations

import ast
import fnmatch
import re

import pytest

import stable_envs as se
from tests.helpers import families
from tests.helpers.exemptions import CHECKS, ENTRIES, owners, param, source_owners
from tests.helpers.reference.cases import ORIGINAL_GATES, SWM_GATES
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


def _tree(path: str) -> ast.Module:
    return ast.parse((families.ROOT / path).read_text())


def _marked(nodeid: str, mark: str) -> bool:
    """Whether the test function carries ``pytest.mark.<mark>``: itself, through its module or an alias."""
    path, _, function = nodeid.partition("::")
    tree = _tree(path)
    carriers = {
        t.id
        for node in tree.body
        if isinstance(node, ast.Assign) and f"mark.{mark}" in ast.unparse(node.value)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    defined = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    return "pytestmark" in carriers or any(
        f"mark.{mark}" in ast.unparse(d) or ast.unparse(d) in carriers for d in defined.decorator_list
    )


def _imports(tree: ast.Module) -> set[str]:
    """Top-level names of the modules a file imports or importorskips, anywhere in it."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
        elif (
            isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("importorskip")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            found.add(str(node.args[0].value).split(".")[0])
    return found


def _gate(name: str, table: dict, label: str) -> str:
    gate = table.get(name)
    assert gate, f"{name}: no entry in tests/helpers/reference/cases.py::{label}"
    assert _defines(gate), f"{name}: its gate {gate} does not exist"
    owner = families.family_of_file(gate.partition("::")[0].rsplit("/", 1)[-1])
    family = owners(se.registry.REGISTRY[name])
    assert [owner] == family, f"{name}: its gate {gate} runs in profile {owner}, not {family}"
    return gate


def _swm_source(name: str) -> bool:
    cls = se.registry.REGISTRY[name]
    return bool(swm_ids_for_class(name)) or str(getattr(cls, "reference_id", "") or "").startswith("swm/")


@pytest.mark.parametrize("name", envs("original"))
def test_env_is_compared_with_the_library_it_ports(name):
    gate = _gate(name, ORIGINAL_GATES, "ORIGINAL_GATES")
    assert "stable_worldmodel" not in _imports(_tree(gate.partition("::")[0])), f"{gate} imports stable_worldmodel"
    assert not _marked(gate, "swm"), f"{gate} is marked swm; the original layer must not need stable-worldmodel"
    assert not _marked(gate, "nightly"), f"{gate} is nightly; every env needs a source comparison on pull requests"


@pytest.mark.parametrize("name", envs("swm"))
def test_env_with_a_swm_source_is_compared_with_it(name):
    if not _swm_source(name):
        assert name not in SWM_GATES, f"{name} has no stable-worldmodel registration; drop its SWM_GATES entry"
        return
    gate = _gate(name, SWM_GATES, "SWM_GATES")
    assert _marked(gate, "swm"), f"{gate} compares with stable-worldmodel and must carry pytest.mark.swm"


#: Installed in every profile by the test dependency group.
_EVERYWHERE = {"pygame"}


def _requires(family: str) -> set[str]:
    spec = families.FAMILIES[family]
    names = {re.split(r"[<>=!\[; ]", entry)[0].replace("-", "_") for entry in spec.get("requires", ())}
    return names | ({"stable_worldmodel"} if "swm" in spec.get("capabilities", ()) else set())


_FAMILY_MARK = re.compile(r"mark\.family\(['\"](\w+)['\"]\)")
_FAMILY_IMPORTS = {"stable_worldmodel"}.union(*(_requires(f) for f in families.FAMILIES)) - _EVERYWHERE
_TEST_FILES = sorted(
    p.relative_to(families.ROOT).as_posix()
    for bucket in ("unit", "integration", "regression")
    for p in (families.ROOT / "tests" / bucket).glob("test_*.py")
)


@pytest.mark.parametrize("path", _TEST_FILES)
def test_a_test_file_runs_in_a_profile_that_installs_its_imports(path):
    """A family's requirements imported where no profile running the file installs them would only ever skip."""
    text = (families.ROOT / path).read_text()
    tree = _tree(path)
    imports = _imports(tree)
    family = families.family_of_file(path.rsplit("/", 1)[-1])
    named = _FAMILY_MARK.findall(text)
    missing = (imports & _FAMILY_IMPORTS) - set().union(*(_requires(f) for f in {family, *named}))
    assert not missing, (
        f"{path} runs in profiles {sorted({family, *named})}, none of which installs {sorted(missing)}: give the "
        f"file (ci/families.toml `tests`) or the test (pytest.mark.family) to a family that does"
    )
    module = [
        m
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in n.targets)
        for m in _FAMILY_MARK.findall(ast.unparse(n.value))
    ]
    for test in (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")):
        marks = [m for d in test.decorator_list for m in _FAMILY_MARK.findall(ast.unparse(d))]
        owner = (marks or module or [family])[0]
        missing = (_imports(test) & _FAMILY_IMPORTS) - _requires(owner)
        assert not missing, f"{path}::{test.name} runs in profile {owner}, which does not install {sorted(missing)}"
    if "stable_worldmodel" in imports:
        assert "mark.swm" in text, f"{path} imports stable_worldmodel without pytest.mark.swm"


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
