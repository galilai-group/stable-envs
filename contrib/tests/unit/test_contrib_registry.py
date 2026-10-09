"""The contrib classes against the frozen registry baseline, split out of ``tests/unit/test_registry.py``.

The baseline (``tests/fixtures/regression/references/registry_baseline.json``) was recorded when these
modules were ``stable_envs.envs.<name>``. ``tests/unit/test_registry.py`` now compares only the packaged
rows; the rows whose module was a contrib family are compared here, with the same fingerprint rule:
if this interpreter imports a different set of contrib modules than the baseline did, the class-by-class
comparison is skipped, loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import contrib
from stable_envs import registry as R
from tests.helpers.reference import canonical_from_frozen

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "tests" / "fixtures" / "regression" / "references" / "registry_baseline.json"
OLD_MODULES = {f"stable_envs.envs.{name}": f"contrib.{name}" for name in contrib.MODULES}


def test_every_contrib_class_resolves_to_the_same_class_as_at_the_baseline():
    if not BASELINE.exists():
        pytest.skip(f"no baseline registry table recorded at {BASELINE}")
    base = json.loads(BASELINE.read_text())
    then_errors = {k for k in base["import_errors"] if k in contrib.MODULES}
    if then_errors != set(contrib.import_errors()):
        pytest.skip(
            "this interpreter imports a different set of contrib modules than the one the baseline "
            f"was recorded on: baseline {sorted(then_errors)} vs {sorted(contrib.import_errors())}"
        )
    now = {r["class_name"]: r for r in R.registry_table() if r["module"].startswith("contrib.")}
    then = {canonical_from_frozen(r["name"]): r for r in base["registry_table"] if r["module"] in OLD_MODULES}
    assert sorted(now) == sorted(then), "unexpected registered contrib class change"
    skip_keys = {
        "module",
        "name",
        "class_name",
        "family",
        "domain",
        "render_state_dim",
        "state_dim",
        "reference_id",
    }
    for name, row in then.items():
        now_cmp = {k: v for k, v in now[name].items() if k not in skip_keys}
        then_cmp = {k: v for k, v in row.items() if k not in skip_keys}
        assert now_cmp == then_cmp, f"the registry row for {name} changed"
        assert now[name]["module"] == OLD_MODULES[row["module"]], f"{name} is not in the module it moved to"
        assert R._resolve(name) is R.REGISTRY[name]


def test_contrib_load_records_its_failures_rather_than_hiding_them():
    errs = contrib.load()
    assert errs == contrib.import_errors()
    assert set(errs) <= set(contrib.MODULES)
    assert all(isinstance(v, str) and v for v in errs.values())
    registered = {c.__module__.rsplit(".", 1)[-1] for c in R.REGISTRY.values() if c.__module__.startswith("contrib.")}
    assert set(errs) & registered == set(), "a module cannot both have failed to import and have registered a class"
