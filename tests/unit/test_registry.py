"""The registry after the move: the same classes under the same names, plus the 42 SWM ids.

TWO ORACLES, and they are different in kind:

  * the 42 SWM ids are asserted against the ACCEPTED STEP-8 SUPPORT MATRIX
    (``tests/fixtures/regression/support_matrix.json``), which is the frozen behavioral oracle this tree is not
    allowed to change. The alias table in ``registry.py`` is a copy of it and this test is what stops
    the copy drifting.
  * the canonical class names are asserted against the BASELINE REGISTRY TABLE recorded from
    ``21cb306`` into the frozen registry baseline
    (``tests/fixtures/regression/references/registry_baseline.json``) — same interpreter, same optional
    dependencies, so "the same classes as before" is a comparison and not an impression.

Which env modules import depends on what is installed, so the recorded fingerprint (the set of modules
that failed to import) is compared first and the class-by-class comparison is skipped, loudly, if the
interpreter is not the one the baseline was taken on.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import stable_envs as snv
from stable_envs import registry as R

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "tests" / "fixtures" / "regression" / "support_matrix.json"
BASELINE = ROOT / "tests" / "fixtures" / "regression" / "references" / "registry_baseline.json"
SWM_IDS_IN_MATRIX = sorted({r["swm_id"] for r in json.loads(MATRIX.read_text())["rows"]})


def test_the_swm_alias_table_is_the_step8_support_matrix_row_for_row():
    rows = json.loads(MATRIX.read_text())["rows"]
    assert len(rows) == 42, "the oracle is 42 SWM ids; this file is not the accepted matrix"
    expected = {r["swm_id"]: tuple(r["stable_envs_cells"]) for r in rows}
    assert R.SWM_IDS == expected
    assert len(R.SWM_IDS) == 42
    assert sum(1 for v in R.SWM_IDS.values() if v) == 40, "40 pixel-producing ids"
    assert R.SYMBOLIC_IDS == ("swm/CraftaxClassicSymbolic-v1", "swm/CraftaxSymbolic-v1")


@pytest.mark.parametrize("swm_id", SWM_IDS_IN_MATRIX)
def test_every_swm_id_resolves_to_the_cells_the_matrix_records(swm_id):
    cells = R.SWM_IDS[swm_id]
    assert R.canonical_names(swm_id) == cells
    live = R._class_for_frozen_cell(cells[0]) if len(cells) == 1 else None
    if len(cells) == 1 and live is not None:
        assert R.info(swm_id)["class"] is live
        assert swm_id in R.info(live.__name__)["swm_ids"]
    elif len(cells) > 1:
        with pytest.raises(KeyError, match="different environments"):
            R.make(swm_id)
    elif not cells:
        with pytest.raises(KeyError, match="SYMBOLIC"):
            R.make(swm_id)


def test_an_unknown_name_lists_what_is_known():
    with pytest.raises(KeyError, match="unknown env"):
        R.make("NoSuchEnv")
    with pytest.raises(KeyError, match="known SWM ids"):
        R.make("swm/NoSuchId-v9")


def test_every_canonical_class_name_resolves_to_the_same_class_as_at_the_baseline():
    if not BASELINE.exists():
        pytest.skip(f"no baseline registry table recorded at {BASELINE}")
    base = json.loads(BASELINE.read_text())
    if set(base["import_errors"]) != set(snv.import_errors()):
        pytest.skip(
            "this interpreter imports a different set of env families than the one the "
            f"baseline was recorded on: baseline {sorted(base['import_errors'])} vs "
            f"{sorted(snv.import_errors())}"
        )
    now = {r["class_name"]: r for r in R.registry_table()}
    then = {R._canonical_from_frozen(r["name"]): r for r in base["registry_table"]}
    assert sorted(now) == sorted(then), "the set of registered class names changed"
    skip_keys = {"module", "name", "class_name", "family", "domain", "render_state_dim", "state_dim"}
    for name, row in then.items():
        # ``module`` may move when unpublished shims are retired. The class is still the same
        # object, reachable at its canonical module; the old import path is gone on purpose.
        # Family/domain/render_state_dim are new explicit metadata, not a behavior change.
        now_cmp = {k: v for k, v in now[name].items() if k not in skip_keys}
        then_cmp = {k: v for k, v in row.items() if k not in skip_keys}
        assert now_cmp == then_cmp, f"the registry row for {name} changed"
        if now[name]["module"] != row["module"]:
            assert getattr(importlib.import_module(now[name]["module"]), name) is R.REGISTRY[name]
        assert R.make.__module__ and R._resolve(name, "make") is R.REGISTRY[name]


def test_list_envs_and_info_answer_without_constructing_anything():
    names = R.list_envs()
    assert names == sorted(R.REGISTRY) and "PushT" in names
    assert R.list_envs(family="pusht") == ["PushT"]
    s = R.info("PushT")
    assert s["render_state_dim"] == 5 and s["family"] == "pusht" and s["fidelity_tier"] == "parity"
    assert s["class_name"] == "PushT" and s["id"] == "push-t"
    assert s["swm_ids"] == ("swm/PushT-Discrete-v1", "swm/PushT-v1")
    assert "pusht" in R.list_families()
    assert "spec" not in dir(snv)
    assert not hasattr(R, "spec")
    assert snv.info is R.info


def test_unpublished_historical_class_names_do_not_resolve():
    """Pre-v0 renames are not public aliases. Frozen SWM ids still find the live class."""
    samples = ("PointmazeLargeNavigate", "MujocoAnt", "PFRocketLanding")
    for old in samples:
        live = R._canonical_from_frozen(old)
        if live == old or live not in R.REGISTRY:
            continue
        with pytest.raises(KeyError, match="unknown env"):
            R.make(old)
        with pytest.raises(AttributeError):
            getattr(snv, old)
        assert getattr(snv, live) is R.REGISTRY[live]
    if "PyFlytRocketLanding" in R.REGISTRY:
        assert R.info("swm/PFRocketLanding-v0")["class"] is R.REGISTRY["PyFlytRocketLanding"]
    if "PointMazeLargeNavigate" in R.REGISTRY:
        with pytest.raises(KeyError, match="different environments"):
            R.make("swm/OGBMaze-v0")


def test_discovery_records_its_failures_rather_than_hiding_them():
    errs = snv.import_errors()
    assert isinstance(errs, dict)
    assert all(isinstance(v, str) and v for v in errs.values())
    assert set(errs) & {*[c.__module__.rsplit(".", 1)[-1] for c in R.REGISTRY.values()]} == set(), (
        "a module cannot both have failed to import and have registered a class"
    )
