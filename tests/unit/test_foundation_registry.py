"""The registry after the move: the same classes under the same names, plus the 42 SWM ids.

TWO ORACLES, and they are different in kind:

  * the 42 SWM ids are asserted against the ACCEPTED STEP-8 SUPPORT MATRIX
    (``examples/results/step8/support_matrix.json``), which is the behavioral oracle Step 9 is not
    allowed to change. The alias table in ``registry.py`` is a copy of it and this test is what stops
    the copy drifting.
  * the canonical class names are asserted against the BASELINE REGISTRY TABLE recorded from
    ``21cb306`` by ``scratch/step9/foundation/pusht_oracle.py``
    (``examples/results/step9/foundation/registry_baseline.json``) — same interpreter, same optional
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
MATRIX = ROOT / "examples" / "results" / "step8" / "support_matrix.json"
BASELINE = ROOT / "examples" / "results" / "step9" / "foundation" / "registry_baseline.json"
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
    if len(cells) == 1 and cells[0] in R.REGISTRY:
        assert R.spec(swm_id)["class"] is R.REGISTRY[cells[0]]
        assert swm_id in R.spec(cells[0])["swm_ids"]
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
        pytest.skip(f"no baseline registry table recorded at {BASELINE}; run "
                    f"scratch/step9/foundation/pusht_oracle.py")
    base = json.loads(BASELINE.read_text())
    if set(base["import_errors"]) != set(snv.import_errors()):
        pytest.skip("this interpreter imports a different set of env families than the one the "
                    f"baseline was recorded on: baseline {sorted(base['import_errors'])} vs "
                    f"{sorted(snv.import_errors())}")
    now = {r["name"]: r for r in R.registry_table()}
    then = {r["name"]: r for r in base["registry_table"]}
    assert sorted(now) == sorted(then), "the set of registered class names changed"
    for name, row in then.items():
        # ``module`` is compared SEPARATELY, and by what it guarantees rather than by its spelling.
        # Wave 1 moves files on purpose (plan §3 renames ``swm_nav``/``simple_point_maze`` to
        # ``navigation``, ``dmcontrol.py`` to a package, ...), and the guarantee the baseline row was
        # protecting is that the class is still THE SAME CLASS and still reachable under the import
        # path it had. A workstream that moves a class must leave that path working, which is what
        # the deprecated shim modules are for; the semantic columns below may not move at all.
        assert {k: v for k, v in now[name].items() if k != "module"} == \
               {k: v for k, v in row.items() if k != "module"}, \
            f"the registry row for {name} changed"
        if now[name]["module"] != row["module"]:
            old = importlib.import_module(row["module"])
            assert getattr(old, name, None) is R.REGISTRY[name], (
                f"{name} moved from {row['module']} to {now[name]['module']} and the old import path "
                f"no longer yields it; a moved class keeps a shim until Wave 3 retires it")
        assert R.make.__module__ and R._resolve(name, "make") is R.REGISTRY[name]


def test_list_envs_and_spec_answer_without_constructing_anything():
    names = R.list_envs()
    assert names == sorted(R.REGISTRY) and "PushT" in names
    assert R.list_envs(family="pusht") == ["PushT"]
    s = R.spec("PushT")
    assert s["state_dim"] == 5 and s["family"] == "pusht" and s["fidelity_tier"] == "parity"
    assert s["swm_ids"] == ("swm/PushT-Discrete-v1", "swm/PushT-v1")
    assert "pusht" in R.list_families()


def test_discovery_records_its_failures_rather_than_hiding_them():
    errs = snv.import_errors()
    assert isinstance(errs, dict)
    assert all(isinstance(v, str) and v for v in errs.values())
    assert set(errs) & {*[c.__module__.rsplit(".", 1)[-1] for c in R.REGISTRY.values()]} == set(), (
        "a module cannot both have failed to import and have registered a class")
