"""The registry knows stable-envs class names. The 42 SWM ids live in regression helpers.

TWO ORACLES, and they are different in kind:

  * the 42 SWM ids are asserted against the ACCEPTED STEP-8 SUPPORT MATRIX
    (``tests/fixtures/regression/support_matrix.json``), which is the frozen behavioral oracle this tree is not
    allowed to change. The test-only crosswalk is derived from that file.
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
from tests.helpers.reference import (
    SWM_IDS,
    SYMBOLIC_IDS,
    canonical_from_frozen,
    canonical_names,
    class_for_frozen_cell,
    swm_ids_for_class,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "tests" / "fixtures" / "regression" / "support_matrix.json"
BASELINE = ROOT / "tests" / "fixtures" / "regression" / "references" / "registry_baseline.json"
SWM_IDS_IN_MATRIX = sorted({r["swm_id"] for r in json.loads(MATRIX.read_text())["rows"]})


def test_the_swm_crosswalk_is_the_step8_support_matrix_row_for_row():
    rows = json.loads(MATRIX.read_text())["rows"]
    assert len(rows) == 42, "the oracle is 42 SWM ids; this file is not the accepted matrix"
    expected = {r["swm_id"]: tuple(r["stable_envs_cells"]) for r in rows}
    assert SWM_IDS == expected
    assert len(SWM_IDS) == 42
    assert sum(1 for v in SWM_IDS.values() if v) == 40, "40 pixel-producing ids"
    assert SYMBOLIC_IDS == ("swm/CraftaxClassicSymbolic-v1", "swm/CraftaxSymbolic-v1")
    assert not hasattr(R, "SWM_IDS")
    assert not hasattr(R, "SYMBOLIC_IDS")
    assert not hasattr(R, "canonical_names")


@pytest.mark.parametrize("swm_id", SWM_IDS_IN_MATRIX)
def test_every_swm_id_joins_the_cells_the_matrix_records(swm_id):
    cells = SWM_IDS[swm_id]
    assert canonical_names(swm_id) == tuple(canonical_from_frozen(cell) for cell in cells)
    live = class_for_frozen_cell(cells[0]) if len(cells) == 1 else None
    with pytest.raises(KeyError, match="unknown env"):
        R.make(swm_id)
    if len(cells) == 1 and live is not None:
        assert swm_id in swm_ids_for_class(live.__name__)


def test_an_unknown_name_lists_what_is_known():
    with pytest.raises(KeyError, match="unknown env"):
        R.make("NoSuchEnv")
    with pytest.raises(KeyError, match="unknown env"):
        R.make("swm/NoSuchId-v9")
    with pytest.raises(KeyError, match="unknown env"):
        R.make("swm/PushT-v1")


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
    then = {canonical_from_frozen(r["name"]): r for r in base["registry_table"]}
    assert sorted(now) == sorted(then), "the set of registered class names changed"
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
        # ``module`` may move when unpublished shims are retired. The class is still the same
        # object, reachable at its canonical module; the old import path is gone on purpose.
        # Family/domain/render_state_dim are new explicit metadata, not a behavior change.
        now_cmp = {k: v for k, v in now[name].items() if k not in skip_keys}
        then_cmp = {k: v for k, v in row.items() if k not in skip_keys}
        assert now_cmp == then_cmp, f"the registry row for {name} changed"
        if now[name]["module"] != row["module"]:
            assert getattr(importlib.import_module(now[name]["module"]), name) is R.REGISTRY[name]
        assert R.make.__module__ and R._resolve(name) is R.REGISTRY[name]


def test_list_envs_and_info_answer_without_constructing_anything():
    names = R.list_envs()
    assert names == sorted(R.REGISTRY) and "PushT" in names
    assert R.list_envs(family="pusht") == ["PushT"]
    s = R.info("PushT")
    assert s["render_state_dim"] == 5 and s["family"] == "pusht" and s["fidelity_tier"] == "parity"
    assert s["class_name"] == "PushT" and s["id"] == "push-t"
    assert "swm_ids" not in s
    assert "pusht" in R.list_families()
    assert "spec" not in dir(snv)
    assert not hasattr(R, "spec")
    assert snv.info is R.info


def test_unpublished_historical_class_names_do_not_resolve():
    """Pre-v0 renames are not public aliases. Frozen SWM ids are not either."""
    samples = ("DMCAcrobotSwingup", "PointmazeLargeNavigate", "MujocoAnt", "PFRocketLanding")
    for old in samples:
        live = canonical_from_frozen(old)
        if live == old or live not in R.REGISTRY:
            continue
        with pytest.raises(KeyError, match="unknown env"):
            R.make(old)
        with pytest.raises(AttributeError):
            getattr(snv, old)
        assert getattr(snv, live) is R.REGISTRY[live]
    if "PyFlytRocketLanding" in R.REGISTRY:
        with pytest.raises(KeyError, match="unknown env"):
            R.info("swm/PFRocketLanding-v0")
    if "PointMazeLargeNavigate" in R.REGISTRY:
        with pytest.raises(KeyError, match="unknown env"):
            R.make("swm/OGBMaze-v0")


def test_discovery_records_its_failures_rather_than_hiding_them():
    errs = snv.import_errors()
    assert isinstance(errs, dict)
    assert all(isinstance(v, str) and v for v in errs.values())
    assert set(errs) & {*[c.__module__.rsplit(".", 1)[-1] for c in R.REGISTRY.values()]} == set(), (
        "a module cannot both have failed to import and have registered a class"
    )
