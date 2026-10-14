"""The registry knows stable-envs class names; the 38 SWM ids live in the test-only crosswalk.

Two oracles:

  * the SWM ids are asserted against the frozen SWM-id -> cell mapping
    (``tests/fixtures/swm/frozen_cells.json``), from which the crosswalk is read;
  * the canonical class names are asserted against the frozen registry baseline
    (``tests/fixtures/regression/references/registry_baseline.json``).

Which env modules import depends on what is installed, so the baseline's import-failure set is compared
first and the class-by-class comparison is skipped if this interpreter differs.
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
    canonical_from_frozen,
    canonical_names,
    class_for_frozen_cell,
    swm_ids_for_class,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "tests" / "fixtures" / "swm" / "frozen_cells.json"
BASELINE = ROOT / "tests" / "fixtures" / "regression" / "references" / "registry_baseline.json"
SWM_IDS_IN_MATRIX = sorted({r["swm_id"] for r in json.loads(MATRIX.read_text())["rows"]})
#: Families the baseline recorded that this package does not ship.
UNSHIPPED_MODULES = frozenset(
    {
        "carracing",
        "crafter",
        "griddly",
        "highway",
        "humanoidbench",
        "jumanji",
        "lunarlander",
        "maze2d",
        "metaworld",
        "minigrid",
        "minihack",
        "mujoco_locomotion",
        "overcooked",
        "pgx",
        "procgen",
        "robocasa",
        "sokoban",
    }
)
UNSHIPPED_BASELINE_MODULES = frozenset(f"stable_envs.envs.{name}" for name in UNSHIPPED_MODULES)


def test_the_swm_crosswalk_is_the_frozen_mapping_row_for_row():
    rows = json.loads(MATRIX.read_text())["rows"]
    assert len(rows) == 38, "the oracle is 38 SWM ids; this file is not the accepted mapping"
    expected = {r["swm_id"]: tuple(r["stable_envs_cells"]) for r in rows}
    assert SWM_IDS == expected
    assert len(SWM_IDS) == 38
    assert all(SWM_IDS.values()), "every id produces pixels"
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
    base_errors = {k for k in base["import_errors"] if k not in UNSHIPPED_MODULES}
    if base_errors != set(snv.import_errors()):
        pytest.skip(
            "this interpreter imports a different set of env families than the one the "
            f"baseline was recorded on: baseline {sorted(base_errors)} vs "
            f"{sorted(snv.import_errors())}"
        )
    now = {r["class_name"]: r for r in R.registry_table()}
    then = {
        canonical_from_frozen(r["name"]): r
        for r in base["registry_table"]
        if r["module"] not in UNSHIPPED_BASELINE_MODULES
    }
    assert sorted(set(now) - {"PushTDiscrete"}) == sorted(then), "unexpected registered class change"
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
        # The module path and the explicit family/domain/width metadata may differ; the class must not.
        now_cmp = {k: v for k, v in now[name].items() if k not in skip_keys}
        then_cmp = {k: v for k, v in row.items() if k not in skip_keys}
        assert now_cmp == then_cmp, f"the registry row for {name} changed"
        if now[name]["module"] != row["module"]:
            assert getattr(importlib.import_module(now[name]["module"]), name) is R.REGISTRY[name]
        assert R._resolve(name) is R.REGISTRY[name]


def test_list_envs_and_info_answer_without_constructing_anything():
    names = R.list_envs()
    assert names == sorted(R.REGISTRY) and "PushT" in names
    assert R.list_envs(family="pusht") == ["PushT", "PushTDiscrete"]
    s = R.info("PushT")
    assert s["render_state_dim"] in (0, None) and s["family"] == "pusht" and s["fidelity_tier"] == "parity"
    assert s["reference_id"] == "swm/PushT-v1"
    assert s["class_name"] == "PushT" and s["id"] == "push-t"
    assert "swm_ids" not in s
    assert "pusht" in R.list_families()
    assert "spec" not in dir(snv)
    assert not hasattr(R, "spec")
    assert snv.info is R.info


def test_renamed_class_names_do_not_resolve():
    """Renamed classes and frozen SWM ids are not public aliases."""
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
