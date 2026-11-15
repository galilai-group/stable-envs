"""A StateSchema stores named fields in group leaves and reads them back exactly, from the State alone."""

from pathlib import Path

import numpy as np
import pytest
import torch

from stable_envs import get_field, schema_of, set_field
from stable_envs._core.state_layout import StateLayout
from stable_envs._core.state_schema import Field, StateSchema, raw_field_access, unfold
from stable_envs._core.tensordicts import stack_leaves, to_device

ROOT = Path(__file__).resolve().parents[2]

#: Adopted families: the top-level State keys that only the schema may index, and the files to scan besides the
#: family package and the tests named after the family.
ADOPTED: dict[str, tuple[tuple[str, ...], list[Path]]] = {
    # Cube and Scene share stable_envs/envs/ogbench with the mazes, so the key names no package and the files are
    # listed. "variation" is also the reset option key, so it is not a section here.
    "ogbench_manipulation": (
        ("compatibility", "physics", "controller", "output", "task", "lifecycle", "scene", "model", "rng"),
        [
            *sorted((ROOT / "stable_envs/envs/ogbench").glob("manipulation*.py")),
            *(
                ROOT / "stable_envs/envs/ogbench/simulation" / name
                for name in (
                    "compatibility.py",
                    "manipulation.py",
                    "manipulation_batch.py",
                    "manipulation_control.py",
                    "manipulation_process.py",
                    "manipulation_warp.py",
                    "render_state.py",
                )
            ),
            *(
                ROOT / "tests" / name
                for name in (
                    "unit/test_ogb_source_reset.py",
                    "unit/test_ogbench_render_state.py",
                    "unit/test_ogbench_state_export.py",
                    "unit/test_ogbench_validation_columns.py",
                    "unit/test_ogbench_lane_pool.py",
                    "helpers/reference/gates/ogb_variation.py",
                    "helpers/reference/ogbench_state_export.py",
                )
            ),
        ],
    ),
}

FIELDS = [
    Field("t/pose", (3,), torch.float64, "t/float64"),
    Field("t/grid", (2, 2), torch.float64, "t/float64"),
    Field("t/time", (), torch.float64, "t/float64"),
    Field("t/target", (2,), torch.float32, "t/float32"),
    Field("t/count", (), torch.int64, "t/int64"),
    Field("t/seed", (2,), torch.int32, "t/int32"),
    Field("t/flag", (), torch.bool, "t/bool"),
    Field("t/color", (3,), torch.uint8, "t/uint8"),
    Field("history/actions", None, torch.float64, None),
]


def _schema():
    return StateSchema("Test", FIELDS)


def _values(n):
    """Named values that use the whole range of each dtype, so a lossy round trip shows."""
    rng = np.random.default_rng(0)
    return {
        "t/pose": torch.from_numpy(rng.normal(size=(n, 3)) * 1e300),
        "t/grid": torch.from_numpy(rng.normal(size=(n, 2, 2))),
        "t/time": torch.from_numpy(rng.normal(size=n)).nextafter(torch.tensor(np.inf, dtype=torch.float64)),
        "t/target": torch.from_numpy(rng.normal(size=(n, 2)).astype(np.float32)),
        "t/count": torch.from_numpy(rng.integers(-(2**62), 2**62, n)),
        "t/seed": torch.from_numpy(rng.integers(-(2**31), 2**31 - 1, (n, 2)).astype(np.int32)),
        "t/flag": torch.from_numpy(rng.integers(0, 2, n).astype(bool)),
        "t/color": torch.from_numpy(rng.integers(0, 256, (n, 3)).astype(np.uint8)),
        "history/actions": torch.from_numpy(rng.normal(size=(n, 4, 2))),
    }


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def test_declaration_maps_fields_to_slices_of_groups():
    schema = _schema()
    assert schema.order == ("t/float64", "t/float32", "t/int64", "t/int32", "t/bool", "t/uint8", "history/actions")
    assert schema.groups["t/float64"].size == 3 + 4 + 1
    assert schema.slices["t/grid"].start == 3 and schema.slices["t/grid"].stop == 7
    assert schema.slices["t/time"].shape == ()
    assert schema.kept == {"history/actions": torch.float64}
    assert schema.leaves()[0] == ("t/float64", (8,), torch.float64)
    assert schema.leaves()[-1] == ("history/actions", None, torch.float64)
    assert schema == _schema() and hash(schema) == hash(_schema())
    for bad in (
        [Field("t/a", (1,), torch.float64, "g"), Field("t/b", (1,), torch.float32, "g")],
        [Field("t/a", (1,), torch.float64, "g"), Field("t/a", (1,), torch.float64, "g")],
        [Field("t.a", (1,), torch.float64, "g")],
        [Field("t/a", None, torch.float64, "g")],
        [Field("g", (1,), torch.float64, "g")],
    ):
        with pytest.raises(ValueError):
            StateSchema("Bad", bad)


def test_new_builds_one_zero_tensor_per_group():
    schema = _schema()
    state = schema.new(3)
    assert state.batch_size == (3,)
    assert sorted("/".join(k) for k in state.keys(True, True)) == sorted(schema.order)
    assert state["t", "float64"].shape == (3, 8) and state["history", "actions"].shape == (3, 0)
    assert all(not v.any() for v in state.values(True, True))
    assert schema.new((2, 5))["t", "uint8"].shape == (2, 5, 3)
    schema.validate(schema.new(4))


def test_fold_get_set_round_trip_exactly_for_every_dtype():
    schema = _schema()
    values = _values(5)
    state = schema.fold(values, [5])
    schema.validate(state, finite=True)
    for name, value in values.items():
        _same(schema.get(state, name), value)
        _same(get_field(state, name), value)
        _same(get_field(state, name.replace("/", ".")), value)
    named = unfold(state)
    for name, value in values.items():
        _same(named.get(tuple(name.split("/"))), value)
    _same(schema.fold(named, [5])["t", "int64"], state["t", "int64"])
    again = schema.new(5)
    for name, value in values.items():
        if name in schema.kept:
            again.set(tuple(name.split("/")), value.clone())
        else:
            set_field(again, name, value)
    assert (again == state).all()


def test_set_state_round_trip_through_a_layout_export_is_exact():
    schema = _schema()
    values = _values(4)
    layout = StateLayout(schema.leaves(), 4)
    for source in (values, {k: v.numpy() for k, v in values.items()}):
        packed = schema.pack(source)
        state = layout.export({**packed, "history/actions": source["history/actions"]}, "cpu")
        assert schema_of(state) == schema
        schema.validate(state)
        for name, value in values.items():
            _same(get_field(state, name), value)


def test_views_share_memory_and_writes_change_the_state():
    schema = _schema()
    state = schema.fold(_values(3), [3])
    views = schema.views(state, ["t/grid", "t/flag"])
    views["t/grid"][1, 0, 1] = 7.0
    assert float(state["t", "float64"][1, 4]) == 7.0
    buffers = schema.buffers(state)
    buffers["t/uint8"][2] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    _same(get_field(state, "t/color")[2], torch.tensor([1, 2, 3], dtype=torch.uint8))
    host = schema.views(state, ["t/pose"], numpy=True)["t/pose"]
    host[0, 0] = -1.0
    assert float(get_field(state, "t/pose")[0, 0]) == -1.0
    set_field(state, "t/count", 4)
    assert get_field(state, "t/count").tolist() == [4, 4, 4]
    set_field(state, "t/target", np.float32(0.5))
    assert get_field(state, "t/target").eq(0.5).all()


def test_batch_indexing_slicing_stacking_and_concatenation_keep_fields_readable():
    schema = _schema()
    values = _values(6)
    state = schema.fold(values, [6])
    for index in (2, -1):
        row = state[index]
        assert schema_of(row) == schema
        for name, value in values.items():
            _same(get_field(row, name), value[index])
    mask = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)
    for part, rows in ((state[1:4], [1, 2, 3]), (state[torch.tensor([5, 0, 3])], [5, 0, 3]), (state[mask], [0, 2, 4])):
        for name, value in values.items():
            _same(get_field(part, name), value[rows])
    later = schema.fold(_values(6), [6])
    set_field(later, "t/count", 9)
    for stacked in (torch.stack([state, later], 1), stack_leaves([state, later])):
        assert stacked.batch_size == (6, 2)
        _same(get_field(stacked, "t/grid"), torch.stack([values["t/grid"], values["t/grid"]], 1))
        _same(get_field(stacked, "t/count")[:, 1], torch.full((6,), 9))
        _same(get_field(stacked[:, 0], "t/color"), values["t/color"])
    joined = torch.cat([state, later], 0)
    _same(get_field(joined, "t/seed"), torch.cat([values["t/seed"], values["t/seed"]]))
    _same(get_field(joined, "t/count")[6:], torch.full((6,), 9))


def test_clone_and_device_moves_keep_every_field():
    schema = _schema()
    values = _values(3)
    state = schema.fold(values, [3])
    copy = state.clone()
    set_field(copy, "t/flag", True)
    _same(get_field(state, "t/flag"), values["t/flag"])
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for moved in (state.to(device), to_device(state, device)):
            assert schema_of(moved) == schema
            for name, value in values.items():
                _same(get_field(moved, name).cpu(), value)
        built = schema.fold(values, [3], device)
        assert built["t", "float64"].device.type == torch.device(device).type


def test_validation_names_the_wrong_name_shape_or_dtype():
    schema = _schema()
    state = schema.fold(_values(3), [3])
    with pytest.raises(KeyError, match="t/missing"):
        get_field(state, "t/missing")
    with pytest.raises(KeyError, match="t/missing"):
        schema.set(state, "t/missing", 1)
    with pytest.raises(TypeError, match="t/count"):
        set_field(state, "t/count", 1.5)
    with pytest.raises(TypeError, match="t/color"):
        set_field(state, "t/color", 256)
    set_field(state, "t/color", 255)
    with pytest.raises(TypeError, match="t/flag"):
        set_field(state, "t/flag", torch.ones(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="t/pose"):
        set_field(state, "t/pose", torch.ones(3, 2, dtype=torch.float64))
    values = _values(3)
    with pytest.raises(KeyError, match="t/color"):
        schema.fold({k: v for k, v in values.items() if k != "t/color"}, [3])
    with pytest.raises(ValueError, match="t/grid"):
        schema.fold({**values, "t/grid": values["t/grid"].reshape(3, 4)}, [3])
    with pytest.raises(TypeError, match="t/target"):
        schema.fold({**values, "t/target": values["t/target"].double()}, [3])
    cases = {
        "t/int64": lambda s: s.set(("t", "int64"), s["t", "int64"].double()),
        "t/uint8": lambda s: s.set(("t", "uint8"), torch.zeros(3, 4, dtype=torch.uint8)),
        "t/bool": lambda s: s.del_(("t", "bool")),
        "history/actions": lambda s: s.set(("history", "actions"), s["history", "actions"].float()),
        "outside its schema": lambda s: s.set(("t", "other"), torch.zeros(3)),
    }
    for message, change in cases.items():
        bad = state.clone()
        change(bad)
        with pytest.raises(ValueError, match=message):
            schema.validate(bad)
    with pytest.raises(TypeError):
        schema.validate({"t": 1})
    bad = state.clone()
    get_field(bad, "t/time")[1] = float("nan")
    schema.validate(bad)
    with pytest.raises(ValueError, match="t/time"):
        schema.validate(bad, finite=True)


def test_schema_lookup_from_the_state_alone():
    schema = _schema()
    state = schema.fold(_values(2), [2])
    assert schema_of(state) == schema
    assert schema_of(state.select(("t", "uint8"))) == schema
    assert schema_of(state.exclude("t")) is None
    named = unfold(state)
    assert schema_of(named) is None
    _same(get_field(named, "t/grid"), get_field(state, "t/grid"))
    assert unfold(named) is named
    # Another layout with a group of the same key, size and dtype: a State holding only that group is ambiguous.
    other = StateSchema(
        "Other",
        [
            Field("o/x", (1,), torch.float64, "o/float64"),
            Field("t/a", (1,), torch.uint8, "t/uint8"),
            Field("t/b", (2,), torch.uint8, "t/uint8"),
        ],
    )
    assert schema_of(other.new(2)) == other
    with pytest.raises(ValueError, match="several StateSchemas"):
        schema_of(state.select(("t", "uint8")))
    assert schema_of(state) == schema


def test_spec_describes_the_group_leaves():
    spec = _schema().spec("cpu")
    assert spec["t", "float64"].shape == (8,) and spec["t", "uint8"].dtype == torch.uint8


@pytest.mark.parametrize("family", sorted(ADOPTED))
def test_adopted_families_touch_state_fields_only_through_the_schema(family):
    """Raw indexing of a State leaf by key breaks when the layout changes; ``get_field`` and the schema own it."""
    sections, extra = ADOPTED[family]
    paths = [
        *sorted((ROOT / "stable_envs" / "envs" / family).rglob("*.py")),
        *sorted((ROOT / "tests").rglob(f"*{family}*.py")),
        *extra,
    ]
    found = raw_field_access(paths, sections)
    assert not found, f"raw {family} State indexing: {found}"


def test_raw_field_access_finds_the_indexing_forms(tmp_path):
    source = tmp_path / "code.py"
    source.write_text(
        "\n".join(
            [
                's["current", "x"]',
                "s['current/x']",
                's.get(("current", "x"))',
                's.select("current")',
                'get_field(s, "current/x")',
                'unfold(s)["current", "x"]',
                's["currently"]',
            ]
        )
    )
    assert [line.rsplit(":", 1)[1] for line in raw_field_access([source], ["current"])] == ["1", "2", "3", "4"]
