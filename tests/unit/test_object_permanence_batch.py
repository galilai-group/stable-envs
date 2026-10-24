"""Batch plans never silently certify or execute unaudited source members."""

import json
from types import SimpleNamespace

import pytest

from scripts.export import batch_object_permanence as batch


@pytest.fixture
def source(tmp_path, monkeypatch):
    for name, members in (("G34", ["pilot.py"]), ("G99", ["one.py", "two.py"])):
        directory = tmp_path / "object_permanence/generator/tasks" / name
        directory.mkdir(parents=True)
        for member in members:
            (directory / member).write_text("def main(): pass\n")
        (directory / "task.json").write_text(json.dumps({"id": name, "member_scripts": members}))
    monkeypatch.setattr(batch, "check_source", lambda root: None)
    return tmp_path


def test_plan_lists_every_member_without_running_blender(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not execute source")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    output = tmp_path / "data"
    result = batch.run(source, output, tmp_path / "plan.json", plan_only=True)
    assert result["counts"] == {"ready-for-runtime-check": 1, "unsupported": 2}
    assert not output.exists()
    assert [job["member"] for job in result["jobs"]] == [0, 0, 1]


def test_unknown_task_is_rejected_before_any_source_execution(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid selection must not execute source")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="unknown source tasks"):
        batch.run(source, tmp_path / "data", tmp_path / "report.json", tasks="G34,G00")


def test_native_failure_is_reported_and_unaudited_jobs_are_not_executed(source, tmp_path, monkeypatch):
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=-11, stdout="", stderr="source process failed")

    monkeypatch.setattr(batch.subprocess, "run", failed)
    result = batch.run(source, tmp_path / "data", tmp_path / "report.json", blender_python="blender-python")
    assert len(calls) == 1 and calls[0][0] == "blender-python"
    assert result["counts"] == {"failed": 1, "unsupported": 2}
    assert result["jobs"][0]["detail"] == "source process failed"


@pytest.mark.parametrize(
    "code,expected",
    [
        ("cam.data.type = 'ORTHO'", "orthographic-camera-review"),
        ("obj.keyframe_insert(data_path='scale')", "geometry-or-scale-review"),
        ("mat = make_mat(alpha=0.5)", "appearance-or-visibility-review"),
        ("obj.hide_render = True", "appearance-or-visibility-review"),
        ("bpy.ops.rigidbody.object_add()", "physics-bake-review"),
        ("obj.keyframe_insert(data_path='location')", "rigid-candidate-needs-entry-audit"),
        ("obj.modifiers.new(name='window', type='BOOLEAN')", "rigid-candidate-needs-entry-audit"),
    ],
)
def test_capability_triage_is_conservative_and_never_certifies_an_unaudited_member(code, expected):
    import ast

    from scripts.export.object_permanence import capability_group

    assert capability_group(ast.parse(code)) == expected
    assert capability_group(ast.parse(code), audited=True) == "audited-nominal-runtime-gate"


def test_lossless_reference_parts_split_high_entropy_data_without_exceeding_file_limit(tmp_path):
    import hashlib

    import numpy as np

    from scripts.testing.freeze_object_permanence import write_depth_parts

    bits = np.random.default_rng(17).integers(0, 2**32, (31, 128, 128), dtype=np.uint32)
    parts = write_depth_parts(tmp_path, bits)
    restored, end = [], 0
    assert len(parts) > 2  # A high-entropy 30-frame block exceeds the file budget.
    for part in parts:
        path = tmp_path / part["file"]
        assert path.stat().st_size <= 1000 * 1024
        assert hashlib.sha256(path.read_bytes()).hexdigest() == part["sha256"]
        assert part["start"] == end and 0 < part["stop"] - end <= 30
        with np.load(path, allow_pickle=False) as archive:
            value = archive["depth_xor"]
        assert list(value.shape) == part["shape"] and str(value.dtype) == part["dtype"]
        restored.append(value)
        end = part["stop"]
    assert end == len(bits)
    np.testing.assert_array_equal(np.concatenate(restored), bits)


def test_known_source_fidelity_failure_is_not_reported_ready_or_executed(source, tmp_path, monkeypatch):
    directory = source / "object_permanence/generator/tasks/G63"
    directory.mkdir()
    (directory / "task.json").write_text(json.dumps({"id": "G63", "member_scripts": ["comb.py"]}))
    (directory / "comb.py").write_text("def main(): pass\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("known source fidelity failure must not be batch executed")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    for plan_only in (True, False):
        result = batch.run(source, tmp_path / "data", tmp_path / "plan.json", tasks="G63", plan_only=plan_only)
        assert result["counts"] == {"unsupported": 1}
        assert result["jobs"][0]["capability_group"] == "source-fidelity-review"
        assert "source fidelity deferred" in result["jobs"][0]["reason"]
