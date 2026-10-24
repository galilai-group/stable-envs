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
