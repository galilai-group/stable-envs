"""Member routing must preserve identities and independent reference boundaries."""

import json
from types import SimpleNamespace

import pytest

from scripts.export import batch_object_permanence as batch
from scripts.object_permanence_member import Member
from scripts.testing.freeze_object_permanence_member import directory


def test_member_names_preserve_legacy_nominal_and_use_script_indices(tmp_path):
    assert Member("G02").clip == "g02-nominal"
    entry = Member("G02", 2)
    assert entry.clip == "g02-member2"
    assert directory(tmp_path, entry, "stable-envs/opaque-pixel-rays-v1") == (
        tmp_path / "g02/opaque_pixel_rays_v1/member2"
    )
    Member("G02").check_source({"task": "G02"})
    for source in (
        {"task": "G02"},
        {"task": "G03", "member": 2},
        {"task": "G02", "member": 2, "entry_profile": "diversity"},
    ):
        with pytest.raises(ValueError, match="identity differs"):
            entry.check_source(source)


@pytest.fixture
def source(tmp_path, monkeypatch):
    task = tmp_path / "source/object_permanence/generator/tasks/G02_test"
    task.mkdir(parents=True)
    names = [f"script{i}.py" for i in range(3)]
    for name in names:
        (task / name).write_text("def main(): pass\n")
    # Global members are intentionally different from zero-based script indices.
    (task / "task.json").write_text(json.dumps({"id": "G02", "members": [2, 3, 4], "member_scripts": names}))
    monkeypatch.setattr(batch, "check_source", lambda root: None)
    return tmp_path / "source"


def test_all_stages_route_the_same_identity_and_isolate_blender_from_cuda(source, tmp_path, monkeypatch):
    calls = []

    def complete(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="passed", stderr="")

    monkeypatch.setattr(batch.subprocess, "run", complete)
    result = batch.run(
        source,
        tmp_path / "data",
        tmp_path / "report.json",
        tasks="G02",
        members="2",
        stage="all",
        oracle_root=tmp_path / "oracle",
        blender_python="bpy-python",
        render_python="cuda-python",
        visibility_contract="stable-envs/opaque-pixel-rays-v1",
    )
    assert result["counts"] == {"verified": 1}
    assert [c[0] for c in calls] == ["bpy-python"] * 3 + ["cuda-python"]
    assert all(c[c.index("--member") + 1] == "2" and c[c.index("--profile") + 1] == "nominal" for c in calls)
    assert calls[2][2] == "scripts.testing.freeze_object_permanence_member"
    assert calls[3][2] == "scripts.testing.check_object_permanence_member"


def test_failed_validation_cannot_be_certified(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        batch.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="mask mismatch")
    )
    result = batch.run(
        source,
        tmp_path / "data",
        tmp_path / "report.json",
        tasks="G02",
        members="1",
        stage="validate",
        oracle_root=tmp_path / "oracle",
        visibility_contract="stable-envs/opaque-pixel-rays-v1",
    )
    assert result["counts"] == {"failed": 1}
    assert result["jobs"][0]["validation"] == "failed"
