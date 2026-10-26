"""Member routing must preserve identities and independent reference boundaries."""

import json
from types import SimpleNamespace

import pytest

from scripts.export import batch_object_permanence as batch
from scripts.object_permanence_member import Member
from scripts.testing.freeze_object_permanence_member import directory


@pytest.mark.parametrize(
    "task,member,profile",
    [("g02", 0, "nominal"), ("G02", True, "nominal"), ("G02", -1, "nominal"), ("G02", 1, "diversity")],
)
def test_unaudited_identity_rejected(task, member, profile):
    with pytest.raises(ValueError):
        Member(task, member, profile)


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


def test_member_plan_is_explicit_contract_scoped_and_executes_nothing(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planning executed a process")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    for contract, expected in ((None, "unsupported"), ("stable-envs/opaque-pixel-rays-v1", "ready-for-runtime-check")):
        result = batch.run(
            source,
            tmp_path / "data",
            tmp_path / "plan.json",
            tasks="G02",
            members="1,2",
            plan_only=True,
            visibility_contract=contract,
        )
        assert [j["clip"] for j in result["jobs"]] == ["g02-member1", "g02-member2"]
        assert result["counts"] == {expected: 2}
    assert not (tmp_path / "data").exists()


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


def test_invalid_profile_and_empty_member_selection_fail_before_execution(source, tmp_path):
    with pytest.raises(ValueError, match="profile"):
        batch.run(source, tmp_path / "data", tmp_path / "r.json", profile="surface")
    with pytest.raises(ValueError, match="unknown source members"):
        batch.run(source, tmp_path / "data", tmp_path / "r.json", members="1,9")
    with pytest.raises(ValueError, match="oracle_root"):
        batch.run(source, tmp_path / "data", tmp_path / "r.json", stage="validate")


def test_oracle_identity_mismatch_rejected_without_gpu_or_candidate_geometry(tmp_path):
    from scripts.testing.check_object_permanence_member import check

    entry = Member("G02", 1)
    contract = "stable-envs/opaque-pixel-rays-v1"
    path = directory(tmp_path, entry, contract)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"task": "G02", "member": 2, "visibility_contract": contract}))
    with pytest.raises(ValueError, match="identity differs"):
        check(tmp_path / "missing-recordings", tmp_path, entry, contract)


def test_existing_optical_nominal_plan_accepts_its_explicit_contract(source, tmp_path):
    path = source / "object_permanence/generator/tasks/G01_test"
    path.mkdir()
    (path / "scene.py").write_text("def main(): pass\n")
    (path / "task.json").write_text(json.dumps({"id": "G01", "member_scripts": ["scene.py"]}))
    result = batch.run(
        source,
        tmp_path / "data",
        tmp_path / "report.json",
        tasks="G01",
        members="0",
        stage="all",
        oracle_root=tmp_path / "oracle",
        plan_only=True,
        visibility_contract="stable-envs/sampled-optical-rays-v1",
    )
    assert result["counts"] == {"ready-for-runtime-check": 1}


@pytest.mark.parametrize("task,total,accepted", [("G04", 2, {1}), ("G05", 7, {2, 4, 5, 6})])
def test_next_member_batch_keeps_hidden_marker_and_multistage_scale_entries_unregistered(
    tmp_path, monkeypatch, task, total, accepted
):
    path = tmp_path / f"source/object_permanence/generator/tasks/{task}_test"
    path.mkdir(parents=True)
    names = [f"scene{i}.py" for i in range(total)]
    for name in names:
        (path / name).write_text("def main(): pass\n")
    (path / "task.json").write_text(json.dumps({"id": task, "member_scripts": names}))
    monkeypatch.setattr(batch, "check_source", lambda root: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("member plan executed source")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    result = batch.run(
        tmp_path / "source",
        tmp_path / "data",
        tmp_path / "report.json",
        tasks=task,
        members=",".join(str(i) for i in range(1, total)),
        stage="all",
        oracle_root=tmp_path / "oracles",
        plan_only=True,
        visibility_contract="stable-envs/opaque-pixel-rays-v1",
    )
    assert {j["member"] for j in result["jobs"] if j["status"] == "ready-for-runtime-check"} == accepted
    assert {j["member"] for j in result["jobs"] if j["status"] == "unsupported"} == set(range(1, total)) - accepted
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "task,member,version,total",
    [
        ("G06", 1, "v1", 3),
        ("G06", 2, "v1", 3),
        ("G07", 1, "v1", 2),
        ("G08", 1, "v2", 2),
        ("G09", 1, "v2", 3),
        ("G09", 2, "v2", 3),
    ],
)
def test_perspective_and_orthographic_members_require_their_audited_contract(
    tmp_path, monkeypatch, task, member, version, total
):
    from scripts.testing.freeze_object_permanence_member import supports

    path = tmp_path / f"source/object_permanence/generator/tasks/{task}_test"
    path.mkdir(parents=True)
    names = [f"scene{i}.py" for i in range(total)]
    for name in names:
        (path / name).write_text("def main(): pass\n")
    (path / "task.json").write_text(json.dumps({"id": task, "member_scripts": names}))
    monkeypatch.setattr(batch, "check_source", lambda root: None)

    def forbidden(*args, **kwargs):
        raise AssertionError("planning must not execute source or render")

    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    for selected in ("v1", "v2"):
        contract = f"stable-envs/opaque-pixel-rays-{selected}"
        result = batch.run(
            tmp_path / "source",
            tmp_path / "data",
            tmp_path / "report.json",
            tasks=task,
            members=str(member),
            stage="all",
            oracle_root=tmp_path / "oracles",
            plan_only=True,
            visibility_contract=contract,
        )
        expected = "ready-for-runtime-check" if selected == version else "unsupported"
        assert result["counts"] == {expected: 1}
        assert supports(Member(task, member), contract) == (selected == version)
    assert not (tmp_path / "data").exists()
