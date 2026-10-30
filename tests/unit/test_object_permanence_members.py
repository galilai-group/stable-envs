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


def test_main_construction_stops_before_output_and_requires_its_boundary(tmp_path):
    import runpy

    from scripts.export.object_permanence import source_main_prefix

    script = tmp_path / "source.py"
    script.write_text(
        "calls = []\n"
        "def ensure_dirs(): raise AssertionError('output setup')\n"
        "def clear_scene(): calls.append('clear')\n"
        "def build_base(): calls.append('base'); return {}\n"
        "def build_part(): calls.append('part'); return 7\n"
        "def animate(scene, part): calls.append(('animate', part))\n"
        "def render_png(scene): raise AssertionError('render')\n"
        "def save(): raise AssertionError('save')\n"
        "def main():\n"
        "    ensure_dirs()\n"
        "    clear_scene()\n"
        "    scene = build_base()\n"
        "    part = build_part()\n"
        "    animate(scene=scene, part=part)\n"
        "    render_png(scene)\n"
        "    save()\n"
    )
    namespace = runpy.run_path(str(script))
    with pytest.raises(ValueError, match="boundary missing"):
        source_main_prefix(namespace, "missing")
    assert namespace["calls"] == []
    source_main_prefix(namespace, "render_png")
    assert namespace["calls"] == ["clear", "base", "part", ("animate", 7)]
    assert namespace["part"] == 7


def test_hidden_annotations_do_not_render_but_cannot_appear_mid_clip(monkeypatch):
    import sys

    import numpy as np

    from scripts.export import object_permanence as exporter

    class Object:
        type = "MESH"
        hide_render = False

        def __init__(self, name, solid):
            self.name, self.solid = name, solid

        def get(self, key, default=None):
            return self.solid if key == "pb_solid" else default

        def visible_get(self):
            return True

    target, annotation = Object("target", True), Object("annotation", False)
    annotation.hide_render = True
    scene = SimpleNamespace(
        objects=[target, annotation],
        camera=True,
        frame_start=1,
        frame_end=2,
        frame_current=1,
        render=SimpleNamespace(fps=24, fps_base=1),
        collection=SimpleNamespace(hide_render=False, children=[]),
    )
    scene.frame_set = lambda frame: setattr(scene, "frame_current", frame)
    monkeypatch.setitem(
        sys.modules,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(
                view_layer=SimpleNamespace(update=lambda: None), evaluated_depsgraph_get=lambda: None
            )
        ),
    )
    monkeypatch.setattr(exporter, "validate_instances", lambda *a, **k: None)
    monkeypatch.setattr(exporter, "_camera", lambda *a, **k: {})
    monkeypatch.setattr(exporter, "_pose", lambda *a: (np.array([0, 0, 0, 1, 0, 0, 0]), np.ones(3)))
    monkeypatch.setattr(exporter, "_mesh", lambda *a: {"vertices": np.eye(3, dtype=np.float32)})
    monkeypatch.setattr(exporter, "_rgba", lambda *a, **k: [1, 0, 0, 1])
    monkeypatch.setattr(exporter, "source_geometry", lambda *a: None)
    recording = exporter.sample_scene(scene)
    assert [o["name"] for o in recording["objects"]] == ["target"]
    assert scene.frame_current == 1

    def reveal(frame):
        scene.frame_current = frame
        annotation.hide_render = frame == 1

    scene.frame_set = reveal
    with pytest.raises(ValueError, match="object set changes"):
        exporter.sample_scene(scene)
    assert scene.frame_current == 1 and annotation.hide_render
    annotation.solid = True
    with pytest.raises(ValueError, match="hidden objects"):
        exporter.sample_scene(scene)
