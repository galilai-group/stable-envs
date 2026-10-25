"""Native realization identities bind source, recording and independent oracle."""

import hashlib
import json

import pytest

from scripts.export import object_permanence as exporter
from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS


@pytest.mark.parametrize("task", ["g98", "g132"])
def test_native_scene_recording_and_oracle_bind_one_hash_checked_source_realization(task):
    directory = FIXTURES / f"source/object_permanence/{task}/nominal"
    path = directory / "source.blend"
    native = json.loads((directory / "manifest.json").read_text())
    candidate = json.loads((RECORDINGS / f"clips/{task}-nominal/manifest.json").read_text())
    oracle = json.loads(
        (FIXTURES / f"semantics/object_permanence/{task}/opaque_pixel_rays_v1/nominal/manifest.json").read_text()
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == native["sha256"]
    assert path.stat().st_size < 1000 * 1024
    assert native["frames"] == [1, 120]
    assert native["source_commit"] == candidate["source"]["commit"] == oracle["source_commit"]
    assert native["source_script_sha256"] == candidate["source"]["script_sha256"] == oracle["source_script_sha256"]
    identity = candidate["source"]["realization"]
    assert identity == oracle["source_realization"]
    assert identity["sha256"] == native["sha256"]
    assert identity["manifest_sha256"] == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    assert native["source_preparation_recipe"] == candidate["source"]["preparation_recipe"]
    assert native["source_license"] == "CC-BY-NC-4.0"


def test_tampered_native_scene_is_rejected_before_blender_loading(tmp_path, monkeypatch):
    task = "G98"
    root = tmp_path / "source"
    root.mkdir()
    script = root / "scene.py"
    script.write_text("# upstream source\n")
    taskfile = root / "task.json"
    taskfile.write_text("{}")
    directory = tmp_path / "native/g98/nominal"
    directory.mkdir(parents=True)
    path = directory / "source.blend"
    path.write_bytes(b"native Blender scene")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metadata = {
        "schema": "stable-envs/native-source-realization-v1",
        "task": task,
        "member": 0,
        "source_commit": exporter.SOURCE_COMMIT,
        "source_script": "scene.py",
        "source_script_sha256": digest(script),
        "source_task_manifest_sha256": digest(taskfile),
        "source_preparation_recipe": {"build": "build_scene_by_kind", "animate": "animate_scene_by_kind"},
        "file": "source.blend",
        "sha256": digest(path),
    }
    (directory / "manifest.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(exporter, "NATIVE_SCENES", tmp_path / "native")
    assert exporter.source_realization(root, task)["sha256"] == digest(path)
    path.write_bytes(b"changed scene with same filename")
    with pytest.raises(ValueError, match="identity/checksum"):
        exporter.source_realization(root, task)


def test_native_reference_preparation_does_not_import_candidate_export_or_runtime():
    import inspect

    from scripts.testing.freeze_object_permanence import reference_source_realization

    source = inspect.getsource(reference_source_realization)
    assert "scripts.export" not in source and "stable_envs" not in source


def test_different_native_realization_is_rejected_before_rendering(monkeypatch):
    from scripts.testing import check_object_permanence_rendering as checker

    directory = FIXTURES / "semantics/object_permanence/g98/opaque_pixel_rays_v1/nominal"
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["source_realization"]["sha256"] = "0" * 64
    monkeypatch.setattr(checker, "load_geometry_directory", lambda _: (manifest, {}))
    monkeypatch.setattr(checker, "RigidSceneRenderer", lambda *a, **kw: pytest.fail("renderer must not be created"))
    with pytest.raises(ValueError, match="native realizations differ"):
        checker.check(RECORDINGS, directory, "g98", visibility_contract="stable-envs/opaque-pixel-rays-v1")
