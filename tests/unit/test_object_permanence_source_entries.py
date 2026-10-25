"""Offline legacy entry extraction must preserve animation and avoid source I/O."""

import pytest

from scripts.export.object_permanence import source_execution_context, source_prefix
from scripts.testing.freeze_object_permanence import reference_execution_context


def test_legacy_prefix_preserves_all_frames_without_running_output_setup_or_tail(tmp_path):
    script = tmp_path / "source.py"
    output = tmp_path / "forbidden"
    script.write_text(
        f"import os\nOUT_DIR = {str(output)!r}\nos.makedirs(OUT_DIR, exist_ok=True)\n"
        "frames = []\nfor frame in range(1, 121):\n    frames.append((frame, frame * 2))\n"
        "def render_png():\n    raise AssertionError('source rendering executed')\n"
        "render_png()\nraise AssertionError('source output tail executed')\n"
    )
    result = source_prefix(script, "render_png")
    assert result["frames"] == [(i, i * 2) for i in range(1, 121)]
    assert not output.exists()
    assert "render_png" not in result


@pytest.mark.parametrize("text", ["value = 1\n", "import os\nos.makedirs('unexpected')\ndef render_png(): pass\n"])
def test_legacy_prefix_rejects_missing_boundary_or_unaudited_output_setup(tmp_path, text):
    script = tmp_path / "source.py"
    script.write_text(text)
    with pytest.raises(ValueError, match="boundary missing|unaudited source output setup"):
        source_prefix(script, "render_png")


@pytest.mark.parametrize("task", ["G34", "G90"])
def test_source_context_does_not_change_unregistered_or_legacy_namespaces(tmp_path, task):
    script = tmp_path / "missing.py"
    assert source_execution_context(tmp_path, task, script) is None
    assert reference_execution_context(script, task) is None


@pytest.mark.parametrize("change", ["values", "source_bindings", "factorial", "universal_viewpoint", "rewrite"])
def test_source_context_refuses_non_nominal_parameters_or_source_rewriting(tmp_path, change):
    script = tmp_path / "object_permanence/generator/tasks/G44_scene/source.py"
    script.parent.mkdir(parents=True)
    script.write_text("factor = DIVERSITY.get('scale', 1.0)\n")
    settings = {"values": {}, "source_bindings": {}, "factorial": None, "universal_viewpoint": False}
    if change != "rewrite":
        settings[change] = {"unexpected": 1}
    module = tmp_path / "object_permanence/generator/core/diversity.py"
    module.parent.mkdir()
    module.write_text(
        f"def load_script_diversity(*args, **kwargs): return {settings!r}\n"
        f"def apply_source_bindings(text, settings): return (text + {('changed' if change == 'rewrite' else '')!r}, {{}})\n"
    )
    with pytest.raises(ValueError, match="must not vary or rewrite"):
        source_execution_context(tmp_path, "G44", script)
    with pytest.raises(ValueError, match="unmodified surface-profile"):
        reference_execution_context(script, "G44")


def test_source_reference_context_mismatch_is_rejected_before_constructing_a_renderer(monkeypatch):
    from copy import deepcopy

    from scripts.testing import check_object_permanence_rendering as checker
    from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT
    from tests.helpers.reference.object_permanence_recordings import RECORDINGS, load_ray_geometry

    manifest, arrays = load_ray_geometry("g44")
    mismatched = deepcopy(manifest)
    mismatched["source_execution_context"]["globals"]["DIVERSITY"] = {"ball_radius_scale": 1.18}
    monkeypatch.setattr(checker, "load_geometry_directory", lambda _: (mismatched, arrays))
    with pytest.raises(ValueError, match="execution contexts differ"):
        checker.check(RECORDINGS, "unused", "g44", visibility_contract=CONTRACT)
