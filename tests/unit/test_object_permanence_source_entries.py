"""Offline legacy entry extraction must preserve animation and avoid source I/O."""

import pytest

from scripts.export.object_permanence import source_prefix


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
