"""The MS-Human model download rejects a bad checksum and replaces an incomplete folder, offline."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from stable_envs.envs.mshuman import assets


@pytest.fixture
def fake_archive(monkeypatch):
    """Serve a one-file stand-in for the pinned tarball instead of downloading it."""
    xml, buf = b"<mujoco/>", io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(f"MS-Human-700-{assets._COMMIT}/{assets._XML}")
        info.size = len(xml)
        tar.addfile(info, io.BytesIO(xml))
    data = buf.getvalue()
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(data))
    monkeypatch.setattr(assets, "_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.delenv("STABLE_ENVS_MSHUMAN_HOME", raising=False)


def test_a_checksum_mismatch_raises_and_creates_nothing(fake_archive, monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        assets._ensure(tmp_path / "ms-human-700")
    assert not (tmp_path / "ms-human-700").exists()


def test_an_incomplete_folder_is_replaced_by_the_download(fake_archive, tmp_path):
    root = tmp_path / "ms-human-700"
    root.mkdir()
    (root / "half-cloned.txt").write_text("")
    assert assets._ensure(root) == root
    assert (root / assets._XML).exists() and not (root / "half-cloned.txt").exists()
