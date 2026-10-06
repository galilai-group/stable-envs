import hashlib
import subprocess

import pytest

from stable_envs.testing.provenance import verify_recorded_code


def test_historical_identity_is_separate_from_worktree_and_never_checks_out(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    path = tmp_path / "engine.py"
    path.write_text("raise RuntimeError('must never execute')\n")
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    git("add", path.name)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Recorded")
    commit = git("rev-parse", "HEAD")
    assert verify_recorded_code(files, tmp_path)["mode"] == "worktree"
    path.write_text("new implementation\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_recorded_code(files, tmp_path)
    result = verify_recorded_code(files, tmp_path, revision=commit)
    assert result["mode"] == "historical" and result["commit"] == commit
    assert path.read_text() == "new implementation\n" and git("rev-parse", "HEAD") == commit
    with pytest.raises(subprocess.CalledProcessError):
        verify_recorded_code(files, tmp_path, revision="missing-commit")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_recorded_code({path.name: "0" * 64}, tmp_path, revision=commit)


@pytest.mark.parametrize("files", [{}, {"../outside.py": "0" * 64}, {"/outside.py": "0" * 64}])
def test_missing_or_unsafe_identity_cannot_pass(files, tmp_path):
    with pytest.raises(ValueError):
        verify_recorded_code(files, tmp_path)
