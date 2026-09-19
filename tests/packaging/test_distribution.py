"""Packaging and CI contracts: metadata, extras, docs config, and workflow safety."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib
yaml = pytest.importorskip("yaml")

EXPECTED_AUTHORS = ["Randall Balestriero", "Sami BuGhanem"]
WORKFLOWS = (
    "certification.yml",
    "docs.yml",
    "pre-commit.yml",
    "release.yml",
    "testing.yml",
)


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _workflow(name: str) -> dict:
    return yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(),
        Loader=yaml.BaseLoader,
    )


def test_authors_are_exactly_two_ordered_entries() -> None:
    authors = [entry["name"] for entry in _pyproject()["project"]["authors"]]
    assert authors == EXPECTED_AUTHORS


def test_about_merges_integration_author_spelling() -> None:
    about = (ROOT / "stable_envs" / "__about__.py").read_text()
    assert '__author__ = "Randall Balestriero, Sami BuGhanem"' in about
    assert "Bou Ghanem" not in about


def test_core_dependencies_are_minimal() -> None:
    deps = _pyproject()["project"]["dependencies"]
    names = {re.match(r"[A-Za-z0-9_.-]+", dep).group(0).lower() for dep in deps}
    assert names == {"gymnasium", "loguru", "numpy", "tensordict", "torch"}


def test_optional_extras_are_installable_on_pypi() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "all" not in extras
    assert "dev" in extras and "docs" in extras and "mjwarp" in extras
    assert "metaworld" in extras and "humanoidbench" in extras
    blob = "\n".join(req for reqs in extras.values() for req in reqs)
    assert "nvdiffrast" not in blob
    assert " @ " not in blob
    assert "git+" not in blob


def test_dynamic_version_reads_about() -> None:
    project = _pyproject()
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"]["attr"] == ("stable_envs.__about__.__version__")


def test_mkdocs_yaml_is_canonical() -> None:
    canonical = ROOT / "mkdocs.yaml"
    assert canonical.is_file()
    cfg = yaml.safe_load(canonical.read_text())
    assert cfg["site_name"] == "stable-envs"
    assert cfg["site_url"] == "https://galilai-group.github.io/stable-envs/"
    assert cfg["theme"]["name"] == "material"
    assert "nav" in cfg
    assert not (ROOT / "mkdocs.yml").exists()


def test_workflows_exist_and_default_to_read_permissions() -> None:
    for name in WORKFLOWS:
        data = _workflow(name)
        assert data["permissions"]["contents"] == "read"


def test_release_workflow_is_tag_only_trusted_publishing() -> None:
    data = _workflow("release.yml")
    assert "pull_request" not in data["on"]
    assert "push" in data["on"] and "tags" in data["on"]["push"]
    publish = data["jobs"]["publish"]
    assert publish["permissions"]["id-token"] == "write"
    assert publish["environment"]["name"] == "release"
    steps = "\n".join(str(step) for step in publish["steps"])
    assert "PYPI_API_TOKEN" not in steps
    assert "password" not in steps.lower()
    assert "pypa/gh-action-pypi-publish" in steps


def test_docs_deploy_only_on_main_push() -> None:
    data = _workflow("docs.yml")
    deploy = data["jobs"]["deploy"]
    assert "github.ref == 'refs/heads/main'" in str(deploy["if"])
    assert "github.event_name == 'push'" in str(deploy["if"])


def test_certification_is_manual_and_profiled() -> None:
    data = _workflow("certification.yml")
    assert list(data["on"]) == ["workflow_dispatch"]
    assert "cpu" in data["jobs"] and "gpu" in data["jobs"]
    gpu = data["jobs"]["gpu"]
    assert "self-hosted" in gpu["runs-on"]
    assert int(gpu["strategy"]["max-parallel"]) == 1


def test_pre_commit_does_not_reformat_research_history() -> None:
    cfg = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    ruff = next(hook for repo in cfg["repos"] for hook in repo["hooks"] if hook["id"] == "ruff")
    assert "--no-fix" in ruff["args"]
    assert "stable_envs/" in ruff["files"]


def test_ordinary_ci_does_not_invoke_scratch() -> None:
    github = ROOT / ".github"
    hits = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(github.rglob("*"))
        if path.suffix in {".yml", ".yaml"} and "scratch/" in path.read_text()
    ]
    assert hits == []
    testing = (github / "workflows" / "testing.yml").read_text()
    release = (github / "workflows" / "release.yml").read_text()
    assert "tests/packaging/test_distribution.py" in testing
    assert "scripts/ci/audit_dist.py" in testing
    assert "scripts/ci/smoke_installed.py" in testing
    assert "scripts/ci/audit_dist.py" in release
    assert "scripts/ci/smoke_installed.py" in release
