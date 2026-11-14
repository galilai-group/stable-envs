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
    "ci.yml",
    "regression.yml",
    "docs.yml",
    "release.yml",
    "extras-smoke.yml",
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


def test_requires_python_matches_the_ci_matrix() -> None:
    """Metadata and CI must tell the same story. 3.13+ is not claimed until a job runs there."""
    assert _pyproject()["project"]["requires-python"] == ">=3.10,<3.13"
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert '"3.10"' in ci and '"3.11"' in ci and '"3.12"' in ci
    assert '"3.13"' not in ci


def test_core_dependencies_are_minimal() -> None:
    deps = _pyproject()["project"]["dependencies"]
    names = {re.match(r"[A-Za-z0-9_.-]+", dep).group(0).lower() for dep in deps}
    assert names == {"gymnasium", "jax", "loguru", "numpy", "tensordict", "torch"}


def test_optional_extras_are_installable_on_pypi() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    assert "all" not in extras
    assert "dev" in extras and "docs" in extras and "mjwarp" in extras
    assert set(extras["pusht"]) <= set(extras["pusht-gpu"])
    assert "jax[cuda12]>=0.4.35; sys_platform == 'linux'" in extras["pusht-gpu"]
    blob = "\n".join(req for reqs in extras.values() for req in reqs)
    assert "nvdiffrast" not in blob
    assert " @ " not in blob
    assert "git+" not in blob


def test_console_script_points_at_the_cli() -> None:
    assert _pyproject()["project"]["scripts"] == {"stable-envs": "stable_envs.cli:main"}
    assert "def main(" in (ROOT / "stable_envs" / "cli.py").read_text()


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
    assert cfg["theme"]["name"] == "shadcn"
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


def test_regression_is_manual_and_profiled() -> None:
    data = _workflow("regression.yml")
    assert list(data["on"]) == ["workflow_dispatch"]
    assert "cpu" in data["jobs"] and "gpu" in data["jobs"]
    gpu = data["jobs"]["gpu"]
    assert "self-hosted" in gpu["runs-on"]
    assert int(gpu["strategy"]["max-parallel"]) == 1


def test_pre_commit_hygiene_excludes_only_frozen_artifacts() -> None:
    cfg = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    ruff = next(hook for repo in cfg["repos"] for hook in repo["hooks"] if hook["id"] == "ruff")
    assert "--fix" in ruff["args"] and "--exit-non-zero-on-fix" in ruff["args"]
    assert "stable_envs/" in ruff["files"]
    whitespace = next(hook for repo in cfg["repos"] for hook in repo["hooks"] if hook["id"] == "trailing-whitespace")
    exclude = whitespace.get("exclude") or ""
    assert "tests/fixtures/" in exclude
    assert "docs/assets/" in exclude
    assert r".*\.md$" not in exclude
    assert exclude.strip() != r"^(docs/|"


def test_lightweight_regression_runs_on_ordinary_ci() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "macos-latest" not in ci
    noxfile = (ROOT / "noxfile.py").read_text()
    assert '"tests/regression"' in noxfile and "not gpu" in noxfile
    extras = (ROOT / ".github" / "workflows" / "extras-smoke.yml").read_text()
    assert "scripts/ci/smoke_extra.py" in extras
    assert "workflow_dispatch" in extras


def test_ordinary_ci_does_not_invoke_scratch() -> None:
    github = ROOT / ".github"
    hits = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(github.rglob("*"))
        if path.suffix in {".yml", ".yaml"} and "scratch/" in path.read_text()
    ]
    assert hits == []
    ci = (github / "workflows" / "ci.yml").read_text()
    release = (github / "workflows" / "release.yml").read_text()
    assert "tests/packaging/test_distribution.py" in ci
    assert "scripts/ci/audit_dist.py" in ci
    assert "scripts/ci/smoke_installed.py" in ci
    assert "scripts/ci/audit_dist.py" in release
    assert "scripts/ci/smoke_installed.py" in release


def test_ci_ok_is_the_one_aggregate_of_every_gating_job() -> None:
    """Branch protection requires only ci-ok, so a new matrix entry never needs a settings change."""
    jobs = _workflow("ci.yml")["jobs"]
    ok = jobs["ci-ok"]
    assert "always()" in ok["if"] and "pull_request.draft" in ok["if"]
    assert set(ok["needs"]) == {"plan", "static", "core", "family", "docs", "package", "install"}
    assert "re-actors/alls-green" in str(ok["steps"])
    allowed = {name.strip() for name in ok["steps"][0]["with"]["allowed-skips"].split(",")}
    assert {"static", "docs", "plan"}.isdisjoint(allowed)


def test_pull_requests_test_affected_families_and_main_tests_all() -> None:
    data = _workflow("ci.yml")
    assert set(data["on"]) >= {"pull_request", "push", "schedule", "workflow_dispatch"}
    assert data["on"]["push"]["branches"] == ["main"]
    plan = "\n".join(str(step) for step in data["jobs"]["plan"]["steps"])
    assert "scripts/ci/plan.py" in plan
    assert '"$EVENT" == push || "$EVENT" == schedule' in plan and "scope=all" in plan
    family = data["jobs"]["family"]
    assert family["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.families) }}"
    assert "nox -s" in str(family["steps"]) and "CI" in str(family["steps"])


def test_ordinary_ci_installs_from_the_lockfile() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'UV_FROZEN: "1"' in ci and "uv lock --check" in ci
    assert not (ROOT / ".github" / "workflows" / "testing.yml").exists()
    assert not (ROOT / ".github" / "workflows" / "pre-commit.yml").exists()


def test_a_red_full_run_opens_one_tracking_issue() -> None:
    data = _workflow("ci.yml")
    assert data["on"]["schedule"][0]["cron"]
    report = data["jobs"]["report"]
    assert report["needs"] == "ci-ok"
    assert "failure()" in report["if"] and "push" in report["if"] and "schedule" in report["if"]
    assert report["permissions"] == {"issues": "write"}
