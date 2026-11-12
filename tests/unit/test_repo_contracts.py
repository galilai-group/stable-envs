"""Repository checks: required fixtures exist, no scratch paths are cited, and every docs page is in the nav."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.helpers.repository import maintained_paths

ROOT = Path(__file__).resolve().parents[2]
MAINTAINED_PATHS = maintained_paths(ROOT)

REQUIRED_FIXTURES = (
    "tests/fixtures/swm/frozen_cells.json",
    "tests/fixtures/regression/census/assignment.json",
    "tests/fixtures/regression/references/registry_baseline.json",
)


def test_the_regression_fixtures_are_present():
    missing = [rel for rel in REQUIRED_FIXTURES if not (ROOT / rel).is_file()]
    assert not missing, f"required regression fixtures are missing: {missing}"


def test_scratch_is_not_part_of_the_maintained_tree():
    assert not any(name.startswith("scratch/") for name in MAINTAINED_PATHS), "scratch/ is not a product artifact"


_CITE = re.compile(r"scratch/[A-Za-z0-9_./{}*,\\-]+")
_MAINTAINED = (
    "stable_envs",
    "tests/unit",
    "tests/integration",
    "tests/gpu",
    "tests/regression",
    "tests/packaging",
    "tests/conftest.py",
    "scripts",
    "docs/testing.md",
    "docs/quickstart.md",
    "docs/install.md",
    "docs/index.md",
    "docs/environments.md",
    "docs/benchmarks.md",
    "README.md",
    "CONTRIBUTING.md",
    "mkdocs.yaml",
    "pyproject.toml",
)
_ALLOW_SCRATCH_MENTION = {
    "tests/unit/test_repo_contracts.py",
    "tests/packaging/test_distribution.py",
}


def test_maintained_product_files_do_not_cite_scratch_paths():
    """Maintained files do not cite ``scratch/`` paths."""
    hits = []
    for rel in _MAINTAINED:
        path = ROOT / rel
        files = [path] if path.is_file() else list(path.rglob("*")) if path.is_dir() else []
        for f in files:
            if not f.is_file() or f.suffix not in {".py", ".md", ".yml", ".yaml", ".toml", ".sh"}:
                continue
            key = f.relative_to(ROOT).as_posix()
            if key not in MAINTAINED_PATHS:
                continue
            if key in _ALLOW_SCRATCH_MENTION or key.startswith("scripts/regression/"):
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                for m in _CITE.finditer(line):
                    hits.append(f"{key}:{i}: {m.group(0)}")
    assert not hits, "maintained files still cite scratch/ paths:\n  " + "\n  ".join(hits[:40])


# ------------------------------------------------------------------ documentation


def _nav_targets(node, out):
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for x in node:
            _nav_targets(x, out)
    elif isinstance(node, dict):
        for v in node.values():
            _nav_targets(v, out)


def test_every_page_in_the_mkdocs_nav_exists():
    """A nav entry with no page behind it is a 404 in the built site."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(re.sub(r"!![A-Za-z_.]+\s*", "", (ROOT / "mkdocs.yaml").read_text()))
    targets: list[str] = []
    _nav_targets(cfg.get("nav", []), targets)
    assert len(targets) > 20, f"the nav parsed to only {len(targets)} entries; the parse is wrong"
    missing = [t for t in targets if not t.startswith(("http://", "https://")) and not (ROOT / "docs" / t).exists()]
    assert not missing, f"mkdocs.yaml lists pages that are not there: {missing}"


# Internal API pages linked from the contributor docs rather than listed in the nav.
_NAV_OPTIONAL = {
    "api/style.md",
    "api/scene.md",
    "api/raster2d.md",
    "api/gfxdraw.md",
    "api/raycast3d.md",
    "api/mjwarp.md",
    "api/spaces.md",
}


def test_every_docs_page_is_reachable_from_the_nav():
    """User-guide and contributor pages must be in the nav."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(re.sub(r"!![A-Za-z_.]+\s*", "", (ROOT / "mkdocs.yaml").read_text()))
    listed: list[str] = []
    _nav_targets(cfg.get("nav", []), listed)
    targets = set(listed)
    pages = {
        name.removeprefix("docs/") for name in MAINTAINED_PATHS if name.startswith("docs/") and name.endswith(".md")
    }
    allowed_orphans = pages & _NAV_OPTIONAL
    orphans = sorted(pages - targets - allowed_orphans)
    assert not orphans, (
        f"these pages exist under docs/ and are in no nav section, so the site builds them and nothing "
        f"links to them: {orphans}"
    )
