"""The repo's own promises about itself: fixtures resolve, docs are reachable, launchers agree.

None of this is about the renderer. It is about claims the repository makes that nothing else
checks: the regression fixtures the suite loads, the MkDocs nav, and the cluster launchers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.helpers.repository import maintained_paths

ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
MAINTAINED_PATHS = maintained_paths(ROOT)

REQUIRED_FIXTURES = (
    "tests/fixtures/swm/frozen_cells.json",
    "tests/fixtures/semantic_recovery_matrix.json",
    "tests/fixtures/regression/census/assignment.json",
    "tests/fixtures/regression/references/registry_baseline.json",
)


def test_the_regression_fixtures_are_present():
    missing = [rel for rel in REQUIRED_FIXTURES if not (ROOT / rel).is_file()]
    assert not missing, f"required regression fixtures are missing: {missing}"


def test_scratch_is_not_part_of_the_maintained_tree():
    assert not any(name.startswith("scratch/") for name in MAINTAINED_PATHS), "scratch/ is not a product artifact"


_CITE = re.compile(r"(?<!oscar/)scratch/[A-Za-z0-9_./{}*,\\-]+")
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
    "docs/gallery.md",
    "docs/environments.md",
    "docs/benchmarks.md",
    "docs/adding-an-env.md",
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
    """Historical probes live on milestone tags. The product tree must not point at them."""
    hits = []
    for rel in _MAINTAINED:
        path = ROOT / rel
        files = [path] if path.is_file() else list(path.rglob("*")) if path.is_dir() else []
        for f in files:
            if not f.is_file() or f.suffix not in {".py", ".md", ".yml", ".yaml", ".toml", ".sh", ".sbatch"}:
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
                if "/oscar/scratch" in line:
                    continue
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
    """A nav entry with no page behind it is a 404 in the built site and a warning nobody reads."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(re.sub(r"!![A-Za-z_.]+\s*", "", (ROOT / "mkdocs.yaml").read_text()))
    targets: list[str] = []
    _nav_targets(cfg.get("nav", []), targets)
    assert len(targets) > 20, f"the nav parsed to only {len(targets)} entries; the parse is wrong"
    missing = [t for t in targets if not t.startswith(("http://", "https://")) and not (ROOT / "docs" / t).exists()]
    assert not missing, f"mkdocs.yaml lists pages that are not there: {missing}"


# Historical research notes and internal API pages are linked from the archive
# landing page / contributor docs rather than listed in the main nav.
_NAV_OPTIONAL = {
    "api/style.md",
    "api/scene.md",
    "api/raster2d.md",
    "api/gfxdraw.md",
    "api/tilemap.md",
    "api/raycast3d.md",
    "api/mjwarp.md",
    "api/rendernet.md",
    "api/spaces.md",
}


def test_every_docs_page_is_reachable_from_the_nav():
    """User-guide and contributor pages must be in the nav."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(re.sub(r"!![A-Za-z_.]+\s*", "", (ROOT / "mkdocs.yaml").read_text()))
    listed: list[str] = []
    _nav_targets(cfg.get("nav", []), listed)
    targets = set(listed)
    pages = {name.removeprefix("docs/") for name in MAINTAINED_PATHS
             if name.startswith("docs/") and name.endswith(".md")}
    allowed_orphans = pages & _NAV_OPTIONAL
    orphans = sorted(pages - targets - allowed_orphans)
    assert not orphans, (
        f"these pages exist under docs/ and are in no nav section, so the site builds them and nothing "
        f"links to them: {orphans}"
    )



# ------------------------------------------------------------------ the launchers


def _sbatch_directive(path: Path, key: str) -> str | None:
    for line in path.read_text().splitlines():
        if line.startswith(f"#SBATCH --{key}="):
            return line.split("=", 1)[1].strip()
    return None


LAUNCHERS = sorted((ROOT / "scripts").glob("run_*.sbatch"))


def test_there_are_launchers_to_check():
    assert len(LAUNCHERS) >= 3, f"expected the three sbatch launchers, found {LAUNCHERS}"


def test_all_launchers_exclude_the_same_measured_faulty_nodes():
    """A node that returns different pixels for the same scene invalidates every number measured on it.

    The lists used to disagree — the DM Control launcher excluded ``gpu-068`` and ``gpu-082`` and the
    other two did not — which means the full suite and the smoke subset were both still schedulable
    onto hardware the DM Control run had already been told to avoid. The failure that produces is not a
    crash: it is 25 OGBench cells disagreeing by 69-90/255 and reading exactly like a state-sufficiency
    violation in the ray caster, followed by a SIGABRT that no ``try``/``except`` catches.
    """
    lists = {p.name: _sbatch_directive(p, "exclude") for p in LAUNCHERS}
    missing = [n for n, v in lists.items() if not v]
    assert not missing, f"these launchers exclude no nodes at all: {missing}"
    sets = {n: frozenset(v.split(",")) for n, v in lists.items()}
    distinct = set(sets.values())
    assert len(distinct) == 1, (
        "the launchers do not agree on which nodes are unfit to measure on, so a run submitted through "
        "one of them can land on hardware another has condemned:\n  "
        + "\n  ".join(f"{n}: {sorted(s)}" for n, s in sorted(sets.items()))
    )


def test_every_launcher_names_both_the_partition_and_the_account():
    """Setting one without the other is silently rejected by this scheduler."""
    for p in LAUNCHERS:
        for key in ("partition", "account", "qos", "gres"):
            assert _sbatch_directive(p, key), f"{p.name} does not set --{key}"


def test_every_launcher_runs_through_the_supervisor_rather_than_bare_pytest():
    """A bare pytest reports NOTHING when the interpreter dies, which it has done four times in a week.

    SIGABRT/SIGSEGV out of MuJoCo, EGL or the CUDA driver: the test that died is not named, every test
    after it never runs, and the whole artifact of a twenty-minute job is a shell exit code. The
    supervisor turns that into one CRASHED test and finishes the run, and it costs one extra process
    when nothing crashes.
    """
    for p in LAUNCHERS:
        src = p.read_text()
        assert "tests._runner.supervisor" in src, (
            f"{p.name} does not run through the supervisor, so a crash in it produces no report"
        )
        assert "--json" in src and "--rundir" in src, (
            f"{p.name} does not ask the supervisor for its artifacts, so a preempted job leaves nothing behind to read"
        )
