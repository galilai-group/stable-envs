"""The repo's own promises about itself: cited probes resolve, docs are reachable, launchers agree.

None of this is about the renderer, and all of it is about claims the repository makes that nothing was
checking.

THE CITATIONS ARE THE REPRODUCTION PATH. 217 distinct ``scratch/...`` paths are named from ``docs/``,
``examples/paper/main.tex``, ``README.md`` and from source comments inside ``stable_envs/`` itself
("measured by ``scratch/census/bbox_probe.py``", "raw JSON in ``scratch/ogbench/out/``"). Every measured
number in the documentation rests on one of them. They are the reason ``scratch/`` is tracked at all,
they are what a ``.gitignore`` rule can silently exclude, and they are what a probe rename silently
breaks — an audit today found one citation already dead, pointing at a ``scratch/pgx/probe4_state.py``
that has never existed. A dead citation is not cosmetic: it is a measurement whose provenance is gone.

So the extraction is done here rather than by hand, over the same files a reader would follow, with the
bash brace-expansion and glob forms the docs actually use (``probe{13..20}*.py``,
``probe_{mjwarp,texgen}.py``). And the ``.gitignore`` is asserted against it, because the rules that
keep 3.4 GB of rendered evidence out of the history are one pattern away from also excluding the
scripts that produced it.
"""

from __future__ import annotations

import glob
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Where a citation can be written. Source comments count: several of the sharpest ones live in
#: ``stable_envs/envs/*.py`` docstrings, next to the constant they justify.
SEARCH_ROOTS = ("docs", "examples", "stable_envs", "scripts", "tests")
SEARCH_FILES = ("README.md", "CONTRIBUTING.md", "mkdocs.yaml", "pyproject.toml")
SEARCH_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".sh", ".sbatch", ".tex", ".txt"}

_CITE = re.compile(r"scratch/[A-Za-z0-9_./{}*,\-]+")
_TRAILING = ".,;:)]}>\"'`"

#: Cited paths that are DELIBERATELY not in the repository, each with what rebuilds it. They are
#: directories of rendered evidence rather than code, they are named from the docs as "the raw numbers
#: live here", and the standing rule is that the repo is code only. Listing them makes the exclusion a
#: decision with a reason attached instead of a citation that quietly stopped resolving.
#: Keys are compared with any trailing slash removed, because the prose writes a directory both ways
#: ("raw JSON in ``scratch/ogbench/out/``" and "``scratch/robocasa/out/corpus``") and an exemption that
#: depended on which one an author typed would be an exemption that silently lapses.
REGENERABLE = {
    "scratch/robocasa/out": "3 GB of expanded MJCFs, qpos banks and union models; "
    "scratch/robocasa/dump_corpus.py rebuilds it in ~15 min on a GPU node",
    "scratch/robocasa/out/corpus": "the same corpus named directly; scratch/robocasa/dump_corpus.py "
    "rebuilds it in ~15 min on a GPU node",
    "scratch/ogbench/out": "raw per-cell JSON written by the probes in scratch/ogbench/, which "
    "re-running them regenerates",
    "scratch/carracing/logs": "SLURM stdout for the CarRacing sweep; scratch/carracing/probe_*.py "
    "re-runs the measurements that produced it",
}


def _norm(cited: str) -> str:
    return cited.rstrip("/")


#: This file is excluded from its own scan. It has to NAME the dead citation it was written to catch
#: (``scratch/pgx/probe4_state.py``, which never existed) in order to explain itself, and a checker
#: that reads its own examples reports a failure it created — which is exactly what happened the first
#: time this ran.
SELF = Path(__file__).resolve()


def _is_reviewer_report(f: Path) -> bool:
    """Independent-review reports under ``examples/results/<step>/<ws>/review_*.md`` are third-party
    records: a reviewer quotes the exact paths it asked an author to scrub, so scanning them re-creates
    the very citations the author removed. They are evidence, not navigation, and are exempt."""
    return f.name.startswith("review_") and f.suffix == ".md" and "results" in f.parts


def _is_archaeology_inventory(f: Path) -> bool:
    """``v0_archaeology.md`` names leave-behind trees on purpose, including after ``git rm``."""
    return f.name == "v0_archaeology.md" and "research" in f.parts


def _files():
    for d in SEARCH_ROOTS:
        p = ROOT / d
        if p.is_dir():
            for f in p.rglob("*"):
                if (
                    f.is_file()
                    and f.suffix in SEARCH_SUFFIXES
                    and "__pycache__" not in f.parts
                    and f.resolve() != SELF
                    and not _is_reviewer_report(f)
                    and not _is_archaeology_inventory(f)
                ):
                    yield f
    for name in SEARCH_FILES:
        if (ROOT / name).is_file():
            yield ROOT / name


def _citations() -> dict[str, list[str]]:
    """``{cited path: [file:line, ...]}`` over everything a reader could follow."""
    out: dict[str, list[str]] = {}
    for f in _files():
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for m in _CITE.finditer(line):
                raw = m.group(0).rstrip(_TRAILING)
                if raw and raw != "scratch/":
                    out.setdefault(raw, []).append(f"{f.relative_to(ROOT)}:{i}")
    return out


def _expand(cited: str) -> tuple[list[str], list[str]]:
    """``(concrete forms, paths that exist)`` for a citation, honouring bash braces and globs.

    The docs cite groups the way somebody would type them at a shell — ``probe{1,2,3}*.py``,
    ``probe{13..20}*.py`` — so a checker that treated those literally would report every one of them
    as broken and be switched off within the day. A group is satisfied when at least one of its
    expansions exists, which is the claim the prose is actually making.
    """
    pending, forms = [cited], []
    while pending:
        s = pending.pop()
        m = re.search(r"\{([^{}]*)\}", s)
        if not m:
            forms.append(s)
            continue
        body = m.group(1)
        rng = re.fullmatch(r"(\d+)\.\.(\d+)", body)
        alts = [str(x) for x in range(int(rng.group(1)), int(rng.group(2)) + 1)] if rng else body.split(",")
        pending += [s[: m.start()] + a + s[m.end() :] for a in alts]
    hits = [h for f in forms for h in glob.glob(str(ROOT / f))]
    return forms, hits


CITATIONS = _citations()


@pytest.fixture(scope="module")
def ignored() -> set[str]:
    """Paths under ``scratch/`` that git is currently ignoring, or a skip if this is not a checkout."""
    p = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--ignored", "--exclude-standard", "--", "scratch"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        pytest.skip(f"not a git checkout, or git is unavailable: {p.stderr.strip()}")
    return set(p.stdout.splitlines())


def test_there_are_citations_to_check():
    """The zero-collection rule applied to this file: an extraction that finds nothing proves nothing."""
    assert len(CITATIONS) > 100, (
        f"only {len(CITATIONS)} scratch citations found; the regex or the search roots have drifted "
        f"and every check below would pass vacuously"
    )


def test_every_cited_scratch_path_resolves():
    """A citation that does not resolve is a measurement whose provenance is gone.

    This is the check that would have caught ``scratch/pgx/probe4_state.py``, cited from
    ``stable_envs/envs/pgx.py`` for the verification of go's chain-id clipping and othello's turn flip,
    pointing at a file that has never existed in this tree or in its history.
    """
    if not (ROOT / "scratch").is_dir():
        pytest.skip("scratch/ is not present in this checkout, so there is nothing to resolve against")
    broken = []
    for cited, sites in sorted(CITATIONS.items()):
        forms, hits = _expand(cited)
        if not hits:
            broken.append(
                f"{cited}\n      cited at {'; '.join(sites[:4])}"
                + (f" (+{len(sites) - 4} more)" if len(sites) > 4 else "")
                + (f"\n      expands to {forms}" if forms != [cited] else "")
            )
    assert not broken, (
        f"{len(broken)} cited scratch path(s) do not resolve. Each one is a number in the docs whose "
        f"provenance can no longer be opened:\n  " + "\n  ".join(broken)
    )


def test_no_cited_probe_is_excluded_by_gitignore(ignored):
    """The rule that makes the ``scratch/`` ignore block safe to extend.

    ``.gitignore`` carries ~40 patterns whose job is to keep 3.4 GB of rendered evidence, vendored C++
    and SLURM stdout out of a repository that is meant to be code only — a push has already failed on
    1.7 GB of shards. Every one of those patterns is one character away from also excluding the probes,
    and nothing about a probe silently disappearing from a fresh clone looks like a failure until
    somebody tries to reproduce a number.

    A cited path that IS deliberately excluded belongs in ``REGENERABLE`` above with the command that
    rebuilds it, which is the same discipline ``.gitignore`` itself uses for the RoboCasa corpus.
    """
    bad = []
    for cited, sites in sorted(CITATIONS.items()):
        if _norm(cited) in REGENERABLE:
            continue
        for hit in _expand(cited)[1]:
            rel = str(Path(hit).relative_to(ROOT))
            if rel in ignored:
                bad.append(f"{rel} (cited as {cited} at {sites[0]})")
    assert not bad, (
        "these cited paths exist here but are excluded by .gitignore, so they are not in a fresh "
        "clone and the citation resolves only on the machine that wrote it:\n  "
        + "\n  ".join(bad)
        + "\n  Either narrow the ignore rule, or record the path in REGENERABLE with what rebuilds it."
    )


def test_every_regenerable_entry_is_still_cited_and_still_names_a_rebuild():
    """The allowlist must not outlive its citations, or it becomes a licence nobody can audit."""
    cited = {_norm(c) for c in CITATIONS}
    stale = [c for c in REGENERABLE if c not in cited]
    assert not stale, (
        f"{stale} are recorded as deliberately-excluded evidence but nothing cites them any more; "
        f"drop the entry rather than leaving an unexplained exemption"
    )
    thin = [c for c, why in REGENERABLE.items() if len(why) < 30 or "scratch/" not in why]
    assert not thin, f"{thin}: an exemption has to name what regenerates the evidence"


def test_the_probe_scripts_themselves_are_all_still_reachable():
    """Narrowed to ``.py``: the scripts ARE the reproduction path, and they are never evidence.

    Stated separately from the general resolution check so that a future ignore rule which starts
    matching Python under ``scratch/`` fails with a message about the reproduction path rather than
    about a broken link.
    """
    if not (ROOT / "scratch").is_dir():
        pytest.skip("scratch/ is not present in this checkout")
    cited_py = sorted({h for c in CITATIONS for h in _expand(c)[1] if h.endswith(".py")})
    assert len(cited_py) > 50, (
        f"only {len(cited_py)} cited probe scripts resolved; the docs cite many more than that, so the "
        f"extraction has drifted"
    )
    missing = [p for p in cited_py if not Path(p).is_file()]
    assert not missing, missing


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
    """User-guide and contributor pages must be in the nav. Research archive pages
    other than the landing page are linked from ``docs/research/index.md``.
    """
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(re.sub(r"!![A-Za-z_.]+\s*", "", (ROOT / "mkdocs.yaml").read_text()))
    listed: list[str] = []
    _nav_targets(cfg.get("nav", []), listed)
    targets = set(listed)
    assert "research/index.md" in targets, "the research archive landing page must be in the nav"
    pages = {str(p.relative_to(ROOT / "docs")) for p in (ROOT / "docs").rglob("*.md")}
    allowed_orphans = {
        p for p in pages if (p.startswith("research/") and p != "research/index.md") or p in _NAV_OPTIONAL
    }
    orphans = sorted(pages - targets - allowed_orphans)
    assert not orphans, (
        f"these pages exist under docs/ and are in no nav section, so the site builds them and nothing "
        f"links to them: {orphans}"
    )


def test_research_archive_landing_page_links_evergreen_and_history():
    landing = (ROOT / "docs" / "research" / "index.md").read_text()
    for needle in (
        "when_does_this_pay_off.md",
        "certification.md",
        "step8_semantic_craftax.md",
        "step8_closure.md",
        "step9_closure.md",
    ):
        assert needle in landing, f"research archive landing page does not link to {needle}"


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
        assert "stable_envs.testing.supervisor" in src, (
            f"{p.name} does not run through the supervisor, so a crash in it produces no report"
        )
        assert "--json" in src and "--rundir" in src, (
            f"{p.name} does not ask the supervisor for its artifacts, so a preempted job leaves nothing behind to read"
        )
