"""The Step 6 report generator, run against this repository's own artifacts.

The page `docs/research/step6_optimization.md` is the final deliverable of Step 6 and section 25
forbids transcribing its counts by hand, so the only thing standing between the page and a stale
number is that `scratch/step6/report.py` still runs over whatever is in `examples/results/step6/`
today. That tree gains a directory every time a workstream lands, and the previous generator did
crash on the first artifact shape it had not been written for (`KeyError: 'env'` on a non-profile
JSON). So this test runs the real generator over the real artifacts into a temp file.

Four things are asserted, and each of them is a way the page could go quietly wrong:

  * it **exits cleanly** on the current tree — the regression the last generator had;
  * **every one of the 49 frozen configurations appears** in the summary table, because a
    configuration silently dropped from the table is a configuration silently dropped from the four
    counts underneath it;
  * **no cluster path reaches the page** — the artifacts are full of `/oscar/scratch/sboughan/...`,
    which `tests/unit/test_repo_contracts.py` would extract as a `scratch/...` citation that does
    not resolve, so the citation contract fails one commit after somebody prints a `shard` field;
  * **no count in the generator is a literal** — the check `test_step3.py` and `test_step4.py` make
    of their evaluators, applied here: the page's numbers must come from the artifacts.

Nothing here asserts a VALUE. What the counts are depends on which workstreams have merged, and a
test that pinned them would have to be edited on every merge, which is the same failure as
maintaining the page by hand.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scratch" / "step6" / "report.py"
RESULTS = ROOT / "examples" / "results" / "step6"
STEP4 = ROOT / "examples" / "results" / "step4" / "certification.json"


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> str:
    """The generated page, produced the way the coordinator produces it."""
    for p in (GEN, RESULTS, STEP4):
        if not p.exists():
            pytest.skip(f"{p.relative_to(ROOT)} is not present in this checkout")
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("step6_report", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("step6") / "step6_optimization.md"

    class Args:
        baseline_sha = None
        integration_sha = None

    out.write_text(mod.build(RESULTS, ROOT, Args()))
    assert out.stat().st_size > 0, "the generator produced an empty page"
    return out.read_text()


def test_every_frozen_configuration_is_in_the_summary(report):
    """All 49, by name. The counts under the table are sums over its rows."""
    configs = json.loads(STEP4.read_text())["configs"]
    assert len(configs) >= 49, f"the frozen Step 4 corpus has shrunk to {len(configs)} configurations"
    missing = [c for c in configs if f"`{c}`" not in report]
    assert not missing, (
        f"{len(missing)} of the {len(configs)} frozen configurations never reach the page, so the "
        f"counts below the table are sums over an incomplete corpus: {missing}")


def test_the_section_25_counts_are_present_and_are_out_of_the_corpus(report):
    """The four counts section 25 names, each denominated in the frozen corpus."""
    n = len(json.loads(STEP4.read_text())["configs"])
    for label in ("covered configs Gate-A pass", "covered configs CPU_FAST+",
                  "covered configs GPU_FAST+", "remaining blockers"):
        m = re.search(re.escape(label) + r"\s*\|\s*(\d+) / (\d+)", report)
        assert m, f"the page does not report a count for {label!r}"
        assert int(m.group(2)) == n, f"{label} is counted out of {m.group(2)}, not the corpus's {n}"
        assert int(m.group(1)) <= n, f"{label} exceeds the size of the corpus"


def test_no_cluster_path_reaches_the_page(report):
    """`/oscar/scratch/...` printed into `docs/` is a citation that cannot resolve in a clone."""
    bad = sorted({m for m in re.findall(r"\S*(?:/oscar/|\$SCR)\S*", report)})
    assert not bad, (
        "cluster paths reached the generated page; `tests/unit/test_repo_contracts.py` extracts the "
        "`scratch/...` inside them as a citation and it will not resolve:\n  " + "\n  ".join(bad))


def test_every_scratch_path_the_page_cites_resolves(report):
    """The same contract `test_repo_contracts.py` applies repo-wide, checked before the page lands."""
    cited = {c.rstrip(".,;:)]}>\"'`") for c in re.findall(r"scratch/[A-Za-z0-9_./-]+", report)}
    broken = sorted(c for c in cited if c != "scratch/" and not (ROOT / c).exists())
    assert not broken, f"the page cites scratch paths that do not exist here: {broken}"


def test_the_appendix_accounts_for_every_json_artifact(report):
    """Unknown shapes are listed, not dropped: the appendix is the generator's own to-do list."""
    assert "artifacts not summarised" in report
    unknown = [p for p in RESULTS.rglob("*.json")
               if f"`{p.relative_to(ROOT)}`" not in report]
    assert not unknown, (
        f"{len(unknown)} JSON artifact(s) are neither summarised nor listed in the appendix, which "
        f"means the page is silently ignoring them: {[str(p.relative_to(ROOT)) for p in unknown[:8]]}")


def test_no_count_or_measurement_is_hardcoded_in_the_generator():
    """No numeric literal in the generator may stand in for a measurement.

    The same rule Step 3 and Step 4 apply to their evaluators. Literals that are structural — string
    slices, table widths, a decimal precision — are allowed by being small; anything that looks like
    a count of configurations or a throughput is not.
    """
    tree = ast.parse(GEN.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if not isinstance(node.value, bool) and abs(node.value) > 30:
                bad.append(f"line {node.lineno}: {node.value}")
    assert not bad, (
        "numeric literals that are too large to be structural appear in the generator; a count or a "
        "measurement must be read from an artifact:\n  " + "\n  ".join(bad))


def test_the_generator_survives_a_tree_with_no_git_and_an_unreadable_artifact(tmp_path):
    """The two failure modes the Oscar copies actually have: no `.git`, and a JSON nobody expected."""
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("step6_report_nogit", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    junk = RESULTS / "_test_unreadable.json"
    junk.write_text("{ this is not json")
    try:
        class Args:
            baseline_sha = "0" * 40
            integration_sha = "deadbeef"

        page = mod.build(RESULTS, tmp_path, Args())          # tmp_path is not a checkout
    finally:
        junk.unlink()
    assert "deadbeef" in page and "_test_unreadable.json" in page, (
        "an unparseable artifact must appear in the appendix rather than stopping the report")
