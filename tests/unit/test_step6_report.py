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


def _counts(report: str) -> dict[str, tuple[int, int]]:
    out = {}
    for label, a, b in re.findall(r"\| ([^|]+?)\s*\|\s*(\d+) / (\d+)\s*\|", report):
        out[label.strip()] = (int(a), int(b))
    return out


def test_correctness_is_counted_over_all_49_and_performance_over_the_registered_defaults(report):
    """The scope split Sami decided on 2026-09-15, asserted as arithmetic.

    Step 2 §4.2 puts certification on the registered SWM default and §12.2 puts `ENVIRONMENT_DONE`
    there too, so a render-parameter-space cell owes correctness and owes no tier. Counting the 13 of
    them as performance blockers — which this page did until the decision — invents an obligation the
    frozen contract never wrote down, and inflates the remaining work by a third.
    """
    configs = json.loads(STEP4.read_text())["configs"]
    n = len(configs)
    m = sum(1 for c in configs.values() if c.get("role") == "registered_default")
    assert 0 < m < n, "the corpus no longer has both registered-default and other roles"
    c = _counts(report)

    assert c["Gate-A pass"][1] == n, "Gate A must be counted over every render-distinct cell"
    for label in ("registered/default CPU_FAST+", "registered/default GPU_FAST+",
                  "registered/default rows with CPU missing",
                  "registered/default rows below the GPU target",
                  "remaining registered/default blockers"):
        assert label in c, f"the summary does not report {label!r}"
        assert c[label][1] == m, (
            f"{label} is counted out of {c[label][1]}, not the {m} registered/default rows; "
            f"performance completion is owed only on the registered SWM default")
        assert c[label][0] <= m


def test_no_parameter_space_cell_is_counted_as_a_performance_blocker(report):
    """The 13 must appear — with their Gate A and their replay identity — and never as a blocker."""
    configs = json.loads(STEP4.read_text())["configs"]
    others = [k for k, v in configs.items() if v.get("role") != "registered_default"]
    assert others, "no non-registered-default cells in the corpus"
    section = report.split("Each remaining blocker", 1)[-1].split("## ", 1)[0]
    named = [c for c in others if f"`{c}`" in section]
    assert not named, (
        f"parameter-space cells are listed as remaining performance blockers: {named}; the frozen "
        f"contract asks them for no tier")
    assert "not required by the frozen contract" in report, (
        "the parameter-space table must say that performance is not owed there")
    for c in others:
        assert f"`{c}`" in report


def test_the_performance_bar_is_read_from_the_frozen_spec_not_restated():
    """`CPU_FAST` / `GPU_FAST` must come out of `scratch/certification.json`, not out of this page."""
    spec = importlib.util.spec_from_file_location("step6_report_bar", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bar = mod.spec_bar()["min"]
    frozen = json.loads((ROOT / "scratch" / "certification.json").read_text())
    assert bar == frozen["ladder"]["environment_done_min_tiers"]
    src = GEN.read_text()
    assert 'CPU_OK = (' not in src and 'GPU_OK = (' not in src, (
        "the generator carries its own copy of the tier bar; it must read the frozen one")


def test_the_deferred_work_and_demonstrated_blocker_sections_exist(report):
    """Remaining work must never read as exhausted.

    A blocker table with no deferred-work section beside it turns "not funded" into "impossible".
    The section is present whether or not the dashboard has populated the field — an empty field is
    reported as an empty field, with the candidates the notes still name.
    """
    assert "## Deferred possible work" in report
    assert "## Demonstrated blockers" in report
    tail = report.split("## Deferred possible work", 1)[1]
    assert ("dashboard records no `deferred_work` entries" in tail
            or re.search(r"\|\s*deferred\s*\|", tail)), (
        "the deferred-work section neither lists entries nor says the field is empty")


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


def test_a_superseded_shard_is_excluded_and_listed_with_its_reason(tmp_path):
    """A shard the workstream withdrew must not label a tier, and must not disappear either.

    Workstreams quarantine a measurement they no longer stand behind by moving it into a
    `superseded_*` directory beside the replacement — gfxdraw's node2338 CPU sweep and raster2d's
    round-1 node2344 sweep are both withdrawn for the same reason, a host load average that makes
    the shard a measurement of the node. The page has to honour that (or it reports a tier nobody
    claims) and has to print it (or a withdrawn measurement vanishes from the record).
    """
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("step6_report_sup", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    live = json.loads((RESULTS / "tilemap" / "sweep" / "cpu_after.json").read_text())
    ws = RESULTS / "_test_supersession"
    quarantine = ws / "sweep" / "superseded_loaded_node"
    quarantine.mkdir(parents=True)
    (quarantine / "cpu_after.json").write_text(json.dumps(live))
    (quarantine / "README.md").write_text("# withdrawn\n\nload average 44 on a shared node.\n")
    stamped = ws / "sweep" / "gpu_after.json"
    stamped.write_text(json.dumps(dict(live, unusable="the before leg ran on a different card")))
    try:
        class Args:
            baseline_sha = None
            integration_sha = None

        page = mod.build(RESULTS, ROOT, Args())
        A = mod.Artifacts(RESULTS)
    finally:
        for f in (quarantine / "cpu_after.json", quarantine / "README.md", stamped):
            f.unlink()
        for d in (quarantine, ws / "sweep", ws):
            d.rmdir()

    paths = {p for p, _ in A.superseded}
    assert "examples/results/step6/_test_supersession/sweep/superseded_loaded_node/cpu_after.json" in paths
    assert "examples/results/step6/_test_supersession/sweep/gpu_after.json" in paths
    assert not any(o["ws"] == "_test_supersession"
                   for v in A.tier_observations().values() for o in v), (
        "a withdrawn shard was still labelled with a tier")
    assert "load average 44 on a shared node." in page, "the README's reason did not reach the page"
    assert "the before leg ran on a different card" in page, "the `unusable` reason did not reach the page"


def test_wave0_profile_jobs_count_as_recorded_for_every_workstream(report):
    """The shared wave-0 profiling jobs are recorded once at the top of the dashboard, not eight
    times, and must not be reported as eight unrecorded jobs."""
    import yaml

    dash, _ = yaml.safe_load((RESULTS / "dashboard.yaml").read_text()), None
    wave0 = [str(j) for j in (dash.get("wave0_profile_jobs") or [])]
    if not wave0:
        pytest.skip("the dashboard records no wave0_profile_jobs")
    section = report.split("## Where the dashboard and the artifacts disagree", 1)[-1]
    section = section.split("## Appendix", 1)[0]
    named = [j for j in wave0 if j in section]
    assert not named, f"wave-0 job(s) {named} are still reported as unrecorded disagreements"


def test_the_page_is_reproducible(tmp_path):
    """Two runs over the same artifacts must produce the same bytes.

    A generated page that churns is a page whose diff stops meaning anything, and the way it churns
    here is silent: an ordering that falls out of set iteration changes between processes without a
    single measurement changing.
    """
    pytest.importorskip("yaml")
    pages = []
    for i in range(2):
        spec = importlib.util.spec_from_file_location(f"step6_report_rep{i}", GEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class Args:
            baseline_sha = None
            integration_sha = "fixed"

        pages.append(mod.build(RESULTS, ROOT, Args()))
    assert pages[0] == pages[1], "the generator is not deterministic over one unchanged tree"


def test_the_current_gate_a_record_is_the_newest_by_ancestry(report):
    """Two runs of one certification are two records; only the newest is the verdict.

    texture-fidelity certified the DMC visual cells twice under each reference arm — once before the
    downsample-protocol fix and once after it, the second stamped with a commit that has the first as
    an ancestor. Picking by path order gave the OLD record and printed `DMCWalkerWalk FAIL` on a tree
    where the re-run says PASS. The rule is ancestry on the reported tree, per declared reference
    version, and the records it displaces have to be listed rather than dropped.
    """
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("step6_report_order", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    A = mod.Artifacts(RESULTS)
    order = mod.CommitOrder(ROOT)

    obs = A.gate_a_observations()
    stamped = [o for v in obs.values() for o in v if o.get("commit")]
    if not stamped:
        pytest.skip("no Gate-A certification carries a stamped git_commit in this checkout")
    assert all(o.get("ref") for o in stamped), (
        "a certification with no declared reference version cannot be grouped into an arm")

    checked = 0
    for cfg, rows in obs.items():
        groups = {}
        for o in rows:
            groups.setdefault(o["ref"], []).append(o)
        for ref, group in groups.items():
            if len(group) < 2:
                continue
            cur, old = order.newest(group)
            checked += 1
            for o in old:
                if order.on_tree(o["commit"]) and order.on_tree(cur["commit"]) and o["commit"] != cur["commit"]:
                    assert not order._is_ancestor(cur["commit"], o["commit"]), (
                        f"{cfg}/{ref}: the record chosen as current ({cur['commit'][:8]}) is an "
                        f"ancestor of one it superseded ({o['commit'][:8]})")
                assert f"`{o['path']}`" in report, (
                    f"{cfg}/{ref}: the superseded record {o['path']} is not listed on the page")
    assert checked, "no configuration has two records under one reference; the rule went untested"
    assert "## Correctness after Step 6" in report
    assert "Superseded certification records" in report
