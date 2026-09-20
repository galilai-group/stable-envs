"""The Craftax semantic record has to agree with itself.

The semantic follow-up returned a ``FINDING`` rather than a Gate-H run, and the whole argument rests on two
committed measurements: that both Craftax cells are byte-identical to the official renderer, and
that no downstream Craftax evaluator exists. A prose verdict that has drifted from the JSON
underneath it is worse than no verdict, because it reads as evidence. These tests re-derive every
claim ``examples/results/step8/semantic/findings.md`` makes from the artifacts beside it, so the
document cannot be edited away from its own numbers without the suite failing.

They assert nothing about the renderer — that is Gate A's job — and they run no render, so they
stay fast and need no reference interpreter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "examples" / "results" / "step8" / "semantic"
DOC = ROOT / "docs" / "research" / "step8_semantic_craftax.md"

SIZES = {("CraftaxClassic", "7"), ("CraftaxClassic", "16"), ("CraftaxFull", "10"), ("CraftaxFull", "16")}


def load(name: str) -> dict:
    p = REC / name
    if not p.exists():
        pytest.fail(f"missing semantic artifact: {p.relative_to(ROOT)}")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def identity() -> dict:
    return load("identity_recheck.json")


@pytest.fixture(scope="module")
def channels() -> dict:
    return load("channel_probe.json")


@pytest.fixture(scope="module")
def inventory() -> dict:
    return load("evaluator_inventory.json")


@pytest.fixture(scope="module")
def findings() -> str:
    p = REC / "findings.md"
    if not p.exists():
        pytest.fail(f"missing {p.relative_to(ROOT)}")
    return p.read_text()


def test_the_identity_recheck_covers_both_cells_at_every_certified_size(identity):
    got = {(c, s) for c, row in identity["cells"].items() for s in row["sizes"]}
    assert got == SIZES, got


def test_every_frame_is_byte_identical_to_the_official_renderer(identity):
    """The single fact the FINDING rests on. If this fails the verdict is void, not stale."""
    for cell, row in identity["cells"].items():
        assert row["corpus_unchanged"], f"{cell}: the frozen Step-7 corpus moved"
        for size, e in row["sizes"].items():
            assert e["digest_ours"] == e["digest_official"], f"{cell}@{size}"
            assert e["byte_identical"] and e["exact_frac"] == 1.0 and e["max_abs"] == 0
            assert e["matches_frozen"], f"{cell}@{size}: differs from the frozen Step-7 digest"
    assert identity["all_byte_identical"] is True


def test_the_reference_is_the_declared_one(identity):
    assert identity["craftax"] == "1.6.1"
    assert identity["jax"] == "0.11.1"
    assert identity["jax_threefry_partitionable"] is True


def test_every_digest_quoted_in_the_findings_is_one_the_recheck_measured(findings, identity):
    quoted = set(re.findall(r"\b[0-9a-f]{64}\b", findings))
    assert quoted, "the findings quote no digest at all"
    measured = {
        e[k]
        for row in identity["cells"].values()
        for e in row["sizes"].values()
        for k in ("digest_ours", "digest_official")
    }
    assert quoted <= measured, quoted - measured


def test_the_added_channels_are_information_bearing_and_separable(channels):
    """§3 of the findings: the strip is a fifth to a third of the frame, night moves the world,
    and neither leaks into the other. A regression here would change the argument, not just a
    number."""
    for cell, row in channels["cells"].items():
        for size, e in row["sizes"].items():
            assert e["strip_frac_of_frame"] > 0.2, f"{cell}@{size}"
            assert e["day_vs_night"]["world_rows"]["px_changed_frac"] > 0.9
            assert e["awake_vs_sleeping"]["world_rows"]["px_changed_frac"] > 0.9
            # the strip is concatenated after the night chain, so lighting never reaches it
            assert e["day_vs_night"]["strip_rows"]["max_abs"] == 0
            assert e["awake_vs_sleeping"]["strip_rows"]["max_abs"] == 0
            # and the inventory never reaches the world rows
            assert e["inventory_actual_vs_empty"]["world_rows"]["max_abs"] == 0
            assert e["inventory_actual_vs_empty"]["strip_rows"]["px_changed_frac"] > 0.2


def test_the_strip_fractions_quoted_in_the_findings_match_the_probe(findings, channels):
    for cell, row in channels["cells"].items():
        for e in row["sizes"].values():
            pct = f"{100 * e['strip_frac_of_frame']:.1f} %"
            assert pct in findings, f"{cell}: {pct} is not in findings.md"


def test_no_downstream_craftax_evaluator_is_claimed_to_exist(inventory):
    assert inventory["any_need_satisfied"] is False
    for need, row in inventory["needs"].items():
        assert row["satisfied"] is False, need
    assert not inventory["needs"]["checkpoint"]["craftax_entries"]
    assert not inventory["needs"]["episode_corpus"]["craftax_collector"]
    assert not inventory["needs"]["planner_config"]["craftax_eval_config"]
    assert not inventory["step5_evaluator"]["craftax_arm"]
    assert not inventory["swm_compat_workflow"]["craftax_arm"]


def test_the_verdict_line_says_what_the_artifacts_say(findings, inventory, identity):
    head = findings.splitlines()[2]
    assert "`FINDING`" in head, head
    assert "No Gate-H run was made" in head
    assert "**Bucket: `DOCUMENTED_LIMITATION`**" in findings
    # a FINDING is only honest while both halves of its premise hold
    assert identity["all_byte_identical"] and not inventory["any_need_satisfied"]


def test_the_merge_gate_table_has_all_ten_items(findings):
    rows = re.findall(r"^\| (\d+) \| .+ \| .+ \|$", findings, re.M)
    assert [int(r) for r in rows] == list(range(1, 11)), rows


def test_the_docs_page_exists_and_is_in_the_nav():
    assert DOC.exists(), f"missing {DOC}"
    landing = (ROOT / "docs" / "research" / "index.md").read_text()
    assert "research/index.md" in (ROOT / "mkdocs.yaml").read_text()
    assert "step8_semantic_craftax.md" in landing
