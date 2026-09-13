"""``stable_envs.testing.provenance``: a bound measured against one reference version is not judged
against another.

The measured case: DM Control's trackcom cells, declared under mujoco 3.11.0, were judged in Step 4
under 3.12.0, whose finite-plane texture re-anchoring (changelog 3.12.0 item 15, commit cc7fb98c) had
phase-shifted the reference's floor checker. Three renderer FAILs were recorded for a version drift.
The rule here is the existing one -- a dependency failure is not a fidelity failure -- applied one
step later: the reference at the wrong version is a comparison that could not be made.
"""
from __future__ import annotations

import pytest

from stable_envs.testing.provenance import (REFERENCE_JUDGED_CHECKS, block_on_reference_mismatch,
                                             normalize_package, reference_version_mismatch)

DECLARED = {"mujoco": "3.11.0", "mujoco_warp": "3.11.0"}


def test_matching_versions_are_not_a_mismatch():
    assert reference_version_mismatch(DECLARED, {"mujoco": "3.11.0", "mujoco_warp": "3.11.0",
                                                 "torch": "2.14.0"}) == []


def test_a_different_version_is_named_with_both_sides():
    bad = reference_version_mismatch(DECLARED, {"mujoco": "3.12.0", "mujoco_warp": "3.11.0"})
    assert bad == ["mujoco: declared 3.11.0, interpreter had 3.12.0"]


def test_an_unrecorded_or_unknown_version_cannot_confirm_a_match():
    """``fidelity_step4.provenance`` wrote "present" for a package with no ``__version__``; "could
    not confirm" is the blocked case, not the passing one."""
    for recorded in ({}, {"mujoco": None}, {"mujoco": "present"}, {"mujoco": ""}):
        bad = reference_version_mismatch({"mujoco": "3.11.0"}, recorded)
        assert bad == ["mujoco: declared 3.11.0, version not recorded by the measuring interpreter"], recorded


def test_nothing_declared_is_not_a_mismatch():
    """The pre-field state of the world: every existing declaration is judged exactly as before."""
    assert reference_version_mismatch({}, {"mujoco": "3.12.0"}) == []
    assert reference_version_mismatch(None, None) == []


def test_dash_and_underscore_spell_one_package():
    assert normalize_package("mujoco-warp") == normalize_package("mujoco_warp") == "mujoco_warp"
    assert reference_version_mismatch({"mujoco-warp": "3.11.0"}, {"mujoco_warp": "3.11.0"}) == []


def test_only_checks_that_read_the_reference_are_blocked():
    """A self-consistency check (determinism, batch invariance, the 2x downsample) never reads the
    official renderer, so a reference at the wrong version says nothing about it."""
    wrong = {"mujoco": "3.12.0", "mujoco_warp": "3.12.0"}
    for check in REFERENCE_JUDGED_CHECKS:
        rec = block_on_reference_mismatch(check, DECLARED, wrong)
        assert rec is not None and rec["status"] == "blocked_environment", check
        assert "3.11.0" in rec["why"] and "3.12.0" in rec["why"]
        assert rec["declared_reference_versions"] == DECLARED
        assert rec["recorded_reference_packages"] == {"mujoco": "3.12.0", "mujoco_warp": "3.12.0"}
    for check in ("check_determinism", "check_batch_invariance", "check_resolution_consistency"):
        assert block_on_reference_mismatch(check, DECLARED, wrong) is None, check


def test_a_matching_reference_blocks_nothing():
    for check in REFERENCE_JUDGED_CHECKS:
        assert block_on_reference_mismatch(check, DECLARED, dict(DECLARED)) is None
        assert block_on_reference_mismatch(check, {}, {"mujoco": "3.12.0"}) is None


@pytest.mark.parametrize("check", REFERENCE_JUDGED_CHECKS)
def test_the_blocked_record_carries_the_mismatch_for_the_report(check):
    rec = block_on_reference_mismatch(check, {"mujoco": "3.11.0"}, {"mujoco": "3.12.0"})
    assert rec["reference_version_mismatch"] == ["mujoco: declared 3.11.0, interpreter had 3.12.0"]
    assert rec["why"].startswith("reference version mismatch")
