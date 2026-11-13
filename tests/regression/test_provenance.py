"""``stable_envs.testing.provenance``: a bound measured against one reference version is not judged
against another.

A dependency failure is not a fidelity failure: a reference at the wrong version (for example mujoco
3.12.0, whose finite-plane texture re-anchoring phase-shifts the floor checker declared under 3.11.0)
is a comparison that could not be made.
"""

from __future__ import annotations

import pytest

from stable_envs.testing.provenance import (
    REFERENCE_JUDGED_CHECKS,
    block_on_reference_mismatch,
    normalize_package,
    reference_version_mismatch,
)

DECLARED = {"mujoco": "3.11.0", "mujoco_warp": "3.11.0"}


def test_matching_versions_are_not_a_mismatch():
    assert reference_version_mismatch(DECLARED, {"mujoco": "3.11.0", "mujoco_warp": "3.11.0", "torch": "2.14.0"}) == []


def test_a_different_version_is_named_with_both_sides():
    bad = reference_version_mismatch(DECLARED, {"mujoco": "3.12.0", "mujoco_warp": "3.11.0"})
    assert bad == ["mujoco: declared 3.11.0, interpreter had 3.12.0"]


def test_an_unrecorded_or_unknown_version_cannot_confirm_a_match():
    """A package with no ``__version__`` (recorded as "present") cannot confirm a match."""
    for recorded in ({}, {"mujoco": None}, {"mujoco": "present"}, {"mujoco": ""}):
        bad = reference_version_mismatch({"mujoco": "3.11.0"}, recorded)
        assert bad == ["mujoco: declared 3.11.0, version not recorded by the measuring interpreter"], recorded


def test_nothing_declared_is_not_a_mismatch():
    """With no versions declared or recorded, nothing is a mismatch."""
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


import ast  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from stable_envs.testing import conformance as C  # noqa: E402
from tests._runner import render_worker as W  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _repro_constants() -> dict:
    """``N`` and ``MEASURED_ON`` out of tests/gpu/test_reproducibility.py, read as text: importing the
    module parameterizes fixtures over the registry and needs a GPU-capable interpreter."""
    src = (ROOT / "tests" / "gpu" / "test_reproducibility.py").read_text()
    out = {}
    for n in ast.parse(src).body:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("N", "MEASURED_ON", "DOWNSAMPLE_MARGIN")
        ):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def test_the_downsample_protocol_is_the_one_the_bound_was_measured_under():
    """``DOWNSAMPLE_EXACT_FRAC`` was measured by the GPU test module's ``inputs`` fixture:
    ``render_worker.sample_inputs(env, N)`` at the default appearance. The library's copy of that
    protocol must be the same N and the same seed, or the floor is applied to a different draw."""
    k = _repro_constants()
    assert k["N"] == C.DOWNSAMPLE_PROTOCOL_N
    assert f"{C.DOWNSAMPLE_PROTOCOL_N} states per env" in k["MEASURED_ON"]
    assert "render_worker.STATE_SEED" in k["MEASURED_ON"] and "default appearance" in k["MEASURED_ON"]
    proto = C.downsample_protocol()
    assert proto["n"] == k["N"] and proto["state_seed"] == W.STATE_SEED
    assert proto["style"] == "default appearance"


def test_the_protocol_states_are_the_fixture_draw():
    """Bit-for-bit what ``render_worker.sample_inputs`` gives, and nothing the caller passed."""

    class Env:
        style_dim = 0

        def sample_states(self, n, rng=None):
            return rng.standard_normal((n, 3)).astype(np.float32)

    env = Env()
    got = C.downsample_protocol_states(env)
    want, _ = W.sample_inputs(env, C.DOWNSAMPLE_PROTOCOL_N)
    assert got.shape == (C.DOWNSAMPLE_PROTOCOL_N, 3) and got.dtype == np.float32
    assert np.array_equal(got, want)
    assert np.array_equal(got, C.downsample_protocol_states(env))  # deterministic
    rec = C.downsample_protocol(got)
    assert rec["states_sha256"] == C.frames_digest(got) and rec["n_states"] == C.DOWNSAMPLE_PROTOCOL_N
    assert "states_sha256" not in C.downsample_protocol()  # nothing pinned without a draw
