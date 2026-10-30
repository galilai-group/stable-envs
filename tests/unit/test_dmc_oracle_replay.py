"""Each DM Control corpus replays through the batch adapter, from any reached State."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import stable_envs as se
from stable_envs.envs.dmcontrol.simulation.factory import make_batch
from tests.helpers.reference.crosswalk import canonical_names, class_for_frozen_cell
from tests.helpers.reference.oracle import assert_outputs, state_at
from tests.helpers.reference.saved import FROZEN_FLOAT64_ATOL
from tests.helpers.reference.semantics import compare_state

FIXTURES = Path(__file__).parents[1] / "fixtures/semantics"
REFERENCE_COMMIT = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
FIRST_FAMILY_KEYS = ("reward", "terminated", "truncated", "observation", "score")

# (corpus directory, domain, adapter factory, assert_outputs kwargs)
CORPORA = [
    *[
        ("dmc_first_family", d, lambda b, d=d: make_batch(d, b), dict(keys=FIRST_FAMILY_KEYS, equal_nan=False))
        for d in ("acrobot", "pendulum", "cartpole")
    ],
    *[
        ("dmc_aligned", d, lambda b, d=d: make_batch(d, b), dict(equal_nan=True))
        for d in ("ball_in_cup", "finger", "manipulator", "quadruped")
    ],
    *[
        ("dmc_continuation", d, lambda b, d=d: make_batch(d, b), dict(equal_nan=True))
        for d in ("cheetah", "hopper", "walker")
    ],
    ("dmc_humanoid", None, lambda b: make_batch("humanoid", b), dict(equal_nan=True)),
    ("reacher", None, lambda b: make_batch("reacher", b), dict(keys=FIRST_FAMILY_KEYS, equal_nan=False)),
]


def _entry(directory: str, domain: str | None) -> dict:
    manifest = json.loads((FIXTURES / directory / "manifest.json").read_text())
    if directory == "reacher":
        return manifest
    return manifest["domain"] if domain is None else manifest["domains"][domain]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "directory,domain", [(d, dom) for d, dom, _, _ in CORPORA], ids=[f"{d}-{dom}" for d, dom, _, _ in CORPORA]
)
def test_corpus_file_matches_its_manifest_hash(directory, domain):
    entry = _entry(directory, domain)
    assert _sha256(FIXTURES / directory / entry["file"]) == entry["sha256"]


@pytest.mark.parametrize("directory", sorted({d for d, *_ in CORPORA} - {"reacher"}))
def test_corpora_are_recorded_against_the_pinned_swm_commit(directory):
    manifest = json.loads((FIXTURES / directory / "manifest.json").read_text())
    assert manifest["reference_commit"] == REFERENCE_COMMIT


@pytest.mark.parametrize(
    "directory,domain,factory,compare",
    CORPORA,
    ids=[f"{d}-{dom}" for d, dom, _, _ in CORPORA],
)
def test_adapter_replays_every_recorded_step_from_restored_states(directory, domain, factory, compare):
    entry = _entry(directory, domain)
    with np.load(FIXTURES / directory / entry["file"], allow_pickle=False) as data:
        env = factory(3)
        try:
            # Each step starts from the recorded state, so platform roundoff cannot compound
            # through chaotic dynamics; the solver's warm start is only a starting guess.
            for t, action in enumerate(data["actions"]):
                expected = state_at(data, t, batch=3, physics=env.physics_layout)
                env.set_state(expected)
                compare_state(env.get_state(), expected)
                result = env.step(action)
                compare_state(
                    result["state"].exclude("warmstart"),
                    state_at(data, t + 1, batch=3, physics=env.physics_layout).exclude("warmstart"),
                    atol=FROZEN_FLOAT64_ATOL,
                )
                assert_outputs(result, data, t, **compare)
        finally:
            env.close()


def test_acrobot_identity_is_balance_and_the_old_public_name_is_gone():
    root = FIXTURES / "dmc_first_family"
    identity = json.loads((root / "acrobot_identity.json").read_text())
    assert identity["task"] == "dm_control.suite.acrobot.Balance(sparse=True)"
    assert _sha256(root / identity["oracle_manifest"]) == identity["oracle_manifest_sha256"]
    assert canonical_names("swm/AcrobotDMControl-v0") == ("DMCAcrobotBalance",)
    assert class_for_frozen_cell("DMCAcrobotSwingup") is se.DMCAcrobotBalance
    assert se.info("DMCAcrobotBalance")["reference_id"] == "dm_control/acrobot-swingup_sparse"
    with pytest.raises(KeyError):
        se.make("DMCAcrobotSwingup", device="cpu")
