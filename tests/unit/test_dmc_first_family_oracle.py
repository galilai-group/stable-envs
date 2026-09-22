"""Replay immutable reference-only oracles for the bounded DMC cohort."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._adapters.dmcontrol import FirstFamilyAdapter
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/dmc_first_family"


def test_acrobot_identity_preserves_oracle_and_rejects_old_public_name():
    import stable_envs as se
    from tests.helpers.reference.crosswalk import canonical_names, class_for_frozen_cell

    identity = json.loads((ROOT / "acrobot_identity.json").read_text())
    assert identity["task"] == "dm_control.suite.acrobot.Balance(sparse=True)"
    assert (
        hashlib.sha256((ROOT / identity["oracle_manifest"]).read_bytes()).hexdigest()
        == identity["oracle_manifest_sha256"]
    )
    assert canonical_names("swm/AcrobotDMControl-v0") == ("DMCAcrobotBalance",)
    assert class_for_frozen_cell("DMCAcrobotSwingup") is se.DMCAcrobotBalance
    assert se.info("DMCAcrobotBalance")["reference_id"] == "dm_control/acrobot-swingup_sparse"
    with pytest.raises(KeyError):
        se.make("DMCAcrobotSwingup", device="cpu")


def test_dmc_first_family_oracle_hashes():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for entry in manifest["domains"].values():
        assert hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.skipif(importlib.metadata.version("mujoco") != "3.12.0", reason="Oracle is pinned to MuJoCo 3.12.0")
@pytest.mark.parametrize("domain", ["acrobot", "pendulum", "cartpole"])
def test_dmc_first_family_replays_independent_full_state_corpus(domain):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    data = np.load(ROOT / manifest["domains"][domain]["file"])

    def state_at(t):
        state = TensorDict({}, [3])
        for key in data.files:
            if key.startswith("state/"):
                state[tuple(key[6:].split("/"))] = torch.from_numpy(data[key][t].copy())
        return state

    env = FirstFamilyAdapter(domain, 3)
    try:
        env.set_state(state_at(0))
        for t, action in enumerate(data["actions"]):
            result = env.step(action)
            compare_state(result["state"], state_at(t + 1))
            for key in ("reward", "terminated", "truncated", "observation", "score"):
                torch.testing.assert_close(result[key].cpu(), torch.from_numpy(data[key][t]), atol=0, rtol=0)
    finally:
        env.close()
