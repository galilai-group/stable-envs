import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stable_envs._core.physics import state_key
from stable_envs.envs.dmcontrol.simulation.first_family import FirstFamilyAdapter
from tests.helpers.reference.oracle import assert_outputs, state_at
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/dmc_first_family"


def _registered_physics(state, layout):
    """The corpus predates State physics fields; it was recorded at the wrappers' registered physics."""
    batch = state.batch_size[0]
    for leaf, offset, width in zip(layout.leaves, layout.offsets, layout.widths, strict=True):
        value = torch.tensor(layout.nominal[offset : offset + width], dtype=torch.float64)
        state[state_key(leaf)] = value.expand(batch, width).clone()
    return state


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
    with np.load(ROOT / manifest["domains"][domain]["file"], allow_pickle=False) as data:
        env = FirstFamilyAdapter(domain, 3)
        try:
            # Restore reached source states after a different candidate history.
            for start in (0, 7, 24):
                expected = _registered_physics(state_at(data, start, batch=3), env.physics_layout)
                env.set_state(expected)
                compare_state(env.get_state(), expected)
                for t, action in enumerate(data["actions"][start:], start=start):
                    result = env.step(action)
                    compare_state(
                        result["state"], _registered_physics(state_at(data, t + 1, batch=3), env.physics_layout)
                    )
                    assert_outputs(
                        result,
                        data,
                        t,
                        keys=("reward", "terminated", "truncated", "observation", "score"),
                        equal_nan=False,
                    )
        finally:
            env.close()
