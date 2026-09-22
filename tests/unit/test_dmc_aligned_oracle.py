"""Replay immutable reference-only aligned DMC oracles."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._adapters.dmcontrol_aligned import DMCAlignedAdapter
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/dmc_aligned"


def test_aligned_oracle_hashes_and_pin():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["reference_commit"] == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
    for entry in manifest["domains"].values():
        assert hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.skipif(importlib.metadata.version("mujoco") != "3.12.0", reason="MuJoCo-pinned oracle")
@pytest.mark.parametrize("domain", ["ball_in_cup", "finger", "manipulator", "quadruped"])
def test_aligned_oracle_replays_every_output_leaf(domain):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    data = np.load(ROOT / manifest["domains"][domain]["file"])
    state = TensorDict({}, [3])
    for key in data.files:
        if key.startswith("state/"):
            state[tuple(key[6:].split("/"))] = torch.from_numpy(data[key][0].copy())
    env = DMCAlignedAdapter(domain, 3)
    try:
        env.set_state(state)
        for t, action in enumerate(data["actions"]):
            result = env.step(action)
            expected = TensorDict({}, [3])
            for key in data.files:
                if key.startswith("state/"):
                    expected[tuple(key[6:].split("/"))] = torch.from_numpy(data[key][t + 1].copy())
            compare_state(result["state"], expected)
            for key in result.keys(True, True):
                if key == "state" or (isinstance(key, tuple) and key[0] == "state"):
                    continue
                name = key if isinstance(key, str) else "/".join(key)
                torch.testing.assert_close(result[key].cpu(), torch.from_numpy(data[f"output/{name}"][t]), atol=0, rtol=0, equal_nan=True)
    finally:
        env.close()
