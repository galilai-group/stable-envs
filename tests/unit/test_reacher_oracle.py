"""Replay the independent frozen full-State oracle; no reference wrapper needed here."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._adapters.dmcontrol import ReacherAdapter
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/reacher"


def test_reacher_oracle_hash():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert hashlib.sha256((ROOT / manifest["file"]).read_bytes()).hexdigest() == manifest["sha256"]


@pytest.mark.skipif(importlib.metadata.version("mujoco") != "3.12.0", reason="Oracle is pinned to MuJoCo 3.12.0")
def test_reacher_replays_independent_full_state_corpus():
    data = np.load(ROOT / "swm_176d545_mujoco312.npz")

    def state_at(t):
        state = TensorDict({}, [3])
        for key in data.files:
            if key.startswith("state/"):
                state[tuple(key[6:].split("/"))] = torch.from_numpy(data[key][t].copy())
        return state

    env = ReacherAdapter(3)
    try:
        env.set_state(state_at(0))
        for t, action in enumerate(data["actions"]):
            result = env.step(action)
            compare_state(result["state"], state_at(t + 1))
            for key in ("reward", "terminated", "truncated", "observation", "score"):
                torch.testing.assert_close(result[key].cpu(), torch.from_numpy(data[key][t]), atol=0, rtol=0)
    finally:
        env.close()
