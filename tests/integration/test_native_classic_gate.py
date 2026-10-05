"""Frozen real-reference corpus for classic-control defaults."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.native_acceptance import CASES


@pytest.mark.parametrize("domain", ["cartpole", "pendulum"])
def test_frozen_reference_oracle(domain):
    root = Path(__file__).parents[1] / "fixtures" / "semantics" / "native_classic"
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["domains"][domain]
    path = root / entry["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    oracle = np.load(path)
    with se.make(CASES[domain]["stable_env"], num_envs=3, device="cpu") as env:
        states = [env.reset(seed=manifest["seed"])]
        outputs = []
        for action in oracle["actions"]:
            output = env.step(action)
            outputs.append(output)
            states.append(output["state"])
        for key in states[0].keys(True, True):
            name = key if isinstance(key, str) else "/".join(key)
            actual = torch.stack([state[key] for state in states]).cpu().numpy()
            assert np.array_equal(actual, oracle[f"state/{name}"])
        for key in ("reward", "terminated", "truncated", "observation", "info_state"):
            actual = torch.stack([output[key] for output in outputs]).cpu().numpy()
            assert np.array_equal(actual, oracle[key])
