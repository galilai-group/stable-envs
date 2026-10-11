import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from stable_envs.envs.dmcontrol.simulation.humanoid import DMCHumanoidAdapter
from tests.helpers.reference.oracle import assert_outputs, state_at
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/dmc_humanoid"


def test_dmc_humanoid_oracle_hash():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["reference_commit"] == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
    entry = manifest["domain"]
    assert hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]


@pytest.mark.skipif(importlib.metadata.version("mujoco") != "3.12.0", reason="Oracle is pinned to MuJoCo 3.12.0")
def test_dmc_humanoid_replays_independent_full_output_corpus():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    with np.load(ROOT / manifest["domain"]["file"], allow_pickle=False) as data:
        env = DMCHumanoidAdapter(3)
        try:
            # Restore reached source states after a different candidate history.
            for start in (0, 7, 24):
                env.set_state(state_at(data, start, batch=3, physics=env.physics_layout))
                compare_state(env.get_state(), state_at(data, start, batch=3, physics=env.physics_layout))
                for t, action in enumerate(data["actions"][start:], start=start):
                    result = env.step(action)
                    compare_state(result["state"], state_at(data, t + 1, batch=3, physics=env.physics_layout))
                    assert_outputs(result, data, t, equal_nan=True)
        finally:
            env.close()
