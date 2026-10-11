import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from stable_envs.envs.dmcontrol.simulation.aligned import DMCAlignedAdapter
from tests.helpers.reference.oracle import assert_outputs, state_at
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
    with np.load(ROOT / manifest["domains"][domain]["file"], allow_pickle=False) as data:
        env = DMCAlignedAdapter(domain, 3)
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
