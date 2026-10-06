import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest

from stable_envs._adapters.dmcontrol import ReacherAdapter
from tests.helpers.reference.oracle import assert_outputs, state_at
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/reacher"


def test_reacher_oracle_hash():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert hashlib.sha256((ROOT / manifest["file"]).read_bytes()).hexdigest() == manifest["sha256"]


@pytest.mark.skipif(importlib.metadata.version("mujoco") != "3.12.0", reason="Oracle is pinned to MuJoCo 3.12.0")
def test_reacher_replays_independent_full_state_corpus():
    with np.load(ROOT / "swm_176d545_mujoco312.npz", allow_pickle=False) as data:
        env = ReacherAdapter(3)
        try:
            # Restore reached source states after a different candidate history.
            for start in (0, 7, 24):
                env.set_state(state_at(data, start, batch=3))
                compare_state(env.get_state(), state_at(data, start, batch=3))
                for t, action in enumerate(data["actions"][start:], start=start):
                    result = env.step(action)
                    compare_state(result["state"], state_at(data, t + 1, batch=3))
                    assert_outputs(
                        result,
                        data,
                        t,
                        keys=("reward", "terminated", "truncated", "observation", "score"),
                        equal_nan=False,
                    )
        finally:
            env.close()
