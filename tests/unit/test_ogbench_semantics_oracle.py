"""Replay the independent frozen OGBMaze full-State oracle without importing SWM."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._adapters.ogbench.maze import OGBMazeAdapter
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).parents[1] / "fixtures/semantics/ogbench_maze"


def _np_state_equal(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


def test_ogbench_maze_oracle_hash():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert hashlib.sha256((ROOT / manifest["file"]).read_bytes()).hexdigest() == manifest["sha256"]


@pytest.mark.skipif(importlib.metadata.version("mujoco") != "3.12.0", reason="Oracle is pinned to MuJoCo 3.12.0")
def test_ogbench_maze_replays_independent_full_state_corpus():
    pytest.importorskip("ogbench")
    data = np.load(ROOT / "swm_176d545_ogbench121_mujoco312.npz")

    def state_at(t):
        state = TensorDict({}, [3])
        for key in data.files:
            if key.startswith("state/"):
                state[tuple(key[6:].split("/"))] = torch.from_numpy(data[key][t].copy())
        return state

    env = OGBMazeAdapter(3)
    try:
        env.set_state(state_at(0))
        for t, action in enumerate(data["actions"]):
            result = env.step(action)
            compare_state(result["state"], state_at(t + 1))
            for key in data.files:
                if key.startswith("output/"):
                    path = tuple(key[7:].split("/"))
                    torch.testing.assert_close(result[path].cpu(), torch.from_numpy(data[key][t]), atol=0, rtol=0)
    finally:
        env.close()


def test_ogbench_maze_adapter_isolates_legacy_rng_and_validates_before_mutation():
    pytest.importorskip("ogbench")
    outside = np.random.get_state()
    env = OGBMazeAdapter(3, seed=17)
    try:
        assert _np_state_equal(outside, np.random.get_state())
        saved = env.get_state()
        incomplete = saved.clone()
        del incomplete["rng", "env_state"]
        with pytest.raises(ValueError, match="Full State missing"):
            env.set_state(incomplete)
        compare_state(env.get_state(), saved)
        invalid = saved.clone()
        invalid["task", "id"][2, 0] = 99
        with pytest.raises(ValueError, match="Invalid OGBench task id"):
            env.set_state(invalid)
        compare_state(env.get_state(), saved)
        env.reset(seed=None)
        env.step(np.zeros((3, 2), np.float32))
        assert _np_state_equal(outside, np.random.get_state())
    finally:
        env.close()
