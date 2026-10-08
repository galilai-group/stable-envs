"""Bounded semantic gate for historical Humanoid Walk."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs._adapters.dmc.humanoid import HUMANOID_FIELDS

pytest.importorskip("dm_control")
pytest.importorskip("stable_worldmodel")

from scripts.recovery.dmc_humanoid_gate import run_gate


def test_humanoid_walk_identity_specs_and_full_state_semantics():
    assert "DMCHumanoidWalk" in se.list_envs()
    assert "DMCHumanoidStand" not in se.list_envs()
    with se.make("DMCHumanoidWalk", device="cpu", num_envs=1) as env:
        assert env.DOMAIN == "humanoid"
        assert env.TASK == "walk"
        assert env.action_spec.dtype == torch.float64
        assert env.action_spec.shape == (21,)
        assert env.action_spec.low == -1.0
        assert env.action_spec.high == 1.0
        for key, (_, dtype) in HUMANOID_FIELDS.items():
            assert env.state_spec[key].dtype == dtype
        env.set_variation({"light.diffuse_scale": [1.1]})
        assert np.array_equal(
            env._simulator.envs[0].physics.model.opt.gravity,
            np.array([0.0, 0.0, -9.81], np.float64),
        )

    result = run_gate(os.environ.get("SE_DMC_HUMANOID_DEVICE", "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert result["mapping"] == {"swm/HumanoidDMControl-v0": "DMCHumanoidWalk"}
    assert all(case.get("max_state_error", 0) == 0 for case in result["cases"])
    assert result["warmstart_experiment"]["max_observable_error_without_warmstart"] > 0
    assert result["warmstart_experiment"]["max_error_with_schema"] == 0
