"""Bounded semantic gate for three pinned historical DMC registrations."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs._adapters.dmc.continuation import CONFIG, CONTINUATION_FIELDS

pytest.importorskip("dm_control")
pytest.importorskip("stable_worldmodel")

from scripts.recovery.dmc_continuation_gate import CASES, run_gate


def test_continuation_dmc_full_state_semantics():
    for domain, env_name in CASES.items():
        with se.make(env_name, device="cpu", num_envs=1) as env:
            assert env.action_spec.dtype == torch.float64
            assert env.action_spec.shape == (CONFIG[domain]["nu"],)
            assert env.action_spec.low == -1.0
            assert env.action_spec.high == 1.0
            for key, (_, dtype) in CONTINUATION_FIELDS[domain].items():
                assert env.state_spec[key].dtype == dtype
            env.set_variation({"light.diffuse_scale": [1.1]})
            assert np.array_equal(
                env._simulator.envs[0].physics.model.opt.gravity,
                np.array([0.0, 0.0, -9.81], np.float64),
            )
    result = run_gate(os.environ.get("SE_DMC_CONTINUATION_DEVICE", "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert all(case.get("max_state_error", 0) == 0 for case in result["cases"])
    assert all(
        result["warmstart_experiment"][domain]["max_observable_error_without_warmstart"] > 0
        for domain in CASES
    )
