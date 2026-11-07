"""Bounded full-State semantic gates for the first-family, continuation and Humanoid DMC registrations against pinned SWM."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.dmcontrol.simulation.continuation import CONFIG, CONTINUATION_FIELDS
from stable_envs.envs.dmcontrol.simulation.first_family import FIRST_FAMILY_FIELDS
from stable_envs.envs.dmcontrol.simulation.humanoid import HUMANOID_FIELDS

pytest.importorskip("dm_control")
pytestmark = pytest.mark.swm

from tests.helpers.reference.gates import dmc_continuation, dmc_first_family, dmc_humanoid

#: gate -> (gate module, domain -> registration, domain -> State fields, domain -> action width, device variable)
GATES = {
    "first_family": (
        dmc_first_family,
        dmc_first_family.CASES,
        FIRST_FAMILY_FIELDS,
        lambda domain: 1,
        "SE_DMC_FIRST_FAMILY_DEVICE",
    ),
    "continuation": (
        dmc_continuation,
        dmc_continuation.CASES,
        CONTINUATION_FIELDS,
        lambda domain: CONFIG[domain]["nu"],
        "SE_DMC_CONTINUATION_DEVICE",
    ),
    "humanoid": (
        dmc_humanoid,
        {"humanoid": "DMCHumanoidWalk"},
        {"humanoid": HUMANOID_FIELDS},
        lambda domain: 21,
        "SE_DMC_HUMANOID_DEVICE",
    ),
}


@pytest.mark.parametrize(
    "gate",
    [
        "first_family",
        pytest.param("continuation", marks=[pytest.mark.gl, pytest.mark.nightly]),
        pytest.param("humanoid", marks=[pytest.mark.gl, pytest.mark.nightly]),
    ],
)
def test_dmc_full_state_semantics(gate):
    module, cases, fields, nu, device = GATES[gate]
    if gate == "humanoid":
        assert "DMCHumanoidWalk" in se.list_envs()
        assert "DMCHumanoidStand" not in se.list_envs()
    for domain, env_name in cases.items():
        with se.make(env_name, device="cpu", num_envs=1) as env:
            if gate == "humanoid":
                assert env.DOMAIN == "humanoid"
                assert env.TASK == "walk"
            assert env.action_spec.dtype == torch.float64
            assert env.action_spec.shape == (nu(domain),)
            assert env.action_spec.low == -1.0
            assert env.action_spec.high == 1.0
            for key, (_, dtype) in fields[domain].items():
                assert env.state_spec[key].dtype == dtype
            env.set_variation({"light.diffuse_scale": [1.1]})
            assert np.array_equal(
                env._simulator.lanes.envs[0].physics.model.opt.gravity,
                np.array([0.0, 0.0, -9.81], np.float64),
            )
    result = module.run_gate(os.environ.get(device, "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert all(case.get("max_state_error", 0) == 0 for case in result["cases"])
    if gate == "continuation":
        assert all(
            result["warmstart_experiment"][domain]["max_observable_error_without_warmstart"] > 0 for domain in cases
        )
    if gate == "humanoid":
        assert result["mapping"] == {"swm/HumanoidDMControl-v0": "DMCHumanoidWalk"}
        assert result["warmstart_experiment"]["max_observable_error_without_warmstart"] > 0
        assert result["warmstart_experiment"]["max_error_with_schema"] == 0
