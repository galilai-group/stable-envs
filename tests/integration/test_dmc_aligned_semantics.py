"""Integrated exact gate and public specs for the aligned DMC cohort."""

import os

import pytest
import torch

import stable_envs as se
from stable_envs.envs.dmcontrol.simulation.aligned import ALIGNED_FIELDS, CONFIG

pytest.importorskip("dm_control")
pytest.importorskip("stable_worldmodel")
from tests.helpers.reference.gates.dmc_aligned import CASES, run_gate


def test_aligned_full_state_specs_and_gate():
    for domain, name in CASES.items():
        with se.make(name, device="cpu", num_envs=1) as env:
            assert env.action_spec.dtype == torch.float64
            assert env.action_spec.shape == (CONFIG[domain]["nu"],)
            for key, (_, dtype) in ALIGNED_FIELDS[domain].items():
                assert env.state_spec[key].dtype == dtype
    result = run_gate(os.environ.get("SE_DMC_ALIGNED_DEVICE", "cpu"))
    assert result["execution_mode"] == "DELEGATED_CPU"
    assert all(case.get("max_error", 0) == 0 for case in result["cases"])
