"""``snv.Fields`` on a TwoRoom State."""

from __future__ import annotations

import pytest
import torch

import stable_envs as snv

pytestmark = pytest.mark.unit


def test_fields_reads_tworoom_state():
    with snv.make("TwoRoom", num_envs=2, device="cpu") as env:
        state = env.reset(seed=0)
        fields = snv.Fields(state)
        assert torch.equal(fields.agent.position, snv.get_field(state, "agent/position"))
        assert torch.equal(fields["target/position"], snv.get_field(state, "target/position"))
        assert set(dir(fields)) >= {"agent", "target"}
        with pytest.raises(AttributeError, match="agent.nope"):
            fields.agent.nope
