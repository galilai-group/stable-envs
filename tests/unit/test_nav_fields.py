"""``snv.Fields`` on TwoRoom: State fields by name, and variation leaves from the installed rows."""

from __future__ import annotations

import pytest
import torch

import stable_envs as snv

pytestmark = pytest.mark.unit


def test_fields_reads_tworoom_state_and_variation():
    with snv.make("TwoRoom", num_envs=2, device="cpu") as env:
        state = env.reset(seed=0)
        env.set_variation({"wall.axis": 0})
        fields = snv.Fields(state, env)
        assert torch.equal(fields.agent.position, snv.get_field(state, "agent/position"))
        assert torch.equal(fields["target.position"], snv.get_field(state, "target/position"))
        layout = {path: (off, width) for path, off, width in env.variation_space.flat_layout()}
        off, width = layout["door.position"]
        assert torch.equal(fields.door.position, env.get_variation()[:, off : off + width].reshape(2, width))
        assert torch.equal(fields.wall.axis, torch.zeros(2))
        assert "door" in dir(fields)
        with pytest.raises(AttributeError, match="agent.nope"):
            fields.agent.nope
