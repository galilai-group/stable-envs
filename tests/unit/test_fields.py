"""``snv.Fields`` reads State fields by name suffix, and variation leaves of the environment by path."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import stable_envs as snv
from stable_envs import spaces
from stable_envs._core.state_schema import Field, StateSchema

pytestmark = pytest.mark.unit

B = 3
SCHEMA = StateSchema(
    "FieldsTest",
    [
        Field("current/body/agent/position", (2,), torch.float32, "current/float32"),
        Field("current/body/block/position", (2,), torch.float32, "current/float32"),
        Field("current/body/block/angle", (), torch.float32, "current/float32"),
        Field("goal/block/position", (2,), torch.float32, "goal/float32"),
        Field("block/angle", (), torch.float32, "goal/float32"),
        Field("step", (), torch.int64, "meta/int64"),
    ],
)


class StandIn:
    """The parts of an environment that ``Fields`` reads."""

    num_envs = B

    def __init__(self):
        self.variation_space = spaces.Dict(
            {
                "agent": spaces.Dict({"color": spaces.RGBBox((10, 20, 30))}),
                "wall": spaces.Dict({"axis": spaces.Discrete(2)}),
            }
        )
        width = self.variation_space.flat_dim
        self._rows = torch.arange(B * width, dtype=torch.float32).reshape(B, width)

    def get_variation(self):
        return self._rows


def folded():
    state = SCHEMA.new(B)
    for group in state.values(True, True):
        group.copy_(torch.arange(group.numel()).reshape(group.shape))
    return state


def test_values_equal_get_field_for_every_name():
    state = folded()
    fields = snv.Fields(state, StandIn())
    for name in SCHEMA.names:
        value = fields[name.replace("/", ".")]
        assert torch.equal(value, snv.get_field(state, name)), name
    assert torch.equal(fields.agent.position, snv.get_field(state, "current/body/agent/position"))
    assert torch.equal(fields.body.block.position, snv.get_field(state, "current/body/block/position"))
    assert torch.equal(fields.goal.block.position, snv.get_field(state, "goal/block/position"))
    assert torch.equal(fields.step, snv.get_field(state, "step"))
    assert fields.agent.position.shape == (B, 2)


def test_a_suffix_matches_whole_parts_only():
    state = folded()
    fields = snv.Fields(state)
    with pytest.raises(AttributeError):
        fields.osition
    with pytest.raises(AttributeError):
        fields.ody.agent.position


def test_an_ambiguous_chain_names_the_candidates():
    fields = snv.Fields(folded())
    with pytest.raises(ValueError, match="current/body/block/position, goal/block/position"):
        fields.block.position
    with pytest.raises(ValueError, match="current/body/agent/position"):
        fields.position
    with pytest.raises(ValueError, match="ambiguous"):
        fields["block.position"]
    # A field whose full name is the chain is not ambiguous.
    assert torch.equal(fields.block.angle, snv.get_field(folded(), "block/angle"))
    assert torch.equal(fields.body.block.angle, snv.get_field(folded(), "current/body/block/angle"))


def test_a_chain_that_ends_no_field_reads_the_variation_leaf():
    env = StandIn()
    fields = snv.Fields(folded(), env)
    rows = env.get_variation()
    layout = {path: (off, width) for path, off, width in env.variation_space.flat_layout()}
    off, width = layout["agent.color"]
    assert torch.equal(fields.agent.color, rows[:, off : off + width])
    off, _ = layout["wall.axis"]
    assert torch.equal(fields.wall.axis, rows[:, off])
    assert fields.wall.axis.shape == (B,)
    assert torch.equal(fields.agent.position, snv.get_field(folded(), "current/body/agent/position"))
    with pytest.raises(ValueError, match="batch size"):
        snv.Fields(folded()[:1], env).wall.axis
    with pytest.raises(AttributeError):
        snv.Fields(folded()).wall.axis


def test_unknown_names_raise():
    fields = snv.Fields(folded(), StandIn())
    for chain in ("nope", "agent.nope", "wall.axis.nope", "agent.position.nope"):
        with pytest.raises((AttributeError, KeyError)):
            node = fields
            for part in chain.split("."):
                node = getattr(node, part)
    with pytest.raises(AttributeError, match="nope"):
        fields.nope
    with pytest.raises(KeyError, match="nope"):
        fields["nope"]
    assert not hasattr(fields, "nope")
    assert not hasattr(fields, "_private")


def test_fields_does_not_write():
    fields = snv.Fields(folded(), StandIn())
    with pytest.raises(AttributeError, match="read-only"):
        fields.step = 1
    with pytest.raises(AttributeError, match="read-only"):
        fields.agent.position = torch.zeros(B, 2)
    with pytest.raises(TypeError):
        fields["step"] = 1  # type: ignore[index]


def test_a_state_with_one_leaf_per_field():
    state = TensorDict(
        {"current": {"agent": {"position": torch.ones(B, 2)}}, "target": {"position": torch.zeros(B, 2)}},
        batch_size=[B],
    )
    fields = snv.Fields(state)
    assert torch.equal(fields.agent.position, snv.get_field(state, "current/agent/position"))
    assert torch.equal(fields.target.position, snv.get_field(state, "target/position"))
    with pytest.raises(ValueError, match="current/agent/position, target/position"):
        fields.position


def test_dir_lists_the_next_parts():
    fields = snv.Fields(folded(), StandIn())
    assert {"agent", "block", "body", "current", "goal", "step", "wall"} <= set(dir(fields))
    assert dir(fields.agent) == ["color", "position"]
    assert dir(fields.body) == ["agent", "block"]
