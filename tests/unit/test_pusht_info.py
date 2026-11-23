"""``PushT.source_info`` equals the info of the source step bit for bit."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pymunk")
torch = pytest.importorskip("torch")
pytest.importorskip("pygame")
pytest.importorskip("cv2")

import stable_envs as snv
from stable_envs.envs.pusht.simulation.replay import PushTReplay

pytestmark = pytest.mark.unit

KEYS = ("pos_agent", "vel_agent", "block_pose", "goal_pose", "n_contacts")


def _same(got, want):
    for key in KEYS:
        value = np.asarray(want[key])
        assert got[key].dtype == (np.int64 if key == "n_contacts" else value.dtype), key
        assert np.array_equal(got[key], value), key


@pytest.mark.parametrize("seed", [0, 7])
def test_source_info_of_stepped_source_states_equals_source_info(seed):
    """``PushT.source_info`` on States of the source engine, stepped with Pymunk into contact with the block."""
    replay = PushTReplay(seed=seed)
    env = snv.PushT(num_envs=1, device="cpu")
    try:
        base = env.reset(seed=seed)
        schema = snv.schema_of(base)
        source = replay.env
        contacts = 0
        for _ in range(60):
            agent, block = np.asarray(source.agent.position), np.asarray(source.block.position)
            action = np.clip((block - agent) / 100, -1, 1).astype(np.float32)
            info = replay.step(action)[4]
            values = {k: torch.as_tensor(np.asarray(v))[None] for k, v in replay.get_state().items()}
            values.pop("current/goal_image")
            # The Numba State also keeps a contact cache of the batched solver, which the source has not.
            values.update({name: snv.get_field(base, name) for name in schema.names if name not in values})
            got = env.source_info(schema.fold(values, [1]))
            _same({k: got[k][0].numpy() for k in KEYS}, info)
            contacts += info["n_contacts"]
        assert contacts > 0
    finally:
        env.close()
        replay.close()


def test_source_info_after_reset_equals_the_source_reset_info():
    env = snv.PushT(num_envs=3, device="cpu")
    try:
        info = env.source_info(env.reset(seed=4))
        assert info.batch_size == torch.Size([3])
        for i, owner in enumerate(env._simulator.envs):
            _same({k: info[k][i].numpy() for k in KEYS}, owner.env._get_info())
    finally:
        env.close()


def test_source_info_of_rows_equals_the_rows_of_source_info():
    env = snv.PushT(num_envs=4, device="cpu")
    try:
        env.reset(seed=2)
        state = env.step(torch.full((4, 3, 2), 0.5))["state"]
        full, rows = env.source_info(state), torch.tensor([2, 0])
        part = env.source_info(state[rows])
        for key in KEYS:
            assert torch.equal(part[key], full[key][rows]), key
    finally:
        env.close()


def test_source_info_refuses_a_state_of_another_class():
    env = snv.PushT(num_envs=1, device="cpu")
    try:
        with pytest.raises(TypeError, match="PushT State"):
            env.source_info(torch.zeros(1, 3))
    finally:
        env.close()
