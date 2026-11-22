"""``PushT.info`` equals the info of the source step bit for bit."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pytest.importorskip("pymunk")
torch = pytest.importorskip("torch")
pytest.importorskip("pygame")
pytest.importorskip("cv2")

import stable_envs as snv
from stable_envs.envs.pusht.semantics import source_info
from stable_envs.envs.pusht.simulation.replay import PushTReplay

pytestmark = pytest.mark.unit

KEYS = ("pos_agent", "vel_agent", "block_pose", "goal_pose", "n_contacts")


def _same(got, want):
    for key in KEYS:
        value = np.asarray(want[key])
        assert got[key].dtype == (np.int64 if key == "n_contacts" else value.dtype), key
        assert np.array_equal(got[key], value), key


@pytest.mark.parametrize("seed", [0, 7])
def test_info_of_source_states_equals_source_info(seed):
    replay = PushTReplay(config={"resolution": 32}, seed=seed)
    try:
        source = replay.env
        n_steps = int(1 / (source.dt * source.control_hz))
        contacts = 0
        for _ in range(60):
            agent, block = np.asarray(source.agent.position), np.asarray(source.block.position)
            action = np.clip((block - agent) / 100, -1, 1).astype(np.float32)
            info = replay.step(action)[4]
            state = replay.get_state()
            got = source_info(lambda name: np.asarray(state[name])[None], n_steps)
            _same({k: v[0] for k, v in got.items()}, info)
            contacts += info["n_contacts"]
        assert contacts > 0
    finally:
        replay.close()


def test_env_info_after_reset_equals_the_source_reset_info():
    env = snv.PushT(num_envs=3, device="cpu")
    try:
        info = env.info(env.reset(seed=4))
        assert info.batch_size == torch.Size([3])
        for i, owner in enumerate(env._simulator.envs):
            _same({k: info[k][i].numpy() for k in KEYS}, owner.env._get_info())
    finally:
        env.close()


def test_info_of_rows_equals_the_rows_of_info():
    env = snv.PushT(num_envs=4, device="cpu")
    try:
        env.reset(seed=2)
        state = env.step(torch.full((4, 3, 2), 0.5))["state"]
        full, rows = env.info(state), torch.tensor([2, 0])
        part = env.info(state[rows])
        for key in KEYS:
            assert torch.equal(part[key], full[key][rows]), key
    finally:
        env.close()
