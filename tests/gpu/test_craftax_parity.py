"""Craftax on CUDA against upstream Craftax on the same GPU, one jitted lane at a time.

Every lane steps for one long rollout, past the episode ends, as upstream steps without auto-reset.
States, rewards, ends, observations and frames must be equal bit for bit. The CPU tests make the same
comparison on the CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, assert_lane, frame, upstream  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.reference]


def _gpu():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return None


GPU = _gpu()
if not torch.cuda.is_available() or GPU is None:
    pytest.skip("Craftax GPU parity needs CUDA and the craftax-gpu extra", allow_module_level=True)

LANES = 8
STEPS = 256
SEED = 20260811


@pytest.mark.parametrize("name", NAMES)
def test_every_lane_matches_upstream_on_the_gpu_over_a_long_rollout(name):
    from craftax.craftax_env import make_craftax_env_from_name

    with se.make(name, num_envs=LANES, device="cuda") as env, jax.default_device(GPU):
        reset, step = upstream(env.reference_id)
        pixels = make_craftax_env_from_name(env.reference_id.replace("Symbolic", "Pixels"), auto_reset=False)
        draw = jax.jit(pixels.get_obs)
        lanes = [list(reset(jax.random.PRNGKey(SEED + lane))[1:]) for lane in range(LANES)]
        state = env.reset(seed=SEED)
        for lane, (engine, key) in enumerate(lanes):
            assert_lane(state, lane, engine, key)
        rng = np.random.default_rng(SEED)
        ends = 0
        for t in range(STEPS):
            action = torch.as_tensor(rng.integers(0, env.action_high + 1, (LANES, 1)), device="cuda")
            result = env.step(action)
            frames = result["frames"][:, -1].cpu().numpy()
            for lane in range(LANES):
                obs, engine, reward, done, key = step(lanes[lane][1], lanes[lane][0], int(action[lane, 0]))
                lanes[lane] = [engine, key]
                assert_lane(result["state"], lane, engine, key)
                assert result["reward"][lane].item() == float(reward), (t, lane)
                assert result["terminated"][lane].item() == bool(done), (t, lane)
                ends += bool(done)
                np.testing.assert_array_equal(frames[lane], frame(draw(engine)), err_msg=f"step {t} lane {lane}")
                if name.endswith("Symbolic"):
                    np.testing.assert_array_equal(result["observation"][lane].cpu().numpy(), np.asarray(obs))
        assert ends, "no lane reached the end of an episode"


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxPixels"])
def test_a_cuda_step_reads_nothing_back_to_the_host(name):
    with se.make(name, num_envs=LANES, device="cuda") as env:
        env.reset(seed=SEED)
        action = torch.zeros((LANES, 1), dtype=torch.int64, device="cuda")
        env.step(action, frames=None)
        with jax.transfer_guard_device_to_host("disallow"):
            for _ in range(3):
                env.step(action, frames=None)


def test_temporary_device_inputs_live_until_the_step_has_read_them():
    """Actions and render blocks cross into JAX through DLPack as temporaries that torch frees right after.

    Torch reuses the freed memory at once, while the GPU is still busy with earlier work. A step or a
    render that read the memory too late would see other values.
    """
    name, lanes = "CraftaxClassicPixels", 512
    rng = np.random.default_rng(SEED)
    sequence = rng.integers(0, 17, (12, lanes, 1))
    with (
        se.make(name, num_envs=lanes, device="cuda") as env,
        se.make(name, num_envs=lanes, device="cuda") as reference,
    ):
        env.reset(seed=SEED)
        reference.reset(seed=SEED)
        for action in sequence:
            out = env.step(torch.as_tensor(action, device="cuda"), frames=None)
            torch.full((1 << 24,), 255, dtype=torch.uint8, device="cuda")
            expected = reference.step(action, frames=None)
            torch.cuda.synchronize()
            assert_same_state(out, expected)
        state = env.get_state()
        frames = env.render(state[:5])
        torch.full((1 << 24,), 255, dtype=torch.uint8, device="cuda")
        assert torch.equal(frames, reference.render()[:5])
