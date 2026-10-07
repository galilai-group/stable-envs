"""Trackcom GPU batches preserve saved full-State frames across context capacities.

Source-pixel correctness is tested separately by the maintained DMC contract.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import stable_envs as se

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/semantics"
CASES = (
    ("DMCWalkerWalk", "dmc_continuation/walker", 64),
    ("DMCHopperHop", "dmc_continuation/hopper", 64),
    ("DMCCheetahRun", "dmc_continuation/cheetah", 64),
    ("DMCHumanoidWalk", "dmc_humanoid/humanoid_walk", 64),
    ("DMCQuadrupedWalk", "dmc_aligned/quadruped", 64),
    ("DMCQuadrupedWalk", "dmc_aligned/quadruped", 84),
    ("DMCQuadrupedWalk", "dmc_aligned/quadruped", 85),
    ("DMCQuadrupedWalk", "dmc_aligned/quadruped", 224),
)


def _saved_states(fixture: str) -> TensorDict:
    with np.load(FIXTURES / f"{fixture}_swm_176d545_mujoco312.npz", allow_pickle=False) as saved:
        shape = saved["state/qpos"].shape[:2]
        state = TensorDict(
            {
                tuple(key.removeprefix("state/").split("/")): torch.from_numpy(saved[key])
                for key in saved.files
                if key.startswith("state/")
            },
            batch_size=shape,
        ).reshape(-1)
    # Seven distinct rows make the repeating pattern straddle power-of-two boundaries.
    return state[torch.linspace(0, len(state) - 1, 7).long()].to("cuda")


@pytest.mark.parametrize("name,fixture,size", CASES, ids=[f"{name}-{size}" for name, _, size in CASES])
def test_gpu_batch_capacity_and_per_row_style(name: str, fixture: str, size: int):
    pytest.importorskip("dm_control")
    pytest.importorskip("mujoco_warp")
    states = _saved_states(fixture)
    with se.make(name, num_envs=1, size=size, device="cuda") as env:
        styles = env.sample_style(len(states), rng=np.random.default_rng(711))
        reference = torch.cat([env.render(states[i : i + 1], style=styles[i : i + 1]) for i in range(len(states))])
        assert reference.shape == (len(states), size, size, 3)
        assert reference.dtype == torch.uint8 and reference.device.type == "cuda"
        assert not torch.equal(reference, env.render(states)), "Appearance variation did not change any pixels"
        for batch in (1, 3, 9, 63, 64, 65, 255, 256, 257):
            indices = torch.arange(batch, device="cuda") % len(states)
            for order in (indices, indices.flip(0)):
                actual = env.render(states[order], style=styles[order.cpu().numpy()])
                torch.testing.assert_close(actual, reference[order], rtol=0, atol=0)
