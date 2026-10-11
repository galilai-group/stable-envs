"""Column export retains owned snapshots, typed fields and lane RNG replay."""

from functools import partial

import numpy as np
import pytest
import torch

from stable_envs.envs.dmcontrol.simulation.aligned import CONFIG as ALIGNED
from stable_envs.envs.dmcontrol.simulation.aligned import DMCAlignedAdapter
from stable_envs.envs.dmcontrol.simulation.continuation import CONFIG as CONTINUATION
from stable_envs.envs.dmcontrol.simulation.continuation import DMCContinuationAdapter
from stable_envs.envs.dmcontrol.simulation.first_family import FirstFamilyAdapter, ReacherAdapter
from stable_envs.envs.dmcontrol.simulation.humanoid import DMCHumanoidAdapter

CASES = [(FirstFamilyAdapter, (domain,)) for domain in ("acrobot", "pendulum", "cartpole", "cartpole-balance")] + [
    (ReacherAdapter, ()),
    (partial(ReacherAdapter, task="easy"), ()),
    *[(DMCAlignedAdapter, (domain,)) for domain in ALIGNED],
    *[(DMCContinuationAdapter, (domain,)) for domain in CONTINUATION],
    (DMCHumanoidAdapter, ()),
]


def equal(left, right):
    assert left.batch_size == right.batch_size
    assert set(left.keys(True, True)) == set(right.keys(True, True))
    for key in left.keys(True, True):
        assert left[key].dtype == right[key].dtype
        assert left[key].device == right[key].device
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True, msg=str(key))


@pytest.mark.parametrize("factory,args", CASES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_owned_typed_state_across_steps_reset_and_rng_boundary(factory, args, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA gate runs on Oscar")
    adapter = factory(*args, 3, device=device, seed=17)
    try:
        initial = adapter.get_state()
        retained = initial.clone()
        actions = np.full((3, adapter.envs[0].action_spec().shape[0]), 0.25)
        adapter.step(actions)
        adapter.reset(seed=43)
        equal(initial, retained)
        adapter.set_state(initial)
        equal(initial, adapter.get_state())

        # Lane 1 crosses a reset boundary and consumes a cached Gaussian; the
        # other lanes continue from different lifecycle positions.
        boundary = initial.clone()
        boundary["rng", "has_gauss"][1] = True
        boundary["rng", "cached_gaussian"][1] = 0.125
        boundary["lifecycle", "reset_next_step"][1] = True
        boundary["lifecycle", "step_count"][2] = 3
        adapter.set_state(boundary)
        equal(boundary, adapter.get_state())
        expected = adapter.step(actions)
        adapter.set_state(boundary)
        actual = adapter.step(actions)
        equal(expected, actual)
        held = actual.clone()
        # Caller writes must not mutate live physics or a sibling snapshot.
        live = adapter.get_state()
        for value in live.values(True, True):
            value.fill_(0)
        equal(held["state"], adapter.get_state())
    finally:
        adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.get_state()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.step(actions, capture_state=False)
