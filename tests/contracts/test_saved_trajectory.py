"""Disk-only replay of independently generated native State/task trajectories.

CUDA cases test device delivery and chunk/direct frame consistency. They do not
claim source pixel parity. The old frozen corpora remain independent controls.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.testing.corpus import load_trajectory
from tests.helpers.reference.profiles import PROFILES
from tests.helpers.reference.saved import assert_frozen_equal, assert_frozen_state, native_corpus
from tests.helpers.reference.semantics import compare_state

ROOT = Path(__file__).resolve().parents[1] / "fixtures/semantics"
CASES = PROFILES["native"]
pytestmark = pytest.mark.regression


def corpus(case):
    return load_trajectory(ROOT / "shared_native" / case.domain, registration=case.registration)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_old_and_new_source_corpora_agree(case):
    folder = "native_classic" if case.domain in {"cartpole", "pendulum"} else "native_classic_remaining"
    manifest, old = native_corpus(ROOT / folder, case.domain)
    new = corpus(case)
    assert new.metadata["source"]["commit"] == manifest["reference_commit"]
    assert new.metadata["source"]["versions"]["gymnasium"] == manifest["versions"]["gymnasium"]
    assert new.arrays.keys() == old.keys()
    for key, value in old.items():
        assert value.dtype == new.arrays[key].dtype, key
        assert value.shape == new.arrays[key].shape, key
        assert value.tobytes() == new.arrays[key].tobytes(), key


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runs on Oscar")],
        ),
    ],
)
def test_disk_restore_continuation_and_capture(case, device):
    saved = corpus(case)
    with se.make(case.name, num_envs=saved.metadata["batch"], device=device) as env:
        # Restoring real reached states into a different history must reproduce
        # source continuations, not merely a candidate's own get/set round trip.
        for start in (0, 7, 24):
            env.reset(seed=100 + start)
            state = saved.state(start, device=device)
            env.set_state(state)
            compare_state(env.get_state(), saved.state(start))
            for value in state.values(True, True):
                value.zero_()
            compare_state(env.get_state(), saved.state(start))
            for t in range(start, start + 8):
                result = env.step(saved.arrays["actions"][t].copy(), states=None, frames=None)
                assert_frozen_state(env.get_state(), saved.state(t + 1))
                for key in ("reward", "terminated", "truncated", "observation", "info_state"):
                    expected = torch.tensor(saved.arrays[key][t])
                    assert_frozen_equal(result[key], expected)

        # Same loaded state/action path on CPU and CUDA; this assertion compares
        # chunk capture and direct rendering, not our pixels to the upstream renderer.
        for horizon in (1, 8):
            env.set_state(saved.state(7, device=device))
            actions = saved.arrays["actions"][7 : 7 + horizon].transpose(1, 0, 2).copy()
            chunk = env.step(actions[:, 0] if horizon == 1 else actions, states=slice(None), frames=slice(None))
            held = chunk.clone()
            frames = []
            batch = saved.metadata["batch"]
            active = np.ones(batch, dtype=bool)
            endpoint = np.full(batch, 7)
            for t in range(horizon):
                index = 7 + t
                endpoint[active] = index + 1
                expected_state = torch.stack([saved.state(int(k))[lane] for lane, k in enumerate(endpoint)])
                assert_frozen_state(chunk["states"][:, t], expected_state)
                torch.testing.assert_close(chunk["valid"][:, t].cpu(), torch.tensor(active))
                for key in ("reward", "terminated", "truncated", "observation", "info_state"):
                    expected = saved.arrays[key][index].copy()
                    expected[~active] = 0
                    actual = chunk[key] if horizon == 1 else chunk[key][:, t]
                    assert_frozen_equal(actual, expected)
                frames.append(env.render(expected_state.to(device)))
                active &= ~(saved.arrays["terminated"][index] | saved.arrays["truncated"][index])
            assert chunk["frames"].device.type == device
            assert torch.equal(chunk["frames"], torch.stack(frames, dim=1))
            assert_frozen_state(env.get_state(), expected_state)
            env.reset(seed=99)
            compare_state(chunk, held)
