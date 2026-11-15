"""Frozen real-reference corpus for the classic-control defaults, replayed without the source package."""

from pathlib import Path

import pytest
import torch

import stable_envs as se
from tests.helpers.reference.saved import assert_frozen_equal, native_corpus


@pytest.mark.parametrize(
    "domain,name",
    [
        ("cartpole", "CartPole"),
        ("pendulum", "Pendulum"),
        ("acrobot", "Acrobot"),
        ("mountain_car", "MountainCar"),
        ("mountain_car_continuous", "MountainCarContinuous"),
    ],
)
def test_frozen_source_state_and_tasks(domain, name):
    family = "native_classic" if domain in {"cartpole", "pendulum"} else "native_classic_remaining"
    root = Path(__file__).resolve().parents[1] / "fixtures/semantics" / family
    manifest, arrays = native_corpus(root, domain)
    with se.make(name, num_envs=manifest["batch"], device="cpu") as env:
        states = [env.reset(seed=manifest["seed"])]
        outputs = []
        for action in arrays["actions"]:
            output = env.step(action, frames=None, states=None)
            outputs.append(output)
            states.append(output["state"])
        keys = {"state/" + (key if isinstance(key, str) else "/".join(key)) for key in states[0].keys(True, True)}
        assert keys == {key for key in arrays if key.startswith("state/")}
        for key in states[0].keys(True, True):
            name = key if isinstance(key, str) else "/".join(key)
            actual = torch.stack([state[key] for state in states]).cpu().numpy()
            assert actual.dtype == arrays[f"state/{name}"].dtype
            assert_frozen_equal(actual, arrays[f"state/{name}"])
        for key in ("reward", "terminated", "truncated", "observation", "info_state"):
            actual = torch.stack([output[key] for output in outputs]).cpu().numpy()
            assert actual.dtype == arrays[key].dtype
            assert_frozen_equal(actual, arrays[key])
