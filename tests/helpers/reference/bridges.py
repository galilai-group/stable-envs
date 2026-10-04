"""Family-specific witnesses composed by the small shared contracts.

The Rocket bridge compares independently captured native-engine fields, since
its public replay recipe is not a source snapshot. Fetch and native bridges
normalize source-owned fields to full TensorDicts. No oracle calls a candidate
serializer to manufacture expected values.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import torch

from tests.helpers.reference.semantics import compare_state


def load_source(root, family):
    """Load source environment modules without the unrelated learning stack.

    Only the package namespace is supplied; all env, utility and space code is
    loaded unchanged from the verified checkout. Run profiles in fresh processes.
    """
    package = Path(root).resolve() / "stable_worldmodel"
    module = sys.modules.get("stable_worldmodel")
    if module is None:
        module = types.ModuleType("stable_worldmodel")
        module.__path__ = [str(package)]
        sys.modules[module.__name__] = module
    if list(module.__path__) != [str(package)]:
        raise ValueError("Another stable-worldmodel checkout is already imported")
    for name in ("utils", "spaces", "envs"):
        setattr(module, name, importlib.import_module("stable_worldmodel." + name))
    if family == "rocket":
        # The Rocket package __init__ imports its optional learned/expert policy.
        # The registered environment entry point is the unchanged file below it.
        name = "stable_worldmodel.envs.rocket_landing"
        if name not in sys.modules:
            namespace = types.ModuleType(name)
            namespace.__path__ = [str(package / "envs/rocket_landing")]
            sys.modules[name] = namespace


class NativeBridge:
    def __init__(self, case, batch):
        from tests.helpers.reference.native_classic_remaining_semantics import NativeClassicRemainingReference
        from tests.helpers.reference.native_classic_semantics import NativeClassicReference

        reference = (
            NativeClassicReference if case.domain in {"cartpole", "pendulum"} else NativeClassicRemainingReference
        )
        self.source = reference(case.domain, batch)

    def reset(self, seed):
        self.source.reset(seed=seed)

    def compare(self, candidate):
        compare_state(candidate.get_state(), self.source.get_state())

    def step(self, actions):
        return self.source.step(actions)

    def close(self):
        self.source.close()


class FetchBridge:
    def __init__(self, case, batch):
        import gymnasium as gym

        self.envs = []
        try:
            for _ in range(batch):
                self.envs.append(gym.make(case.registration))
        except BaseException:
            self.close()
            raise

    def reset(self, seed):
        for lane, source in enumerate(self.envs):
            source.reset(seed=seed + lane, options={"variation": []})

    def compare(self, candidate):
        from scripts.benchmarks.online_stepping.fetch_source_pipeline import source_state
        from scripts.benchmarks.online_stepping.fetch_source_render_lifecycle import snapshot

        compare_state(candidate.get_state(), source_state([snapshot(source) for source in self.envs]))

    def step(self, actions):
        outputs = [source.step(action.copy()) for source, action in zip(self.envs, actions, strict=True)]
        return {
            **task_columns(outputs),
            "success": torch.as_tensor(np.asarray([row[4]["is_success"] for row in outputs])),
        }

    def close(self):
        for source in self.envs:
            source.close()


class RocketBridge:
    def __init__(self, case, batch):
        from scripts.recovery.rocket_feasibility import RocketLandingEnv

        self.envs = []
        try:
            for _ in range(batch):
                self.envs.append(RocketLandingEnv(render_mode="rgb_array", render_resolution=(64, 64)))
        except BaseException:
            self.close()
            raise

    def reset(self, seed):
        for lane, source in enumerate(self.envs):
            source.reset(seed=seed + lane)

    def compare(self, candidate):
        from scripts.recovery.rocket_feasibility import capture, differences

        for source, replay in zip(self.envs, candidate._simulator.envs, strict=True):
            delta = differences(capture(source), capture(replay.env))
            assert not delta, delta

    def step(self, actions):
        return task_columns([source.step(action.copy()) for source, action in zip(self.envs, actions, strict=True)])

    def close(self):
        for source in self.envs:
            source.close()


def task_columns(outputs):
    return {
        key: torch.as_tensor(np.asarray([row[index] for row in outputs]))
        for key, index in (("reward", 1), ("terminated", 2), ("truncated", 3))
    }


def assert_tasks(actual, expected):
    for key in ("reward", "terminated", "truncated"):
        assert key in actual and key in expected, f"Missing {key}"
        torch.testing.assert_close(actual[key].cpu(), expected[key].cpu(), rtol=0, atol=0)
    if "success" in expected:
        torch.testing.assert_close(actual["success"].cpu(), expected["success"].cpu(), rtol=0, atol=0)


BRIDGES = {"native": NativeBridge, "fetch": FetchBridge, "rocket": RocketBridge}
