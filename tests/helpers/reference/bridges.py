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
from tensordict import TensorDict

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
    # Package initializers import optional expert policies; environment modules
    # themselves are loaded unchanged from the verified checkout.
    packages = {
        "rocket": ("rocket_landing",),
        "ogbench": ("ogbench",),
        "pusht": ("pusht",),
        "dmc": ("dmcontrol",),
        "navigation": ("two_room",),
    }
    for sub in packages.get(family, ()):
        name = "stable_worldmodel.envs." + sub
        if name not in sys.modules:
            namespace = types.ModuleType(name)
            namespace.__path__ = [str(package / "envs" / sub)]
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


class DMCBridge(NativeBridge):
    def __init__(self, case, batch):
        from tests.helpers.reference.dmc_semantics import DMCReference

        self.source = DMCReference(case.domain, batch)


class NavigationBridge(NativeBridge):
    def __init__(self, case, batch):
        from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference

        self.source = NativeNavigationReference(case.domain, batch)

    def step(self, actions, active=None):
        return self.source.step(actions, active=active)


class OGBBridge:
    def __init__(self, case, batch):
        self.domain = case.domain
        self.source = None
        self.envs = []
        if case.domain == "Maze":
            from tests.helpers.reference.ogbench_maze_semantics import OGBMazeReference

            self.source = OGBMazeReference(batch, width=64, height=64)
        else:
            from tests.helpers.reference.ogbench_manipulation_semantics import load_oracle

            self.oracle = load_oracle()
            try:
                for lane in range(batch):
                    self.envs.append(getattr(self.oracle, "Owned" + case.domain)(entropy_seed=lane))
            except BaseException:
                self.close()
                raise

    def reset(self, seed):
        if self.source:
            self.source.reset(seed)
        else:
            for lane, source in enumerate(self.envs):
                source.reset(seed=seed + lane)

    def compare(self, candidate):
        if self.source:
            compare_state(candidate.get_state(), self.source.get_state())
        else:
            from tests.helpers.reference.ogbench_manipulation_semantics import assert_oracle

            for source, actual in zip(self.envs, candidate._simulator.envs, strict=True):
                assert_oracle(actual, source, self.oracle)

    def step(self, actions):
        if self.source:
            return self.source.step(actions)
        rows = [env.step(action.copy()) for env, action in zip(self.envs, actions, strict=True)]
        return {**task_columns(rows), "success": torch.tensor([bool(row[4]["success"]) for row in rows])}

    def close(self):
        if self.source:
            self.source.close()
        for env in self.envs:
            env.close()


class PushTBridge:
    def __init__(self, case, batch):
        module = "env_discrete" if case.domain == "discrete" else "env"
        cls = getattr(importlib.import_module("stable_worldmodel.envs.pusht." + module), case.name)
        self.envs = [cls(resolution=64) for _ in range(batch)]

    def reset(self, seed):
        for lane, source in enumerate(self.envs):
            source.reset(seed=seed + lane)

    def compare(self, candidate):
        from scripts.research_replay.pusht_replay_gate import diagnostics, equal

        for source, replay in zip(self.envs, candidate._simulator.envs, strict=True):
            equal(diagnostics(source), diagnostics(replay.env))

    def step(self, actions):
        rows = [source.step(action.copy()) for source, action in zip(self.envs, actions, strict=True)]
        return {**task_columns(rows), "success": torch.tensor([bool(row[2]) for row in rows])}

    def close(self):
        for source in self.envs:
            source.close()


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


def assert_tasks(actual, expected, device):
    assert all(key in actual and key in expected for key in ("reward", "terminated", "truncated"))
    keys = [key for key in expected.keys() if key != "state"]
    batch = [len(expected["reward"])]
    actual = TensorDict({key: actual[key] for key in keys}, batch)
    assert all(value.device.type == device.type for value in actual.values(True, True))
    compare_state(actual, TensorDict({key: expected[key] for key in keys}, batch))


BRIDGES = {
    "native": NativeBridge,
    "fetch": FetchBridge,
    "rocket": RocketBridge,
    "dmc": DMCBridge,
    "navigation": NavigationBridge,
    "ogbench": OGBBridge,
    "pusht": PushTBridge,
}
