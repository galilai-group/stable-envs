"""Check actual composed call boundaries, especially StepResult and projection ownership."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BENCH = Path(__file__).resolve().parents[2] / "scripts/benchmarks"


@pytest.mark.parametrize("explicit_projection", [False, True])
def test_composed_rollouts_count_each_boundary_once(tmp_path, monkeypatch, explicit_projection):
    monkeypatch.syspath_prepend(str(BENCH))
    spec = importlib.util.spec_from_file_location("rendered_test", BENCH / "rendered_state_ops.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    traces = []

    class Fake:
        def __init__(self):
            self.value = 0
            self.action_spec = SimpleNamespace(sample=lambda *a, **k: torch.zeros((1, 1)))
            self.closed = False
            if explicit_projection:
                self.to_render_state = self.project

        def reset(self, seed):
            self.value = 0

        def step(self, action):
            calls.append("step")
            self.value += 1
            return {"state": self.value, "reward": 123}

        def get_state(self):
            calls.append("get")
            return self.value

        def set_state(self, value):
            calls.append("set")
            self.value = value

        def project(self, state):
            assert isinstance(state, int), "StepResult is not State"
            calls.append("projection")
            return ("render_state", state)

        def render(self, state):
            assert isinstance(state, tuple if explicit_projection else int)
            calls.append("render")
            return 0

        def close(self):
            self.closed = True

    env = Fake()
    import stable_envs

    monkeypatch.setattr(stable_envs, "make", lambda *a, **k: env)
    monkeypatch.setattr(module, "profile", lambda: {})
    monkeypatch.setattr(module, "synchronize", lambda x=None: x)

    def measured(fn, prepare=None, **kwargs):
        if prepare:
            prepare()
        calls.clear()
        fn()
        traces.append(calls.copy())
        return {"median_ns": 100.0, "status": "MEASURED"}

    monkeypatch.setattr(module, "measure", measured)
    args = SimpleNamespace(
        registration="swm/CartPoleControl-v1",
        output=tmp_path / "result",
        device="cpu",
        horizon=16,
        warmup=0,
        iterations=1,
    )
    module.run(args)
    workloads = traces[-16:]
    for cadence, frames in enumerate([16, 4, 1, 0]):
        for mode in range(4):
            trace = workloads[cadence * 4 + mode]
            assert trace.count("step") == (24 if mode == 3 else 16)
            assert trace.count("get") == ([0, 16, 0, 1][mode])
            assert trace.count("set") == int(mode in [2, 3])
            assert trace.count("render") == frames
            assert trace.count("projection") == (frames if explicit_projection else 0)
    assert env.closed
