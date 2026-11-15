"""The shared simulation backend switch: explicit names, the knob, the calibration and its cache."""

import functools

import pytest
import torch
from tensordict import TensorDict

from stable_envs import tuning
from stable_envs._core.tensordicts import to_device
from stable_envs._execution import backends
from stable_envs._execution.host import HostAdapter

BACKENDS = {"cpu": backends.Rungs("small", "small"), "cuda": backends.Rungs("small", "large")}
CHOICES = ("small", "large")


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("STABLE_ENVS_LARGE_BACKEND_MIN_ENVS", raising=False)
    tuning.reset()
    tuning._clear_caches()
    yield
    tuning._clear_caches()


def _measure(costs, calls):
    """A measure whose step time is ``costs[backend](rows)`` seconds, recorded in ``calls``."""

    def measure(backend, rows):
        calls.append((backend, rows))
        return (lambda: costs[backend](rows)), (lambda: None)

    return measure


def _select(num_envs, device="cuda", requested="auto", measure=None):
    return backends.select_backend("test-sim", num_envs, device, requested, BACKENDS, choices=CHOICES, measure=measure)


def test_explicit_names_and_cpu_need_no_measurement():
    assert _select(8, requested="small") == "small"
    assert _select(8, requested="large") == "large"
    assert _select(8, device="cpu") == "small"
    with pytest.raises(ValueError, match="simulation_backend"):
        _select(8, requested="default")


@pytest.mark.parametrize("switch,batch,expected", [(1, 1, "large"), (8, 7, "small"), (8, 8, "large")])
def test_the_knob_sets_the_switch_point(switch, batch, expected):
    with tuning.override(large_backend_min_envs=switch):
        assert _select(batch, requested=None) == expected


def test_without_a_value_auto_uses_the_shipped_switch_point():
    assert _select(3) == "large"
    shipped = {"cpu": backends.Rungs("small", "small"), "cuda": backends.Rungs("small", "large", 4)}
    pick = lambda n: backends.select_backend("test-sim", n, "cuda", "auto", shipped, choices=CHOICES)  # noqa: E731
    assert pick(3) == "small" and pick(4) == "large"
    with tuning.override(large_backend_min_envs=2):
        assert pick(3) == "large"


def test_selection_times_nothing_and_tune_calibrates_and_caches(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: step())
    monkeypatch.setattr(tuning, "_dev", lambda device: str(device))
    monkeypatch.setattr(tuning, "stamp", functools.cache(lambda device: {"device": device}))
    monkeypatch.setattr(backends, "_REGISTERED", {})
    calls = []
    costs = {"small": lambda rows: 1e-4 * rows, "large": lambda rows: 4e-4}
    measure = _measure(costs, calls)
    assert _select(2, measure=measure) == "large"
    assert calls == []
    assert backends.calibrate(2, "cuda") == {"test-sim": "small"}
    assert calls == [("large", 2), ("small", 1), ("small", 2)]
    calls.clear()
    assert _select(3, measure=measure) == "small"
    # From a large batch, the search stops once the small-batch backend is slower.
    assert backends.calibrate(64, "cuda") == {"test-sim": "large"}
    assert calls == [("large", 64), ("small", 1), ("small", 2), ("small", 4), ("small", 8)]
    # A larger range follows a smaller range that chose the large-batch backend.
    assert _select(1024, measure=measure) == "large"
    data, current = tuning.read_cache("cuda")
    assert current
    assert data["entries"]["test-sim|0|2-3"] == {backends.KNOB: 4}
    assert data["entries"]["test-sim|0|64-127"] == {backends.KNOB: 64}


def test_a_backend_that_fails_to_build_loses(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: 1e-4)
    monkeypatch.setattr(tuning, "_dev", lambda device: str(device))
    monkeypatch.setattr(tuning, "stamp", functools.cache(lambda device: {"device": device}))
    monkeypatch.setattr(backends, "_REGISTERED", {})

    def measure(backend, rows):
        if backend == "large":
            raise RuntimeError("no GPU backend")
        return (lambda: None), (lambda: None)

    _select(16, measure=measure)
    assert backends.calibrate(16, "cuda", write=False) == {"test-sim": "small"}


def test_to_device_keeps_cpu_tensordicts_and_their_values():
    td = TensorDict({"a": torch.ones(2, 3), "b": {"c": torch.tensor([True, False])}}, [2])
    assert to_device(td, "cpu") is not None
    assert torch.equal(to_device(td, "cpu")["b", "c"], td["b", "c"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_to_device_moves_every_leaf_in_one_copy():
    td = TensorDict(
        {
            "a": torch.arange(6, dtype=torch.float64).reshape(2, 3),
            "b": {"c": torch.tensor([True, False]), "d": torch.tensor([[1], [2]], dtype=torch.int32)},
            "e": torch.tensor([7, 8], dtype=torch.uint8),
        },
        [2],
    )
    moved = to_device(td, "cuda")
    assert moved.device.type == "cuda"
    for key in td.keys(True, True):
        assert moved[key].dtype == td[key].dtype and moved[key].is_cuda
        assert torch.equal(moved[key].cpu(), td[key])
    td["a"].zero_()
    assert moved["a"].sum().item() == 15


class _Adapter:
    num_envs = 2
    template = object()

    def step(self, actions):
        assert actions.device.type == "cpu"
        return TensorDict({"reward": actions.sum(1), "state": {"x": actions.clone()}}, [2])

    def render_fields(self):
        return {"a": torch.ones(2), "b": torch.zeros(2, 3)}


def test_host_adapter_moves_arguments_to_the_host_and_results_to_the_device():
    adapter = HostAdapter(_Adapter(), "cpu")
    out = adapter.step(torch.ones(2, 2))
    assert out["reward"].tolist() == [2.0, 2.0]
    assert set(adapter.render_fields()) == {"a", "b"}
    assert adapter.template is _Adapter.template
