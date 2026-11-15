"""The shared backend and renderer switch: explicit names, the knobs, the calibration and its cache."""

import functools

import pytest
import torch
from tensordict import TensorDict

from stable_envs import tuning
from stable_envs._core.tensordicts import to_device
from stable_envs._execution import backends, host_render
from stable_envs._execution.host import HostAdapter

BACKENDS = {"cpu": backends.Rungs("small", "small"), "cuda": backends.Rungs("small", "large")}
CHOICES = ("small", "large")


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("STABLE_ENVS_LARGE_BACKEND_MIN_ENVS", raising=False)
    monkeypatch.delenv("STABLE_ENVS_LARGE_RENDERER_MIN_ROWS", raising=False)
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
    assert calls == []
    # From a large batch, the search stops once the small-batch backend is slower.
    assert backends.calibrate(64, "cuda") == {"test-sim": "large"}
    assert calls == [("large", 64), ("small", 1), ("small", 2), ("small", 4), ("small", 8)]
    calls.clear()
    # A larger range follows a smaller range that chose the large-batch backend.
    assert _select(1024, measure=measure) == "large"
    assert calls == []
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


def test_calibrate_times_only_the_named_paths(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: 1e-4)
    monkeypatch.setattr(backends, "_REGISTERED", {})
    calls = []
    measure = _measure({"small": lambda rows: None, "large": lambda rows: None}, calls)
    for path in ("test-sim/a", "test-sim/b"):
        backends.select_backend(path, 4, "cuda", "auto", BACKENDS, choices=CHOICES, measure=measure)
    assert set(backends.calibrate(4, "cuda", paths=("test-sim/b",), write=False)) == {"test-sim/b"}
    assert set(backends.calibrate(4, "cuda", prefixes=("test-sim",), write=False)) == {"test-sim/a", "test-sim/b"}


RENDERERS = {"cpu": backends.Rungs("cpu", "cpu"), "cuda": backends.Rungs("cpu", "cuda", 16)}


def test_renderers_switch_at_their_own_knob():
    pick = lambda n: backends.select_renderer("test-render/Env", n, "cuda", RENDERERS)  # noqa: E731
    assert pick(15) == "cpu" and pick(16) == "cuda"
    assert backends.select_renderer("test-render/Env", 1, "cpu", RENDERERS) == "cpu"
    with tuning.override(large_backend_min_envs=1):
        assert pick(15) == "cpu"
    with tuning.override(large_renderer_min_rows=4):
        assert pick(3) == "cpu" and pick(4) == "cuda"


def test_tune_calibrates_renderers_under_the_render_knob(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: step())
    monkeypatch.setattr(tuning, "_dev", lambda device: str(device))
    monkeypatch.setattr(tuning, "stamp", functools.cache(lambda device: {"device": device}))
    monkeypatch.setattr(backends, "_REGISTERED", {})
    costs = {"cpu": lambda rows: 1e-4 * rows, "cuda": lambda rows: 4e-4}
    measure = _measure(costs, [])
    _select(2, measure=_measure({"small": lambda rows: 0.0, "large": lambda rows: 1.0}, []))
    backends.select_renderer("test-render/Env", 2, "cuda", RENDERERS, measure=measure)
    assert backends.calibrate(2, "cuda", knob=backends.RENDER_KNOB) == {"test-render/Env": "cpu"}
    assert backends.calibrate(64, "cuda", knob=backends.RENDER_KNOB) == {"test-render/Env": "cuda"}
    data, _ = tuning.read_cache("cuda")
    assert data["entries"]["test-render/Env|0|2-3"] == {backends.RENDER_KNOB: 4}
    assert data["entries"]["test-render/Env|0|64-127"] == {backends.RENDER_KNOB: 64}
    assert backends.select_renderer("test-render/Env", 1024, "cuda", RENDERERS) == "cuda"
    assert _select(2) == "large"


def test_cpu_envs_and_explicit_renderers_do_not_switch():
    class Env:
        device = torch.device("cpu")
        render_rungs = RENDERERS

    assert not host_render.on_host(Env(), "test-render", 1)
    Env.device = torch.device("cuda")
    assert not host_render.on_host(Env(), "test-render", 1, backend="torch")
    Env.render_rungs = None
    assert not host_render.on_host(Env(), "test-render", 1)


def test_upload_fills_the_destination():
    frames = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    out = torch.empty_like(frames)
    assert host_render.upload(frames, torch.device("cpu"), out) is out
    assert torch.equal(out, frames)
    assert torch.equal(host_render.draw_into((2, 2, 2, 3), torch.device("cpu"), lambda buffer: frames), frames)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_to_device_keeps_the_leaves_already_on_the_device():
    on_device = torch.arange(4, device="cuda").reshape(2, 2)
    td = TensorDict._new_unsafe(
        {"a": torch.ones(2, 3), "state": TensorDict._new_unsafe({"x": on_device}, batch_size=[2])},
        batch_size=torch.Size([2]),
        device=torch.device("cpu"),
    )
    moved = to_device(td, "cuda")
    assert moved["state", "x"] is on_device
    assert moved["a"].is_cuda and torch.equal(moved["a"].cpu(), td["a"])
    only = TensorDict._new_unsafe({"x": on_device}, batch_size=torch.Size([2]), device=torch.device("cpu"))
    assert to_device(only, "cuda")["x"] is on_device


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
