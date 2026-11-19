"""Backend selection: the default of each device, explicit names, and the renderer switch with its calibration."""

import functools

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs import tuning
from stable_envs._core.tensordicts import pack_to_device, stack_leaves, to_device
from stable_envs._execution import backends, host_render
from stable_envs._execution.host import HostAdapter

DEFAULTS = {"cpu": "host", "cuda": "device"}
CHOICES = ("host", "device")


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
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


def test_auto_selects_the_default_of_the_device_type():
    for requested in ("auto", None):
        assert backends.select_backend("cpu", requested, DEFAULTS, choices=CHOICES) == "host"
        assert backends.select_backend("cuda:0", requested, DEFAULTS, choices=CHOICES) == "device"
        assert backends.select_backend("meta", requested, DEFAULTS, choices=CHOICES) == "host"
        assert backends.select_backend("cuda", requested, {"cpu": "host"}, choices=CHOICES) == "host"


def test_explicit_names_select_that_backend_on_every_device():
    for device in ("cpu", "cuda"):
        assert backends.select_backend(device, "host", DEFAULTS, choices=CHOICES) == "host"
        assert backends.select_backend(device, "device", DEFAULTS, choices=CHOICES) == "device"
    with pytest.raises(ValueError, match="simulation_backend"):
        backends.select_backend("cpu", "default", DEFAULTS, choices=CHOICES)


def test_the_renderer_setting_does_not_move_the_simulation_backend():
    with tuning.override(large_renderer_min_rows=1):
        assert backends.select_backend("cuda", "auto", {"cpu": "host"}, choices=CHOICES) == "host"


RENDERERS = {"cpu": backends.BackendPair("cpu", "cpu"), "cuda": backends.BackendPair("cpu", "cuda", 16)}


def test_renderers_switch_at_their_own_setting():
    pick = lambda n: backends.select_renderer("test-render/Env", n, "cuda", RENDERERS)  # noqa: E731
    assert pick(15) == "cpu" and pick(16) == "cuda"
    assert backends.select_renderer("test-render/Env", 1, "cpu", RENDERERS) == "cpu"
    with tuning.override(large_renderer_min_rows=4):
        assert pick(3) == "cpu" and pick(4) == "cuda"


def test_tune_calibrates_renderers_under_the_render_setting(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: step())
    monkeypatch.setattr(tuning, "_dev", lambda device: str(device))
    monkeypatch.setattr(tuning, "stamp", functools.cache(lambda device: {"device": device}))
    monkeypatch.setattr(backends, "_REGISTERED", {})
    costs = {"cpu": lambda rows: 1e-4 * rows, "cuda": lambda rows: 4e-4}
    calls = []
    measure = _measure(costs, calls)
    assert backends.select_renderer("test-render/Env", 2, "cuda", RENDERERS, measure=measure) == "cpu"
    assert calls == []
    assert backends.calibrate(2, "cuda") == {"test-render/Env": "cpu"}
    assert backends.calibrate(64, "cuda") == {"test-render/Env": "cuda"}
    data, _ = tuning.read_cache("cuda")
    assert data["entries"]["test-render/Env|0|2-3"] == {backends.RENDERER_SETTING: 4}
    assert data["entries"]["test-render/Env|0|64-127"] == {backends.RENDERER_SETTING: 64}
    assert backends.select_renderer("test-render/Env", 1024, "cuda", RENDERERS) == "cuda"


def test_a_renderer_that_fails_to_build_loses(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: 1e-4)
    monkeypatch.setattr(backends, "_REGISTERED", {})

    def measure(renderer, rows):
        if renderer == "cuda":
            raise RuntimeError("no GPU renderer")
        return (lambda: None), (lambda: None)

    backends.select_renderer("test-render/Env", 16, "cuda", RENDERERS, measure=measure)
    assert backends.calibrate(16, "cuda", write=False) == {"test-render/Env": "cpu"}


def test_calibrate_times_only_the_named_paths(monkeypatch):
    monkeypatch.setattr(backends, "_seconds_per_step", lambda step, device: 1e-4)
    monkeypatch.setattr(backends, "_REGISTERED", {})
    measure = _measure({"cpu": lambda rows: None, "cuda": lambda rows: None}, [])
    for path in ("test-render/A", "test-render/B"):
        backends.select_renderer(path, 4, "cuda", RENDERERS, measure=measure)
    assert set(backends.calibrate(4, "cuda", paths=("test-render/B",), write=False)) == {"test-render/B"}
    assert set(backends.calibrate(4, "cuda", prefixes=("test-render",), write=False)) == {
        "test-render/A",
        "test-render/B",
    }


def test_cpu_envs_and_explicit_renderers_do_not_switch():
    class Env:
        device = torch.device("cpu")
        render_backends = RENDERERS

    assert not host_render.on_host(Env(), "test-render", 1)
    Env.device = torch.device("cuda")
    assert not host_render.on_host(Env(), "test-render", 1, backend="torch")
    Env.render_backends = None
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_to_device_returns_a_tensordict_already_on_the_device():
    leaves = {"a": torch.ones(2, 3, device="cuda"), "b": {"c": torch.zeros(2, device="cuda")}}
    td = TensorDict(leaves, [2], device="cuda")
    assert to_device(td, "cuda") is td


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_to_device_keeps_the_batch_size_of_a_node_with_a_time_axis():
    td = TensorDict({"a": torch.ones(2, 3), "states": {"x": torch.arange(4.0).reshape(2, 1, 2)}}, [2])
    td["states"].batch_size = [2, 1]
    moved = to_device(td, "cuda")
    assert moved["states"].batch_size == torch.Size([2, 1])
    assert torch.equal(moved["states", "x"].cpu(), td["states", "x"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pack_to_device_owns_its_memory_and_later_packs_leave_it_alone():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    flags = np.array([True, False])
    nested = np.arange(4, dtype=np.float64).reshape(2, 1, 2)
    leaves = [(("a",), a), (("info", "flag"), flags), (("states", "x"), nested), (("b",), a[:, :2])]
    first = pack_to_device(leaves, (2,), "cuda", ((("states",), (2, 1)),))
    assert first["states"].batch_size == torch.Size([2, 1]) and first["info"].batch_size == torch.Size([2])
    a += 100
    second = pack_to_device(leaves, (2,), "cuda", ((("states",), (2, 1)),))
    torch.cuda.synchronize()
    assert torch.equal(first["a"].cpu(), torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.equal(second["a"].cpu(), torch.from_numpy(a))
    assert torch.equal(first["b"].cpu(), first["a"][:, :2].cpu())
    assert first["info", "flag"].dtype == torch.bool and first["states", "x"].dtype == torch.float64
    assert first["a"].untyped_storage().data_ptr() != second["a"].untyped_storage().data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pack_to_device_reuses_only_a_block_that_no_tensor_uses():
    leaves = [(("x",), np.zeros((2, 3), np.float32)), (("y",), np.zeros(2, np.float64))]
    first = pack_to_device(leaves, (2,), "cuda")
    pointer = first["x"].data_ptr()
    view = first["y"][:1]
    del first
    leaves[0][1][:] = 1
    held = pack_to_device(leaves, (2,), "cuda")
    assert held["x"].data_ptr() != pointer
    del view
    leaves[0][1][:] = 2
    again = pack_to_device(leaves, (2,), "cuda")
    assert again["x"].data_ptr() == pointer
    assert torch.equal(again["x"].cpu(), torch.full((2, 3), 2.0)) and torch.equal(held["x"].cpu(), torch.ones(2, 3))
    leaves[0][1][:] = 3
    while_held = pack_to_device(leaves, (2,), "cuda")
    assert while_held["x"].data_ptr() not in (pointer, held["x"].data_ptr())
    assert torch.equal(again["x"].cpu(), torch.full((2, 3), 2.0))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_stacking_one_packed_state_copies_it_once_into_new_storage(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    flat = torch.arange(10.0, device=device)
    state = TensorDict(
        {"p": flat[0:2].view(2, 1), "q": flat[2:4].view(2, 1), "r": {"s": flat[4:10].view(2, 3)}}, [2], device=device
    )
    stacked = stack_leaves([state])
    assert stacked.batch_size == torch.Size([2, 1])
    for key in state.keys(True, True):
        assert torch.equal(stacked[key], state[key].unsqueeze(1))
        assert stacked[key].untyped_storage().data_ptr() != flat.untyped_storage().data_ptr()
    flat.zero_()
    assert stacked["r", "s"].sum().item() == 4 + 5 + 6 + 7 + 8 + 9


def test_stacking_one_packed_state_keeps_leaves_at_offsets_that_are_not_a_multiple_of_the_first():
    raw = torch.zeros(40, dtype=torch.uint8)
    raw[16:40].view(torch.float32).copy_(torch.arange(1.0, 7.0))
    raw[3:6] = 1
    state = TensorDict({"flag": raw[3:6].view(torch.bool), "x": raw[16:40].view(torch.float32).view(3, 2)}, [3])
    stacked = stack_leaves([state])
    assert torch.equal(stacked["x"], state["x"].unsqueeze(1))
    assert torch.equal(stacked["flag"], state["flag"].unsqueeze(1))


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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_pack_to_device_copies_the_state_under_states_with_a_time_axis(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    leaves = [
        (("state", "x"), np.arange(6, dtype=np.float32).reshape(2, 3)),
        (("state", "n"), np.array([4, 5], np.int64)),
        (("reward",), np.array([0.5, 1.5])),
        (("state_valid",), np.ones((2, 1), np.bool_)),
    ]
    copies = (("state",), ("states",), ("state_valid",))
    result = pack_to_device(leaves, (2,), device, ((("states",), (2, 1)),), copies)
    assert list(result.keys()) == ["state", "reward", "states", "state_valid"]
    assert result["states"].batch_size == torch.Size([2, 1])
    for key in ("x", "n"):
        state, states = result["state", key], result["states", key]
        assert states.shape == (2, 1, *state.shape[1:])
        assert torch.equal(states[:, 0], state)
        assert states.data_ptr() != state.data_ptr()
    result["state", "x"].zero_()
    assert torch.equal(result["states", "x"][:, 0].cpu(), torch.arange(6.0).reshape(2, 3))
