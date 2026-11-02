"""An explicit Warp choice does not change native or existing Reacher MJX routing."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation import execution
from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES


@pytest.fixture
def routes(monkeypatch):
    def route(name):
        return lambda *args, **kwargs: (name, args, kwargs)

    monkeypatch.setattr(execution, "make_local", route("native"))
    monkeypatch.setattr(execution, "process_count", lambda *args, **kwargs: 0)
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.mjwarp",
        SimpleNamespace(DMCWarpAdapter=route("mjwarp")),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.reacher_mjx",
        SimpleNamespace(ReacherMJXAdapter=route("mjx")),
    )


@pytest.mark.parametrize("key", sorted(set(FACTORIES) - {"reacher"}))
def test_supported_registered_tasks_can_explicitly_select_warp(routes, key):
    kind, args, kwargs = execution.make_dmc_adapter(key, 3, "cuda:2", 19, simulation_backend="mjwarp")
    assert (kind, args, kwargs) == ("mjwarp", (key, 3, "cuda:2", 19), {})


@pytest.mark.parametrize("key", sorted(set(FACTORIES) - {"reacher"}))
def test_explicit_warp_rejects_cpu_instead_of_falling_back(routes, key):
    with pytest.raises(ValueError, match="cuda"):
        execution.make_dmc_adapter(key, 3, "cpu", 19, simulation_backend="mjwarp")


@pytest.mark.parametrize("key", sorted(set(FACTORIES) - {"reacher"}))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_other_tasks_keep_native_automatic_routing(routes, key, device):
    kind, args, kwargs = execution.make_dmc_adapter(key, 4096, device, 19)
    assert (kind, args, kwargs) == ("native", (key, 4096, device, 19), {})


@pytest.mark.parametrize(
    "batch,device,backend,threshold,expected",
    [
        (31, "cuda", "auto", 32, "native"),
        (32, "cuda", "auto", 32, "mjx"),
        (4096, "cpu", "auto", 32, "native"),
        (4096, "cuda", "native", 32, "native"),
        (1, "cuda", "mjx", 32, "mjx"),
        (64, "cuda", "auto", 128, "native"),
    ],
)
def test_reacher_backend_and_threshold_remain_explicit(routes, batch, device, backend, threshold, expected):
    kind, args, kwargs = execution.make_dmc_adapter(
        "reacher", batch, device, 19, simulation_backend=backend, mjx_min_envs=threshold
    )
    assert kind == expected
    assert args == ((batch, device, 19) if expected == "mjx" else ("reacher", batch, device, 19))
    assert kwargs == {}


@pytest.mark.parametrize("backend", ["unknown", "mjx"])
def test_unsupported_non_reacher_backend_is_not_silently_ignored(routes, backend):
    with pytest.raises(ValueError):
        execution.make_dmc_adapter("walker", 3, "cuda", 19, simulation_backend=backend)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reacher_warp_integration_is_excluded(routes, device):
    with pytest.raises(ValueError, match="DMCReacher does not support"):
        execution.make_dmc_adapter("reacher", 3, device, 19, simulation_backend="mjwarp")
    with pytest.raises(ValueError, match="simulation_backend"):
        execution.reacher_backend(3, device, "mjwarp")
