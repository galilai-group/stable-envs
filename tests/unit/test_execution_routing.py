"""Process routing respects batch size, CPU allocation, platform, process context and each family's exclusions."""

from types import SimpleNamespace

import pytest
import torch

from stable_envs import tuning
from stable_envs._execution import policy


def _process(*args, **kwargs):
    return ("process", args, kwargs)


def _local(*args, **kwargs):
    return ("local", args, kwargs)


def _batch(*args, **kwargs):
    return ("batch", args, kwargs)


def _classic(monkeypatch, target, batch, device):
    from stable_envs.envs.classic_control.simulation import execution, process

    monkeypatch.setattr(process, "AcrobotProcessAdapter", _process)
    return execution.make_classic_adapter(target, batch, device, 17, _local), (target, batch, device, 17), {}, {}


def _fetch(monkeypatch, target, batch, device):
    from stable_envs.envs.fetch.simulation import execution

    monkeypatch.setattr(execution, "FetchAdapter", _local)
    monkeypatch.setattr(execution, "FetchProcessAdapter", _process)
    args = (batch, device, 17, target, "dense")
    return execution.make_fetch_adapter(*args), args, {}, {"threads_per_process": 1}


#: (family, route, target, devices, whether the target may use processes)
TARGETS = [
    ("classic", _classic, "acrobot", ("cpu", "cuda"), True),
    ("classic", _classic, "cartpole", ("cpu", "cuda"), False),
    ("classic", _classic, "mountain_car_continuous", ("cpu", "cuda"), False),
    ("fetch", _fetch, "Push", ("cpu",), True),
]


@pytest.mark.parametrize(
    "route,target,device,eligible",
    [
        pytest.param(route, target, device, eligible, id=f"{target}-{device}", marks=pytest.mark.family(family))
        for family, route, target, devices, eligible in TARGETS
        for device in devices
    ],
)
@pytest.mark.parametrize(
    "platform,batch,cores,daemon,count",
    [
        ("linux", 64, 8, False, 8),
        ("linux", 256, 32, False, 8),
        ("linux", 64, 4, False, 4),
        ("linux", 64, 6, False, 6),
        ("linux", 63, 8, False, 0),
        ("linux", 64, 2, False, 0),
        ("linux", 64, 8, True, 0),
        ("darwin", 64, 8, False, 0),
        ("win32", 64, None, False, 0),
        ("linux", 64, None, False, 0),
    ],
)
def test_routing(monkeypatch, route, target, device, eligible, platform, batch, cores, daemon, count):
    monkeypatch.setattr(policy.sys, "platform", platform)
    # Real affinity IDs need not be contiguous or start at zero.
    cpus = None if cores is None else [2 * i + 1 for i in range(cores)]
    if cpus is None:
        monkeypatch.delattr(policy.os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(policy.os, "sched_getaffinity", lambda _: set(cpus), raising=False)
    monkeypatch.setattr(policy.mp, "current_process", lambda: SimpleNamespace(daemon=daemon))
    monkeypatch.setattr(policy, "cpu_limit", lambda: 1024)
    (kind, args, kwargs), expected_args, local_kwargs, process_kwargs = route(monkeypatch, target, batch, device)
    expected = count if eligible else 0
    assert kind == ("process" if expected else "local")
    assert args == expected_args
    assert kwargs == (dict(processes=expected, cpus=cpus, **process_kwargs) if expected else local_kwargs)


def _fake_adapters(monkeypatch):
    import sys

    from stable_envs.envs.dmcontrol.simulation import execution

    monkeypatch.setattr(execution, "make_batch", _batch)
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.reacher_mjx",
        SimpleNamespace(ReacherMJXAdapter=lambda *args: ("mjx", args, {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.mjwarp",
        SimpleNamespace(DMCWarpAdapter=lambda *args: ("mjwarp", args, {})),
    )
    return execution


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "batch,device,backend,warp_cutoff,expected",
    [
        (1, "cuda", None, None, "mjwarp"),
        (4096, "cuda:1", None, 64, "mjwarp"),
        (1, "cpu", None, None, "native"),
        (4096, "cpu", "auto", 64, "native"),
        (4096, "cuda", "native", 64, "native"),
        (1, "cuda", "mjwarp", 64, "mjwarp"),
        (64, "cuda:1", "mjwarp", 64, "mjwarp"),
        (1, "cuda", "mjx", 64, "mjx"),
        (63, "cuda", "auto", 64, "native"),
        (64, "cuda", "auto", 64, "mjwarp"),
        (2, "cuda", "auto", 2, "mjwarp"),
    ],
)
def test_reacher_routes_without_changing_the_output_device(monkeypatch, batch, device, backend, warp_cutoff, expected):
    execution = _fake_adapters(monkeypatch)
    assert execution.dmc_backend("reacher", batch, device, backend, mjwarp_min_envs=warp_cutoff) == expected
    kind, args, _ = execution.make_dmc_adapter(
        "reacher", batch, device, 17, simulation_backend=backend, mjwarp_min_envs=warp_cutoff
    )
    assert kind == (expected if expected in {"mjx", "mjwarp"} else "batch")
    assert args == ((batch, device, 17) if expected == "mjx" else ("reacher", batch, device, 17))


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "backend,device,warp_cutoff",
    [
        ("unknown", "cuda", None),
        ("mjx", "cpu", None),
        ("mjwarp", "cpu", None),
        ("auto", "cuda", 0),
        ("auto", "cuda", True),
        ("auto", "cuda", 1.5),
    ],
)
def test_dmc_rejects_invalid_simulation_selection(backend, device, warp_cutoff):
    from stable_envs.envs.dmcontrol.simulation.execution import dmc_backend

    with pytest.raises(ValueError):
        dmc_backend("reacher", 1, device, backend, mjwarp_min_envs=warp_cutoff)


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "key,batch,device,cutoff,expected",
    [
        ("walker", 1, "cuda", None, "native"),
        ("walker", 255, "cuda", None, "native"),
        ("walker", 256, "cuda", None, "mjwarp"),
        ("walker", 4096, "cuda:1", None, "mjwarp"),
        ("walker", 4096, "cpu", None, "native"),
        ("cheetah", 8192, "cuda", None, "native"),
        ("cheetah", 8192, "cuda", 2048, "mjwarp"),
        ("walker", 256, "cuda", 512, "native"),
        ("walker", 512, "cuda", 512, "mjwarp"),
    ],
)
def test_dmc_auto_uses_the_task_cutoff_on_cuda(monkeypatch, key, batch, device, cutoff, expected):
    """Auto is opt-in: native below the cutoff, MuJoCo Warp from it."""
    execution = _fake_adapters(monkeypatch)
    monkeypatch.setattr(execution, "MJWARP_MIN_ENVS", {"walker": 256, "cheetah": None})
    assert execution.dmc_backend(key, batch, device, "auto", mjwarp_min_envs=cutoff) == expected
    kind, args, _ = execution.make_dmc_adapter(
        key, batch, device, 17, simulation_backend="auto", mjwarp_min_envs=cutoff
    )
    assert kind == ("mjwarp" if expected == "mjwarp" else "batch")
    assert args == (key, batch, device, 17)


@pytest.mark.family("dmc")
def test_dmc_cuda_physics_without_warp_names_the_extra(monkeypatch):
    import importlib.util

    from stable_envs.envs.dmcontrol.simulation import mjwarp

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n == "mujoco_warp" else real(n, *a))
    with pytest.raises(ModuleNotFoundError, match=r"stable-envs\[dmcontrol-gpu\]"):
        mjwarp.DMCWarpAdapter("cheetah", 2, "cuda", 0)


@pytest.mark.family("dmc")
def test_dmc_default_is_gpu_physics_on_cuda_and_explicit_names_select(monkeypatch):
    execution = _fake_adapters(monkeypatch)
    for backend, kind in ((None, "mjwarp"), ("native", "batch"), ("mjwarp", "mjwarp")):
        result = execution.make_dmc_adapter("walker", 1, "cuda:1", 17, simulation_backend=backend)
        assert result == (kind, ("walker", 1, "cuda:1", 17), {})
    assert execution.make_dmc_adapter("walker", 1, "cpu", 17) == ("batch", ("walker", 1, "cpu", 17), {})
    for domain, device, backend in (
        ("walker", "cpu", "mjwarp"),
        ("reacher", "cpu", "mjwarp"),
        ("walker", "cuda", "mjx"),
        ("walker", "cuda", "unknown"),
    ):
        with pytest.raises(ValueError):
            execution.make_dmc_adapter(domain, 3, device, 17, simulation_backend=backend)


@pytest.mark.family("dmc")
def test_dmc_cutoff_table_names_simulators_and_positive_cutoffs():
    from stable_envs.envs.dmcontrol.simulation.execution import MJWARP_MIN_ENVS
    from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES

    assert set(MJWARP_MIN_ENVS) == set(FACTORIES)
    assert all(cutoff is None or (isinstance(cutoff, int) and cutoff >= 1) for cutoff in MJWARP_MIN_ENVS.values())


@pytest.mark.family("dmc")
def test_dmc_snapshot_free_step_preserves_active_mask_contract():
    import stable_envs as se
    from tests.helpers.reference.semantics import assert_same_state

    with se.make("DMCCartpoleBalance", num_envs=3, device="cpu", simulation_backend="native", size=32) as env:
        initial = env.get_state()
        actions = torch.zeros(3, env.action_dim, dtype=torch.float64)
        with pytest.raises(ValueError, match="partial active masks are unsupported"):
            env._advance_without_snapshot(actions, active=torch.tensor([True, False, True]))
        assert_same_state(env.get_state(), initial)

        result = env._advance_without_snapshot(actions, active=torch.ones(3, dtype=torch.bool))
        advanced = env.get_state()
        assert "state" not in result.keys()
        env.set_state(initial)
        expected = env._simulator.step(actions)
        assert_same_state(result, expected.exclude("state"))
        assert_same_state(advanced, expected["state"])


@pytest.fixture
def dmc_factory(monkeypatch, tmp_path):
    from stable_envs.envs.dmcontrol.simulation import factory

    monkeypatch.delenv("STABLE_ENVS_DMC_THREADS", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(factory, "make_local", lambda *args: None)
    monkeypatch.setattr(factory, "DMCBatchAdapter", lambda lanes, task, threads, **_: threads)
    tuning.reset()
    tuning._clear_caches()
    yield factory
    tuning._clear_caches()


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "knobs,cpus,batch,expected",
    [
        ({}, 32, 1024, 8),
        ({}, 32, 1, 1),
        ({}, 32, 2, 2),
        ({"dmc_threads": 4}, 32, 1024, 4),
        ({"dmc_threads": 64}, 16, 1024, 16),
        ({"dmc_threads": 8}, None, 1024, 6),
    ],
)
def test_dmc_batch_threads_follow_the_knob_within_cpus_and_lanes(
    monkeypatch, dmc_factory, knobs, cpus, batch, expected
):
    monkeypatch.setattr(dmc_factory, "cpu_limit", lambda: 6 if cpus is None else cpus)
    with tuning.override(**knobs):
        assert dmc_factory.make_batch("cartpole", batch) == expected


@pytest.mark.family("dmc")
def test_dmc_batch_threads_use_the_value_tuned_for_their_batch_range(monkeypatch, dmc_factory):
    monkeypatch.setattr(dmc_factory, "cpu_limit", lambda: 32)
    tuning.write_cache({tuning.entry_key("dmc-sim", 0, 1024): {"dmc_threads": 16}}, "cpu")
    assert dmc_factory.make_batch("cartpole", 1024) == 16
    assert dmc_factory.make_batch("cartpole", 512) == 8
