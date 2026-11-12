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


def _ogbench(monkeypatch, target, batch, device):
    from stable_envs.envs.ogbench.simulation import execution

    monkeypatch.setattr(execution, "OGBMazeProcessAdapter", _process)
    body = dict(loco="point", maze="large")
    return execution.make_maze_adapter(batch, device, 17, _local), (batch, device, 17), body, body


#: (family, route, target, devices, whether the target may use processes)
TARGETS = [
    ("classic", _classic, "acrobot", ("cpu", "cuda"), True),
    ("classic", _classic, "cartpole", ("cpu", "cuda"), False),
    ("classic", _classic, "mountain_car_continuous", ("cpu", "cuda"), False),
    ("fetch", _fetch, "Push", ("cpu",), True),
    ("ogbench", _ogbench, "Maze", ("cpu", "cuda"), True),
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
    (kind, args, kwargs), expected_args, local_kwargs, process_kwargs = route(monkeypatch, target, batch, device)
    expected = count if eligible else 0
    assert kind == ("process" if expected else "local")
    assert args == expected_args
    assert kwargs == (dict(processes=expected, cpus=cpus, **process_kwargs) if expected else local_kwargs)


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "batch,device,backend,threshold,expected",
    [
        (1, "cuda", "auto", 32, "native"),
        (31, "cuda:1", "auto", 32, "native"),
        (32, "cuda:1", "auto", 32, "mjx"),
        (4096, "cpu", "auto", 32, "native"),
        (4096, "cuda", "native", 32, "native"),
        (1, "cuda", "mjx", 32, "mjx"),
        (1, "cuda", "mjwarp", 32, "mjwarp"),
        (64, "cuda:1", "mjwarp", 32, "mjwarp"),
        (32, "cuda", "auto", 64, "native"),
        (64, "cuda", "auto", 64, "mjx"),
    ],
)
def test_reacher_routes_without_changing_the_output_device(monkeypatch, batch, device, backend, threshold, expected):
    import sys

    from stable_envs.envs.dmcontrol.simulation import execution

    assert execution.reacher_backend(batch, device, backend, threshold) == expected
    monkeypatch.setattr(execution, "make_batch", _batch)
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.reacher_mjx",
        SimpleNamespace(
            ReacherMJXAdapter=lambda *args: ("mjx", args, {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.mjwarp",
        SimpleNamespace(DMCWarpAdapter=lambda *args: ("mjwarp", args, {})),
    )
    kind, args, _ = execution.make_dmc_adapter(
        "reacher",
        batch,
        device,
        17,
        simulation_backend=backend,
        mjx_min_envs=threshold,
    )
    assert kind == (expected if expected in {"mjx", "mjwarp"} else "batch")
    assert args == ((batch, device, 17) if expected == "mjx" else ("reacher", batch, device, 17))


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "backend,device,threshold",
    [
        ("unknown", "cuda", 32),
        ("mjx", "cpu", 32),
        ("mjwarp", "cpu", 32),
        ("auto", "cuda", 0),
        ("auto", "cuda", True),
        ("auto", "cuda", 1.5),
    ],
)
def test_reacher_rejects_invalid_simulation_selection(backend, device, threshold):
    from stable_envs.envs.dmcontrol.simulation.execution import reacher_backend

    with pytest.raises(ValueError):
        reacher_backend(1, device, backend, threshold)


@pytest.mark.family("dmc")
def test_dmc_warp_requires_explicit_cuda_selection(monkeypatch):
    import sys

    from stable_envs.envs.dmcontrol.simulation import execution

    monkeypatch.setattr(execution, "make_batch", _batch)
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.mjwarp",
        SimpleNamespace(DMCWarpAdapter=lambda *args: ("mjwarp", args, {})),
    )
    for backend in ("auto", "native", "mjwarp"):
        result = execution.make_dmc_adapter("walker", 64, "cuda:1", 17, simulation_backend=backend)
        assert result == ("mjwarp" if backend == "mjwarp" else "batch", ("walker", 64, "cuda:1", 17), {})
    for domain, device, backend in (
        ("walker", "cpu", "mjwarp"),
        ("reacher", "cpu", "mjwarp"),
        ("walker", "cuda", "mjx"),
        ("walker", "cuda", "unknown"),
    ):
        with pytest.raises(ValueError):
            execution.make_dmc_adapter(domain, 3, device, 17, simulation_backend=backend)


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
    monkeypatch.setattr(factory, "DMCBatchAdapter", lambda lanes, task, threads: threads)
    monkeypatch.setattr(factory.os, "cpu_count", lambda: 6)
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
    monkeypatch.setattr(dmc_factory, "cpu_affinity", lambda: None if cpus is None else list(range(cpus)))
    with tuning.override(**knobs):
        assert dmc_factory.make_batch("cartpole", batch) == expected


@pytest.mark.family("dmc")
def test_dmc_batch_threads_use_the_value_tuned_for_their_batch_range(monkeypatch, dmc_factory):
    monkeypatch.setattr(dmc_factory, "cpu_affinity", lambda: list(range(32)))
    tuning.write_cache({tuning.entry_key("dmc-sim", 0, 1024): {"dmc_threads": 16}}, "cpu")
    assert dmc_factory.make_batch("cartpole", 1024) == 16
    assert dmc_factory.make_batch("cartpole", 512) == 8
