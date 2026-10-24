"""Process routing respects batch size, CPU allocation, platform, process context and each family's exclusions."""

from types import SimpleNamespace

import pytest

from stable_envs._execution import policy


def _process(*args, **kwargs):
    return ("process", args, kwargs)


def _local(*args, **kwargs):
    return ("local", args, kwargs)


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
    if target == "Maze":
        body = dict(loco="point", maze="large")
        return execution.make_maze_adapter(batch, device, 17, _local), (batch, device, 17), body, body
    from stable_envs.envs.ogbench.simulation import manipulation_process

    monkeypatch.setattr(manipulation_process, "OGBManipulationProcessAdapter", _process)
    args = (target, batch, device, 17, 1)
    return execution.make_manipulation_adapter(target, batch, device, 17, 1, _local), args, {}, {}


#: (family, route, target, devices, whether the target may use processes)
TARGETS = [
    ("classic", _classic, "acrobot", ("cpu", "cuda"), True),
    ("classic", _classic, "cartpole", ("cpu", "cuda"), False),
    ("classic", _classic, "mountain_car_continuous", ("cpu", "cuda"), False),
    ("fetch", _fetch, "Push", ("cpu",), True),
    ("ogbench", _ogbench, "Maze", ("cpu", "cuda"), True),
    ("ogbench", _ogbench, "Cube", ("cpu", "cuda"), True),
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
        (32, "cuda", "auto", 64, "native"),
        (64, "cuda", "auto", 64, "mjx"),
    ],
)
def test_reacher_routes_without_changing_the_output_device(monkeypatch, batch, device, backend, threshold, expected):
    import sys

    from stable_envs.envs.dmcontrol.simulation import execution

    monkeypatch.setattr(execution, "make_batch", _local)
    monkeypatch.setitem(
        sys.modules,
        "stable_envs.envs.dmcontrol.simulation.reacher_mjx",
        SimpleNamespace(
            ReacherMJXAdapter=lambda *args: ("mjx", args, {}),
        ),
    )
    kind, args, _ = execution.make_dmc_adapter(
        "reacher",
        batch,
        device,
        17,
        simulation_backend=backend,
        mjx_min_envs=threshold,
    )
    assert kind == ("mjx" if expected == "mjx" else "local")
    assert args == ((batch, device, 17) if expected == "mjx" else ("reacher", batch, device, 17))


@pytest.mark.family("dmc")
@pytest.mark.parametrize(
    "backend,device,threshold",
    [
        ("unknown", "cuda", 32),
        ("mjx", "cpu", 32),
        ("auto", "cuda", 0),
        ("auto", "cuda", True),
        ("auto", "cuda", 1.5),
    ],
)
def test_reacher_rejects_invalid_simulation_selection(backend, device, threshold):
    from stable_envs.envs.dmcontrol.simulation.execution import reacher_backend

    with pytest.raises(ValueError):
        reacher_backend(1, device, backend, threshold)
