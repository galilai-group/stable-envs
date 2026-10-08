"""Maze route selection respects batch size, CPU allocation and spawn constraints."""

from types import SimpleNamespace

import pytest

from stable_envs._adapters import execution as policy

pytest.importorskip("ogbench")
from stable_envs._adapters.ogbench import execution as execution


@pytest.mark.parametrize("family", ["Maze", "Cube", "Scene"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("platform,batch,cores,daemon,expected", [
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
])
def test_ogbench_routing(monkeypatch, family, device, platform, batch, cores, daemon, expected):
    monkeypatch.setattr(policy.sys, "platform", platform)
    if cores is None:
        monkeypatch.delattr(policy.os, "sched_getaffinity", raising=False)
    else:
        # Real affinity IDs need not be contiguous or start at zero.
        cpus = [2 * i + 1 for i in range(cores)]
        monkeypatch.setattr(policy.os, "sched_getaffinity", lambda _: set(cpus), raising=False)
    monkeypatch.setattr(policy.mp, "current_process", lambda: SimpleNamespace(daemon=daemon))
    monkeypatch.setattr(execution, "OGBMazeProcessAdapter", lambda *args, **kwargs: ("process", args, kwargs))
    local = lambda *args: ("local", args, {})
    if family == "Maze":
        kind, args, kwargs = execution.make_maze_adapter(batch, device, 17, local)
        expected_args = (batch, device, 17)
    else:
        from stable_envs._adapters.ogbench import manipulation_process as manipulation

        monkeypatch.setattr(manipulation, "OGBManipulationProcessAdapter",
                            lambda *args, **kwargs: ("process", args, kwargs))
        kind, args, kwargs = execution.make_manipulation_adapter(family, batch, device, 17, 1, local)
        expected_args = (family, batch, device, 17, 1)
    assert kind == ("process" if expected else "local")
    assert args == expected_args
    assert kwargs == (dict(processes=expected, cpus=cpus) if expected else {})
