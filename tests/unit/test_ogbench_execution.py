"""Maze route selection respects batch size, CPU allocation and spawn constraints."""

from types import SimpleNamespace

import pytest

pytest.importorskip("ogbench")
from stable_envs._adapters import ogbench_execution as execution


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
def test_maze_routing(monkeypatch, device, platform, batch, cores, daemon, expected):
    monkeypatch.setattr(execution.sys, "platform", platform)
    if cores is None:
        monkeypatch.delattr(execution.os, "sched_getaffinity", raising=False)
    else:
        # Real affinity IDs need not be contiguous or start at zero.
        cpus = [2 * i + 1 for i in range(cores)]
        monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: set(cpus), raising=False)
    monkeypatch.setattr(execution.mp, "current_process", lambda: SimpleNamespace(daemon=daemon))
    monkeypatch.setattr(execution, "OGBMazeProcessAdapter", lambda *args, **kwargs: ("process", args, kwargs))
    local = lambda *args: ("local", args, {})
    kind, args, kwargs = execution.make_maze_adapter(batch, device, 17, local)
    assert kind == ("process" if expected else "local")
    assert args == (batch, device, 17)
    assert kwargs == (dict(processes=expected, cpus=cpus) if expected else {})
