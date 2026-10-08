"""Execution selection respects batch size, CPU allocation and process context."""

from types import SimpleNamespace

import pytest

from stable_envs._adapters import execution as policy
from stable_envs._adapters.fetch import execution as execution


@pytest.mark.parametrize(
    "platform,batch,cores,daemon,processes",
    [
        ("linux", 64, 8, False, 8),
        ("linux", 1024, 32, False, 8),
        ("linux", 64, 4, False, 4),
        ("linux", 8, 8, False, 0),
        ("linux", 64, 2, False, 0),
        ("linux", 64, 8, True, 0),
        ("darwin", 64, 8, False, 0),
    ],
)
def test_private_execution_selection(monkeypatch, platform, batch, cores, daemon, processes):
    monkeypatch.setattr(policy.sys, "platform", platform)
    monkeypatch.setattr(policy.os, "sched_getaffinity", lambda _: set(range(cores)), raising=False)
    monkeypatch.setattr(policy.mp, "current_process", lambda: SimpleNamespace(daemon=daemon))
    monkeypatch.setattr(execution, "FetchAdapter", lambda *a, **kw: ("local", a, kw))
    monkeypatch.setattr(execution, "FetchProcessAdapter", lambda *a, **kw: ("process", a, kw))
    mode, args, kwargs = execution.make_fetch_adapter(batch, "cpu", 17, "Push", "dense")
    assert args == (batch, "cpu", 17, "Push", "dense")
    assert mode == ("process" if processes else "local")
    assert kwargs == (
        {"processes": processes, "threads_per_process": 1, "cpus": list(range(cores))} if processes else {}
    )
