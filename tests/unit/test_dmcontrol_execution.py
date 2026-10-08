"""Candidate routing must respect correctness exclusions and host constraints."""

from types import SimpleNamespace

import pytest

from stable_envs._adapters import execution as policy

pytest.importorskip("dm_control")
from stable_envs._adapters import dmcontrol_execution as execution


@pytest.mark.parametrize("domain", sorted(execution.PROCESS_DOMAINS) + ["quadruped"])
@pytest.mark.parametrize("platform,batch,cores,daemon,expected", [
    ("linux", 64, 8, False, 8),
    ("linux", 256, 32, False, 8),
    ("linux", 64, 4, False, 4),
    ("linux", 63, 8, False, 0),
    ("linux", 64, 2, False, 0),
    ("linux", 64, 8, True, 0),
    ("darwin", 64, 8, False, 0),
])
def test_routing_preserves_local_fallback(monkeypatch, domain, platform, batch, cores, daemon, expected):
    monkeypatch.setattr(policy.sys, "platform", platform)
    monkeypatch.setattr(policy.os, "sched_getaffinity", lambda _: set(range(cores)), raising=False)
    monkeypatch.setattr(policy.mp, "current_process", lambda: SimpleNamespace(daemon=daemon))
    monkeypatch.setattr(execution, "DMCProcessAdapter", lambda *args, **kwargs: ("process", args, kwargs))
    monkeypatch.setattr(execution, "make_local", lambda *args: ("local", args, {}))
    kind, args, kwargs = execution.make_dmc_adapter(domain, batch, "cpu", 17)
    count = expected if domain != "quadruped" else 0
    assert kind == ("process" if count else "local")
    assert args == (domain, batch, "cpu", 17)
    assert kwargs == (dict(processes=count, cpus=list(range(cores))) if count else {})
