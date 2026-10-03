"""Only measured Acrobot batches select processes; all other cases stay local."""

from types import SimpleNamespace

import pytest

from stable_envs._adapters import acrobot_process
from stable_envs._adapters import classic_execution as execution


@pytest.mark.parametrize("domain", ["acrobot", "cartpole", "pendulum", "mountain_car", "mountain_car_continuous"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("platform,batch,cores,daemon,count", [
    ("linux", 64, 8, False, 8),
    ("linux", 1024, 32, False, 8),
    ("linux", 64, 4, False, 4),
    ("linux", 64, 6, False, 6),
    ("linux", 63, 8, False, 0),
    ("linux", 64, 2, False, 0),
    ("linux", 64, 8, True, 0),
    ("darwin", 64, 8, False, 0),
    ("win32", 64, None, False, 0),
    ("linux", 64, None, False, 0),
])
def test_classic_routing(monkeypatch, domain, device, platform, batch, cores, daemon, count):
    monkeypatch.setattr(execution.sys, "platform", platform)
    cpus = None if cores is None else [2 * i + 1 for i in range(cores)]
    if cpus is None:
        monkeypatch.delattr(execution.os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: set(cpus), raising=False)
    monkeypatch.setattr(execution.mp, "current_process", lambda: SimpleNamespace(daemon=daemon))
    monkeypatch.setattr(acrobot_process, "AcrobotProcessAdapter", lambda *args, **kwargs: ("process", args, kwargs))
    local = lambda *args: ("local", args, {})
    kind, args, kwargs = execution.make_classic_adapter(domain, batch, device, 17, local)
    expected = count if domain == "acrobot" else 0
    assert kind == ("process" if expected else "local")
    assert args == (domain, batch, device, 17)
    assert kwargs == (dict(processes=expected, cpus=cpus) if expected else {})
