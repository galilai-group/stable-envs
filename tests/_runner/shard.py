"""``--shard I/N`` keeps the I-th of N duration-balanced shares of the selected tests.

Costs are seconds per node id from ``ci/durations.json``, which lists tests of at least ``LISTED``
seconds; an unlisted test counts as ``UNKNOWN``. Tests parametrized by one env name share a shard,
so the env's per-process caches (official envs, baked scenes) are built in one job, not in each.
Every process computes the same split from the same selection, so the N shards are disjoint and
together are the whole selection, with or without xdist. ``--record-durations PATH`` merges this
run's per-test times into PATH (how ``ci/durations.json`` is refreshed).

``--env-affinity`` widens a group to every test whose parameters name the env (the first CamelCase
name, as in ``[2-17-Scene]``) or another env reproducing the same source (one ``reference_id``, as the
dataset variants of one OGBench maze do), and under xdist's ``load`` mode keeps each group on one
worker, longest groups first. Each worker then holds the caches of its own envs rather than of
every env, at the price of running a group serially: it suits a profile of many short tests over
few memory-heavy envs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

DURATIONS = Path(__file__).resolve().parents[2] / "ci" / "durations.json"
LISTED = 0.5
UNKNOWN = 0.1

_TIMES: dict[str, float] = defaultdict(float)
_ENV_PARAM = re.compile(r"\[([A-Z][A-Za-z0-9]+)\]$")
_PARAMS = re.compile(r"\[(.+)\]$")
_NAME = re.compile(r"[A-Z][A-Za-z0-9]+")
_NOT_NAMES = {"True", "False", "None"}


def pytest_addoption(parser):
    group = parser.getgroup("shard")
    group.addoption("--shard", metavar="I/N", help="run the I-th of N duration-balanced shares of the selection")
    group.addoption("--record-durations", metavar="PATH", help="merge per-test durations into this JSON file")
    group.addoption(
        "--env-affinity",
        action="store_true",
        help="group tests by the source env their parameters name, on one shard and one xdist worker",
    )


def env_group(nodeid: str) -> str:
    """The env a node id is parametrized by, or the node id itself."""
    env = _ENV_PARAM.search(nodeid)
    return env[1] if env else nodeid


def source_group(nodeid: str) -> str:
    """The source env of the first CamelCase name in a node id's parameters, or the node id itself."""
    from stable_envs.registry import REGISTRY

    params = _PARAMS.search(nodeid)
    for token in re.split(r"[-/]", params[1]) if params else ():
        if _NAME.fullmatch(token) and token not in _NOT_NAMES:
            return getattr(REGISTRY.get(token), "reference_id", None) or token
    return nodeid


def _longest_first(nodeids, durations, key) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for nodeid in nodeids:
        groups[key(nodeid)].append(nodeid)
    cost = {k: sum(durations.get(n, UNKNOWN) for n in members) for k, members in groups.items()}
    return [groups[k] for k in sorted(groups, key=lambda k: (-cost[k], k))]


def split(nodeids, count: int, durations: dict[str, float], key=env_group) -> dict[str, int]:
    """Zero-based shard of each node id: groups longest first, each onto the least loaded shard."""
    loads = [0.0] * count
    out = {}
    for members in _longest_first(nodeids, durations, key):
        shard = loads.index(min(loads))
        loads[shard] += sum(durations.get(n, UNKNOWN) for n in members)
        out.update(dict.fromkeys(members, shard))
    return out


def _parse(spec: str) -> tuple[int, int]:
    index, _, count = spec.partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        raise pytest.UsageError(f"--shard takes I/N with 1 <= I <= N, got {spec!r}")
    return int(index), int(count)


def _worker_affinity(config) -> bool:
    return config.getoption("env_affinity") and config.getoption("dist", "no") == "load"


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    spec = config.getoption("shard")
    if not spec and not _worker_affinity(config):
        return
    durations = json.loads(DURATIONS.read_text()) if DURATIONS.exists() else {}
    key = source_group if config.getoption("env_affinity") else env_group
    if spec:
        index, count = _parse(spec)
        shard = split([item.nodeid for item in items], count, durations, key)
        config.hook.pytest_deselected(items=[item for item in items if shard[item.nodeid] != index - 1])
        items[:] = [item for item in items if shard[item.nodeid] == index - 1]
    if _worker_affinity(config):
        # The scheduler below hands out groups in collection order.
        order = _longest_first([item.nodeid for item in items], durations, key)
        rank = {nodeid: i for i, members in enumerate(order) for nodeid in members}
        items.sort(key=lambda item: rank[item.nodeid])


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config, log):
    if not _worker_affinity(config):
        return None
    from xdist.scheduler import LoadScopeScheduling

    class SourceScheduling(LoadScopeScheduling):
        def _split_scope(self, nodeid):
            return source_group(nodeid)

    config.option.loadscopereorder = False
    return SourceScheduling(config, log)


def pytest_runtest_logreport(report):
    _TIMES[report.nodeid] += report.duration


def pytest_sessionfinish(session):
    path = session.config.getoption("record_durations")
    if not path or hasattr(session.config, "workerinput"):
        return
    path = Path(path)
    merged = json.loads(path.read_text()) if path.exists() else {}
    for nodeid, seconds in _TIMES.items():
        merged.pop(nodeid, None)
        if seconds >= LISTED:
            merged[nodeid] = round(seconds, 1)
    path.write_text(json.dumps(dict(sorted(merged.items())), indent=0) + "\n")
