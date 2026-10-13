"""``--shard I/N`` keeps the I-th of N duration-balanced shares of the selected tests.

Costs are seconds per node id from ``ci/durations.json``, which lists tests of at least ``LISTED``
seconds; an unlisted test counts as ``UNKNOWN``.
Every process computes the same split from the same selection, so the N shards are disjoint and
together are the whole selection, with or without xdist. ``--record-durations PATH`` merges this
run's per-test times into PATH (how ``ci/durations.json`` is refreshed).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

DURATIONS = Path(__file__).resolve().parents[2] / "ci" / "durations.json"
LISTED = 0.5
UNKNOWN = 0.1

_TIMES: dict[str, float] = defaultdict(float)


def pytest_addoption(parser):
    group = parser.getgroup("shard")
    group.addoption("--shard", metavar="I/N", help="run the I-th of N duration-balanced shares of the selection")
    group.addoption("--record-durations", metavar="PATH", help="merge per-test durations into this JSON file")


def split(nodeids, count: int, durations: dict[str, float]) -> dict[str, int]:
    """Zero-based shard of each node id: longest first, each onto the least loaded shard."""
    loads = [0.0] * count
    out = {}
    for nodeid in sorted(nodeids, key=lambda n: (-durations.get(n, UNKNOWN), n)):
        shard = loads.index(min(loads))
        loads[shard] += durations.get(nodeid, UNKNOWN)
        out[nodeid] = shard
    return out


def _parse(spec: str) -> tuple[int, int]:
    index, _, count = spec.partition("/")
    if not (index.isdigit() and count.isdigit() and 1 <= int(index) <= int(count)):
        raise pytest.UsageError(f"--shard takes I/N with 1 <= I <= N, got {spec!r}")
    return int(index), int(count)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    spec = config.getoption("shard")
    if not spec:
        return
    index, count = _parse(spec)
    durations = json.loads(DURATIONS.read_text()) if DURATIONS.exists() else {}
    shard = split([item.nodeid for item in items], count, durations)
    keep = [item for item in items if shard[item.nodeid] == index - 1]
    config.hook.pytest_deselected(items=[item for item in items if shard[item.nodeid] != index - 1])
    items[:] = keep


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
