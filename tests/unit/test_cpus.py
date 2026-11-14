"""The CPU count that sizes thread pools: the affinity capped by a cgroup CPU quota."""

from __future__ import annotations

import pytest

from stable_envs._core import cpus

pytestmark = pytest.mark.unit


def _tree(tmp_path, proc, files):
    for name, text in files.items():
        path = tmp_path / "cg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (tmp_path / "proc").write_text(proc)
    return tmp_path / "cg", tmp_path / "proc"


@pytest.mark.parametrize(
    ("proc", "files", "expected"),
    [
        ("0::/\n", {"cpu.max": "1360000 100000\n"}, 13.6),
        ("0::/\n", {"cpu.max": "max 100000\n"}, None),
        ("0::/pod/app\n", {"cpu.max": "800000 100000\n", "pod/cpu.max": "400000 100000\n"}, 4.0),
        ("0::/pod/app\n", {"pod/app/cpu.max": "250000 100000\n"}, 2.5),
        (
            "4:cpu,cpuacct:/docker/abc\n",
            {
                "cpu,cpuacct/docker/abc/cpu.cfs_quota_us": "300000\n",
                "cpu,cpuacct/docker/abc/cpu.cfs_period_us": "100000\n",
            },
            3.0,
        ),
        (
            "4:cpu,cpuacct:/\n",
            {"cpu,cpuacct/cpu.cfs_quota_us": "-1\n", "cpu,cpuacct/cpu.cfs_period_us": "100000\n"},
            None,
        ),
        ("", {}, None),
    ],
)
def test_cgroup_quota(tmp_path, proc, files, expected):
    root, proc_file = _tree(tmp_path, proc, files)
    assert cpus.cgroup_cpu_quota(root, proc_file) == expected


@pytest.mark.parametrize(
    ("affinity", "quota", "expected"), [(128, 13.6, 14), (8, 13.6, 8), (16, None, 16), (4, 0.5, 1)]
)
def test_cpu_limit_caps_the_affinity_by_the_quota(monkeypatch, affinity, quota, expected):
    monkeypatch.setattr(cpus, "_affinity", lambda: affinity)
    monkeypatch.setattr(cpus, "cgroup_cpu_quota", lambda: quota)
    monkeypatch.setattr(cpus.sys, "platform", "linux")
    cpus.cpu_limit.cache_clear()
    try:
        assert cpus.cpu_limit() == expected
    finally:
        cpus.cpu_limit.cache_clear()


def test_user_thread_settings_win(monkeypatch):
    numba = pytest.importorskip("numba")
    monkeypatch.setattr(cpus, "cpu_limit", lambda: 1)
    monkeypatch.setenv("NUMBA_NUM_THREADS", str(numba.config.NUMBA_NUM_THREADS))
    before = numba.get_num_threads()
    assert cpus.numba_threads() == before
    monkeypatch.delenv("NUMBA_NUM_THREADS")
    try:
        assert cpus.numba_threads() == 1
    finally:
        numba.set_num_threads(before)
