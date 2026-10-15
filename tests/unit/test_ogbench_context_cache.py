"""Exercise mixed-workload residency and safe eviction without a CUDA driver."""

import pytest

from stable_envs.envs.ogbench.context_cache import ContextCache


def cache(budget=600):
    released = []
    return ContextCache(released.append, lambda device: (4000, 8000), budget), released


def key(config, device=0):
    return device, config, 224, 16


def test_repeated_six_context_workload_does_not_rebuild_after_warmup():
    c, released = cache()
    keys = [key((config, render_pass)) for config in range(3) for render_pass in range(2)]
    builds = 0
    for _ in range(20):
        with c.working_set(keys):
            for k in keys:
                if c.get(k) is None:
                    builds += 1
                    c.prepare(k)
                    c.put(k, object(), 100)
    assert builds == 6
    assert c.evictions == 0
    assert c.stats()["retained_bytes"] == {0: 600}
    assert released == []


def test_accounting_is_bytes_not_number_of_contexts():
    c, released = cache(100)
    for i in range(10):
        c.put(key(i), i, 10)
    c.put(key("large"), "large", 60)
    assert c.stats()["retained_bytes"] == {0: 100}
    assert released == list(range(6))
    assert len(c) == 5


def test_active_workload_survives_inactive_entries_even_if_older():
    c, released = cache(200)
    c.put(key("active"), "active", 100)
    c.put(key("inactive"), "inactive", 100)
    with c.working_set([key("active"), key("new")]):
        c.put(key("new"), "new", 100)
    assert released == ["inactive"]
    assert c.active_evictions == 0


def test_compositor_lease_protects_first_pass_during_second_pass_admission():
    c, released = cache(200)
    c.put(key("first"), "first", 100)
    c.put(key("unrelated"), "unrelated", 100)
    with c.lease([key("first"), key("second")]):
        c.put(key("second"), "second", 100)
        assert c.get(key("first")) == "first"
        assert released == ["unrelated"]
    c.put(key("next"), "next", 100)
    assert len(c) == 2


def test_over_budget_lease_fails_without_releasing_live_buffers():
    c, released = cache(100)
    c.put(key("first"), "first", 75)
    with c.lease([key("first"), key("second")]):
        with pytest.raises(MemoryError, match="Reduce chunk_size"):
            c.put(key("second"), "second", 75)
        with pytest.raises(RuntimeError, match="leased"):
            c.clear()
    assert c.get(key("first")) == "first"
    assert c.stats()["retained_bytes"] == {0: 75}
    assert released == []


def test_larger_than_budget_workload_streams_and_reports_active_evictions():
    c, released = cache(200)
    keys = [key(i) for i in range(3)]
    with c.working_set(keys):
        for k in keys:
            with c.lease([k]):
                c.put(k, k, 100)
                assert c.stats()["retained_bytes"][0] <= 200
    assert released == [keys[0]]
    assert c.active_evictions == 1


def test_device_budgets_are_independent_and_keys_do_not_alias():
    c, released = cache(100)
    c.put(key("same", 0), "gpu0", 100)
    c.put(key("same", 1), "gpu1", 100)
    c.put(key("next", 0), "gpu0-new", 100)
    assert released == ["gpu0"]
    assert c.get(key("same", 1)) == "gpu1"
    assert c.stats()["retained_bytes"] == {0: 100, 1: 100}


def test_known_size_evicts_before_context_reconstruction():
    c, released = cache(100)
    c.put(key("a"), "a", 100)
    c.put(key("b"), "b", 100)
    c.prepare(key("a"))
    assert released == ["a", "b"]
    assert c.stats()["retained_bytes"][0] == 0


def test_default_budget_leaves_headroom_and_explicit_budget_is_supported(monkeypatch):
    monkeypatch.delenv("STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES", raising=False)
    c, _ = cache(None)
    assert c.budget(0) == 2000
    monkeypatch.setenv("STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES", "1234")
    other, _ = cache(None)
    assert other.budget(0) == 1234


def test_the_budget_knob_is_read_at_call_time(monkeypatch):
    from stable_envs import tuning

    monkeypatch.delenv("STABLE_ENVS_OGB_CONTEXT_CACHE_BYTES", raising=False)
    c, _ = cache(None)
    assert c.budget(0) == 2000
    with tuning.override(ogb_context_cache_bytes=4321):
        assert c.budget(0) == 4321
    assert c.budget(0) == 2000


def test_lease_and_working_set_are_restored_after_exceptions():
    c, released = cache(100)
    with pytest.raises(ValueError):
        with c.working_set([key("a")]), c.lease([key("a")]):
            c.put(key("a"), "a", 100)
            raise ValueError("drawing failed")
    c.clear()
    assert released == ["a"]
    assert c.stats()["retained_bytes"] == {}
    assert not c._active
