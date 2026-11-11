"""Input validation and two-round stepping retain recoverable public boundaries."""

import math

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.gpu._shared import family_params, resolve_family
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]
DMC_NAMES = [name for name, cls in se.registry.REGISTRY.items() if cls.family == "dmcontrol"]


def _require_pairs(runtime):
    if not runtime.supports_pair:
        pytest.skip("requires CUDA graph conditionals")


@pytest.mark.parametrize("active_device", [None, "cpu", "cuda"])
def test_combined_input_validation_reads_one_device_flag(active_device, monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        actions = torch.full((3, 2), 0.25, device="cuda", dtype=torch.float64)[:, ::2]
        assert not actions.is_contiguous()
        active = None if active_device is None else torch.ones(3, device=active_device, dtype=torch.bool)
        reads = []

        def record(method):
            def conversion(value):
                if value.device.type == "cuda":
                    reads.append(method.__name__)
                return method(value)

            return conversion

        with monkeypatch.context() as patch:
            for name in ("__bool__", "__int__", "item"):
                patch.setattr(torch.Tensor, name, record(getattr(torch.Tensor, name)))
            checked, controls = adapter._prepare_step(actions, active)
        assert reads == ["__int__"]
        assert_same_state(checked, actions)
        assert controls.is_contiguous() and controls.dtype == torch.float32
        assert_same_state(controls, adapter._controls(actions))
        checked, mask = adapter.check_step(actions, active)
        assert_same_state(checked, actions)
        assert mask is None
    finally:
        adapter.close()


def test_combined_invalid_inputs_preserve_pending_resets_and_recover(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        actions = torch.zeros(3, 1, device="cuda", dtype=torch.float64)
        adapter.step(actions)
        pairs = dict(adapter._runtime._pairs)
        initial = adapter.get_state()
        initial["lifecycle", "reset_next_step"].fill_(True)
        adapter.set_state(initial)
        native = adapter._source.get_state()
        physics = adapter._runtime.physical_state()
        counts, pending = adapter._counts.copy(), adapter._pending.copy()

        def unexpected_mutation(*args, **kwargs):
            pytest.fail("Invalid input reached physics or reset RNG")

        with monkeypatch.context() as patch:
            patch.setattr(adapter._runtime, "_advance", unexpected_mutation)
            patch.setattr(adapter._runtime, "release_graphs", unexpected_mutation)
            patch.setattr(adapter._source, "prepare_reset", unexpected_mutation)
            for bad in (float("nan"), float("inf"), -float("inf"), 1e300):
                invalid = actions.clone()
                invalid[-1] = bad
                with pytest.raises(ValueError, match="finite"):
                    adapter.step(invalid)
            for device in ("cpu", "cuda"):
                for mask in (torch.tensor([True, True, False], device=device), torch.ones(3, 1, device=device)):
                    with pytest.raises(ValueError, match="partial active masks"):
                        adapter.step(actions, active=mask)
            with pytest.raises(ValueError, match="action must have finite shape"):
                adapter.step(actions[:, :0])
        assert_same_state(adapter.get_state(), initial)
        assert_same_state(adapter._source.get_state(), native)
        assert_same_state(adapter._runtime.physical_state(), physics)
        assert adapter._runtime._pairs == pairs
        np.testing.assert_array_equal(adapter._counts, counts)
        np.testing.assert_array_equal(adapter._pending, pending)
        adapter.step(actions)
        assert adapter.get_state()["lifecycle", "step_count"].flatten().tolist() == [1, 1, 1]
    finally:
        adapter.close()


def test_combined_validation_preserves_error_priority():
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        actions = torch.full((3, 1), 1e300, device="cuda", dtype=torch.float64)
        bad_mask = torch.zeros(3, device="cuda", dtype=torch.bool)
        with pytest.raises(ValueError, match="partial active masks"):
            adapter.check_step(actions, bad_mask)
        actions[-1] = float("nan")
        with pytest.raises(ValueError, match="action must have finite shape"):
            adapter.check_step(actions, bad_mask)
        with pytest.raises(ValueError, match="action must have finite shape"):
            adapter.check_step(actions, torch.ones(3, 1, device="cuda"))
        with pytest.raises(ValueError, match="action must have finite shape"):
            adapter.check_step(actions[:, :0], bad_mask)
    finally:
        adapter.close()


@pytest.mark.parametrize("name", family_params(DMC_NAMES, family="dmc"))
def test_pair_matches_checked_rounds_for_registered_tasks(name, monkeypatch):
    resolve_family(name, family="dmc", modules=("dmcontrol._family",), subject="MJWarp paired rounds")
    with se.make(name, num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        adapter = env._simulator
        _require_pairs(adapter._runtime)
        initial = adapter.reset(91000)
        actions = torch.as_tensor(
            np.random.default_rng(92717).uniform(-1, 1, (4, 3, adapter.action_dim)),
            device="cuda",
            dtype=torch.float64,
        )
        adapter.step(actions[0])
        adapter.set_state(initial)
        with monkeypatch.context() as patch:
            patch.setattr(adapter._runtime, "supports_pair", False)
            expected = [adapter.step(action) for action in actions]
        adapter.set_state(initial)
        actual = [adapter.step(action) for action in actions]
        for got, want in zip(actual, expected, strict=True):
            assert_same_state(got, want)


def test_warmed_pair_has_one_input_and_one_result_read(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        _require_pairs(adapter._runtime)
        actions = torch.full((3, 1), 0.25, device="cuda", dtype=torch.float64)
        adapter.step(actions)
        reads = []

        def record(method):
            def conversion(value):
                if value.device.type == "cuda":
                    reads.append(method.__name__)
                return method(value)

            return conversion

        def unexpected_cleanup(*args, **kwargs):
            pytest.fail("A warmed ordinary step synchronized or cleared the allocator cache")

        with monkeypatch.context() as patch:
            for name in ("__bool__", "__int__", "item"):
                patch.setattr(torch.Tensor, name, record(getattr(torch.Tensor, name)))
            patch.setattr(torch.cuda, "empty_cache", unexpected_cleanup)
            patch.setattr(torch.cuda, "synchronize", unexpected_cleanup)
            patch.setattr(torch.cuda.Stream, "synchronize", unexpected_cleanup)
            patch.setattr(torch.cuda.Event, "synchronize", unexpected_cleanup)
            adapter.step(actions)
        assert reads == ["__int__", "__int__"]
    finally:
        adapter.close()


@pytest.mark.parametrize("reset_route", ["one_pending", "all_pending", "reset_in_second"])
def test_route_switch_releases_inactive_graphs_and_preserves_outputs(reset_route, monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        runtime, task = adapter._runtime, adapter._task
        _require_pairs(runtime)
        initial = adapter.get_state()
        mixed = initial.clone()
        counts, pending = mixed["lifecycle", "step_count"], mixed["lifecycle", "reset_next_step"]
        counts.fill_(6)
        limits = torch.as_tensor(np.ceil(adapter._limits), device="cuda", dtype=counts.dtype)
        if reset_route == "all_pending":
            counts[:, 0] = limits
            pending.fill_(True)
        elif reset_route == "one_pending":
            counts[0, 0] = limits[0]
            pending[0, 0] = True
        else:
            counts[0, 0] = limits[0] - 1
        actions = torch.full((3, 1), 0.125, device="cuda", dtype=torch.float64)

        # The checked path supplies a reference for both sides of the switch,
        # including the source RNG after the pending reset has committed.
        adapter.set_state(mixed)
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "supports_pair", False)
            expected_mixed = adapter.step(actions)
            expected_native = adapter._source.get_state()
            expected_ordinary = adapter.step(actions)
        expected_reset = adapter.reset(None)

        adapter.set_state(initial)
        held = adapter.step(actions)
        held_copy = held.clone()
        pair = runtime._pairs[task]
        assert not runtime._graphs and adapter._scratch is None
        adapter.set_state(initial)
        assert runtime._pairs[task] is pair, "An ordinary restore discarded its warmed pair"

        first_stream, second_stream = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(first_stream):
            adapter.set_state(mixed)
            actual_mixed = adapter.step(actions)
        # get_state orders the previous caller stream before comparisons.
        actual_state = adapter.get_state()
        assert not runtime._pairs and runtime._graphs
        scratch = adapter._scratch
        if reset_route == "all_pending":
            assert scratch is None
        else:
            assert scratch is not None and scratch.num_envs == 2
        assert_same_state(actual_mixed, expected_mixed)
        assert_same_state(actual_state, expected_mixed["state"])
        assert_same_state(adapter._source.get_state(), expected_native)
        assert_same_state(held, held_copy)
        mixed_copy = actual_mixed.clone()

        with torch.cuda.stream(second_stream):
            actual_ordinary = adapter.step(actions)
        adapter.get_state()
        assert runtime._pairs[task] is not pair
        assert not runtime._graphs and adapter._scratch is None
        if scratch is not None:
            assert scratch._closed
        assert_same_state(actual_ordinary, expected_ordinary)
        assert_same_state(actual_mixed, mixed_copy)
        assert_same_state(held, held_copy)
        assert_same_state(adapter.reset(None), expected_reset)
    finally:
        adapter.close()


def test_failed_reset_route_after_pair_eviction_preserves_state_and_rng(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter
    from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import WarpRuntime

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        runtime = adapter._runtime
        _require_pairs(runtime)
        actions = torch.full((3, 1), 0.125, device="cuda", dtype=torch.float64)
        held = adapter.step(actions)
        held_copy = held.clone()
        initial = adapter.get_state()
        initial["lifecycle", "step_count"][0, 0] = math.ceil(adapter._limits[0])
        initial["lifecycle", "reset_next_step"][0, 0] = True
        adapter.set_state(initial)
        assert runtime._pairs and adapter._scratch is None
        native = adapter._source.get_state()
        physics = runtime.physical_state()
        check_result = WarpRuntime._check_result

        def fail_continuing_round(candidate, *, recorded=False):
            if candidate is not runtime and recorded:
                assert not runtime._pairs, "Reset physics retained the inactive paired graph"
                candidate._data_view("qvel")[-1, -1] = float("nan")
                return check_result(candidate)
            return check_result(candidate, recorded=recorded)

        def unexpected_reset(*args, **kwargs):
            pytest.fail("Failed continuing physics consumed pending reset RNG")

        with monkeypatch.context() as patch:
            patch.setattr(WarpRuntime, "_check_result", fail_continuing_round)
            patch.setattr(adapter._source, "prepare_reset", unexpected_reset)
            with pytest.raises(RuntimeError, match="produced nonfinite qvel"):
                adapter.step(actions)
        assert not runtime._pairs
        assert_same_state(adapter.get_state(), initial)
        assert_same_state(adapter._source.get_state(), native)
        assert_same_state(runtime.physical_state(), physics)
        assert_same_state(held, held_copy)
        # Recovery must reuse the same CUDA context after releasing the pair.
        adapter.set_state(initial)
        output = adapter.step(actions)
        assert output["state", "lifecycle", "step_count"].flatten().tolist() == [1, 4, 4]
        assert not adapter._pending.any()
        adapter.step(actions)
        assert runtime._pairs and adapter._scratch is None
        assert_same_state(held, held_copy)
    finally:
        adapter.close()


def test_finger_pair_reuses_one_round_of_retained_scratch(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter
    from stable_envs.envs.dmcontrol.simulation.mjwarp_pair import PairedRound, _RetainedScratch

    adapter = DMCWarpAdapter("finger-spin", 4, "cuda", 17)
    try:
        runtime = adapter._runtime
        _require_pairs(runtime)
        requests, rounds = [], []
        allocate, record_round = _RetainedScratch.allocate, PairedRound._record_round

        def record_allocation(scratch, size):
            pointer = allocate(scratch, size)
            requests.append((scratch, size, pointer))
            return pointer

        def record_boundary(pair, round_index):
            record_round(pair, round_index)
            assert pair.runtime is runtime
            assert all(scratch is pair._scratch for scratch, _, _ in requests)
            rounds.append((round_index, tuple((size, pointer) for _, size, pointer in requests)))
            requests.clear()

        monkeypatch.setattr(_RetainedScratch, "allocate", record_allocation)
        monkeypatch.setattr(PairedRound, "_record_round", record_boundary)
        actions = torch.full((4, adapter.action_dim), 0.25, device="cuda", dtype=torch.float64)
        adapter.step(actions)
        assert [index for index, _ in rounds] == [1, 2, 1, 2]
        first = rounds[0][1]
        assert first, "Finger physics must exercise retained scratch allocations"
        assert all(sequence == first for _, sequence in rounds)
        scratch = runtime._pairs[adapter._task]._scratch
        assert scratch.sizes == [size for size, _ in first]
        assert [buffer.data_ptr() for buffer in scratch.buffers] == [pointer for _, pointer in first]
        assert sum(scratch.sizes) == sum(buffer.numel() for buffer in scratch.buffers)

        # Replays execute recorded work without allocating another round's pool.
        adapter.step(actions)
        adapter.step(actions)
        assert len(rounds) == 4 and not requests
        assert [buffer.data_ptr() for buffer in scratch.buffers] == [pointer for _, pointer in first]
        assert all(torch.isfinite(value).all() for value in runtime.physical_state().values())
    finally:
        adapter.close()


@pytest.mark.parametrize("domain", ["cartpole-balance", "finger-spin"])
@pytest.mark.parametrize("failed_round", [1, 2])
def test_pair_device_failure_guards_second_round_and_recovers(domain, failed_round, monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter
    from stable_envs.envs.dmcontrol.simulation.mjwarp_pair import PairedRound, PairFailure

    adapter = DMCWarpAdapter(domain, 3, "cuda", 17)
    try:
        runtime = adapter._runtime
        _require_pairs(runtime)
        initial = adapter.get_state()
        initial["task", "cumulative_reward"][:, 0] = torch.tensor([11.0, 17.0, 23.0], device="cuda")
        adapter.set_state(initial)
        actions = torch.full((3, adapter.action_dim), 0.25, device="cuda", dtype=torch.float64)
        adapter._control_step(adapter._controls(actions))
        after_first = adapter.get_state()
        adapter.set_state(initial)
        native = adapter._source.get_state()
        fault = torch.zeros((), dtype=torch.int32, device="cuda")
        fault_matches = torch.zeros(2, dtype=torch.bool, device="cuda")
        nonfinite = torch.full((), float("nan"), dtype=torch.float32, device="cuda")
        rounds = torch.zeros(2, dtype=torch.int32, device="cuda")
        record_round = PairedRound._record_round

        def inject_failure(pair, round_index):
            assert pair.runtime is runtime
            rounds[round_index - 1].add_(1)
            last = runtime._data_view("qvel")[-1, -1]
            torch.eq(fault, round_index, out=fault_matches[round_index - 1])
            torch.where(fault_matches[round_index - 1], nonfinite, last, out=last)
            record_round(pair, round_index)

        monkeypatch.setattr(PairedRound, "_record_round", inject_failure)
        expected = adapter.step(actions)
        adapter.set_state(initial)
        rounds.zero_()
        fault.fill_(failed_round)
        with pytest.raises(PairFailure, match="produced nonfinite qvel") as caught:
            adapter.step(actions)
        assert caught.value.completed_rounds == failed_round - 1
        assert rounds.tolist() == ([1, 0] if failed_round == 1 else [1, 1])
        checkpoint = initial if failed_round == 1 else after_first
        assert_same_state(adapter.get_state(), checkpoint)
        assert_same_state(adapter._source.get_state(), native)
        np.testing.assert_array_equal(adapter._counts, checkpoint["lifecycle", "step_count"].cpu().numpy()[:, 0])
        np.testing.assert_array_equal(adapter._pending, checkpoint["lifecycle", "reset_next_step"].cpu().numpy()[:, 0])
        assert_same_state(adapter.get_state()["task", "cumulative_reward"], initial["task", "cumulative_reward"])

        # Recovery must use the same CUDA context and graph; a stale packed
        # failure must be cleared before the following replay's guard is read.
        fault.zero_()
        adapter.set_state(initial)
        runtime._pairs[adapter._task].status.fill_(127)
        rounds.zero_()
        assert_same_state(adapter.step(actions), expected)
        assert rounds.tolist() == [1, 1]
    finally:
        adapter.close()


@pytest.mark.parametrize("route", ["ordinary", "ends_at_limit", "reset_in_second", "pending_at_start"])
def test_pair_routing_preserves_reset_boundaries(route, monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        runtime = adapter._runtime
        _require_pairs(runtime)
        actions = torch.full((3, 1), 0.25, device="cuda", dtype=torch.float64)
        adapter.step(actions)
        initial = adapter.reset(91000)
        limit = math.ceil(adapter._limits[0])
        if route == "ends_at_limit":
            initial["lifecycle", "step_count"][0, 0] = limit - 2
        elif route == "reset_in_second":
            initial["lifecycle", "step_count"][0, 0] = limit - 1
        elif route == "pending_at_start":
            initial["lifecycle", "step_count"][0, 0] = limit
            initial["lifecycle", "reset_next_step"][0, 0] = True
        adapter.set_state(initial)
        with monkeypatch.context() as patch:
            patch.setattr(runtime, "supports_pair", False)
            expected = adapter.step(actions)
        adapter.set_state(initial)
        calls = []
        advance_pair, control_step = runtime.advance_pair, adapter._control_step

        def paired(*args, **kwargs):
            calls.append("pair")
            return advance_pair(*args, **kwargs)

        def single(*args, **kwargs):
            calls.append("round")
            return control_step(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "advance_pair", paired)
            patch.setattr(adapter, "_control_step", single)
            actual = adapter.step(actions)
        assert calls == (["pair"] if route in {"ordinary", "ends_at_limit"} else ["round", "round"])
        assert_same_state(actual, expected)
    finally:
        adapter.close()


def test_failed_round_before_reset_does_not_consume_reset_rng(monkeypatch):
    from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter

    adapter = DMCWarpAdapter("cartpole-balance", 3, "cuda", 17)
    try:
        initial = adapter.get_state()
        initial["lifecycle", "step_count"][0, 0] = math.ceil(adapter._limits[0]) - 1
        adapter.set_state(initial)
        native = adapter._source.get_state()
        actions = torch.full((3, 1), 0.25, device="cuda", dtype=torch.float64)
        check_result = adapter._runtime._check_result

        def fail_first(*, recorded=False):
            if recorded:
                adapter._runtime._data_view("qvel")[-1, -1] = float("nan")
                return check_result()
            return check_result(recorded=recorded)

        def unexpected_reset(*args, **kwargs):
            pytest.fail("A failed first round consumed a second-round reset")

        with monkeypatch.context() as patch:
            patch.setattr(adapter._runtime, "_check_result", fail_first)
            patch.setattr(adapter._source, "prepare_reset", unexpected_reset)
            with pytest.raises(RuntimeError, match="produced nonfinite qvel"):
                adapter.step(actions)
        assert_same_state(adapter.get_state(), initial)
        assert_same_state(adapter._source.get_state(), native)
    finally:
        adapter.close()
