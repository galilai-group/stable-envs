"""Input validation and two-round stepping retain recoverable public boundaries."""

import numpy as np
import pytest
import torch

from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


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
        initial = adapter.get_state()
        initial["lifecycle", "reset_next_step"].fill_(True)
        adapter.set_state(initial)
        native = adapter._source.get_state()
        physics = adapter._runtime.physical_state()
        counts, pending = adapter._counts.copy(), adapter._pending.copy()
        actions = torch.zeros(3, 1, device="cuda", dtype=torch.float64)

        def unexpected_mutation(*args, **kwargs):
            pytest.fail("Invalid input reached physics or reset RNG")

        with monkeypatch.context() as patch:
            patch.setattr(adapter._runtime, "_advance", unexpected_mutation)
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
