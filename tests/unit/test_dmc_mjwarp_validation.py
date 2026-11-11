"""Result flags retain every validation category without running physics."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
wp = pytest.importorskip("warp")

from stable_envs.envs.dmcontrol.simulation import mjwarp_runtime


def test_result_bitmask_reports_fields_capacity_and_clears_between_calls(monkeypatch):
    # This kernel only depends on Warp; exercise it without requiring a matching
    # MuJoCo physics installation or a CUDA device.
    monkeypatch.setattr(mjwarp_runtime, "wp", wp)
    wp.init()
    fields = {name: wp.zeros((3, 2), dtype=float, device="cpu") for name in mjwarp_runtime._RESULT_FIELDS}
    fields["time"] = wp.zeros(3, dtype=float, device="cpu")
    overflow = wp.zeros(3, dtype=int, device="cpu")
    nacon = wp.zeros(1, dtype=int, device="cpu")
    error = wp.zeros(1, dtype=int, device="cpu")

    def flags():
        error.zero_()
        wp.launch(
            mjwarp_runtime._result_check_kernel(),
            dim=3,
            inputs=[overflow, nacon, 3, 128, *fields.values()],
            outputs=[error],
            device="cpu",
        )
        return error.numpy().item()

    assert flags() == 0
    for index, field in enumerate(fields.values(), start=1):
        view = wp.to_torch(field).reshape(-1)
        for value in (float("nan"), float("inf"), -float("inf")):
            view[-1] = value  # Include a failure outside the first lane.
            assert flags() == 1 << index
        view.zero_()
        assert flags() == 0

    # Finite solver-budget exits do not count as capacity failures.
    wp.to_torch(overflow)[2] = 4
    wp.to_torch(nacon)[0] = 128
    assert flags() == 0
    wp.to_torch(overflow)[2] = 5
    assert flags() == 1
    wp.to_torch(overflow).zero_()
    wp.to_torch(nacon)[0] = 129
    wp.to_torch(fields["qvel"])[1, 0] = float("nan")
    assert flags() == 1 | 4
    wp.to_torch(nacon).zero_()
    wp.to_torch(fields["qvel"]).zero_()
    # Tasks with no activation or acceleration sensors still validate.
    fields["act"] = wp.zeros((3, 0), dtype=float, device="cpu")
    fields["sensordata"] = wp.zeros((3, 0), dtype=float, device="cpu")
    assert flags() == 0


@pytest.mark.parametrize("sigmoid", ["gaussian", "long_tail", "linear", "quadratic"])
def test_cached_tolerance_scalars_match_tensor_parameters(sigmoid):
    from stable_envs.envs.dmcontrol.simulation.mjwarp_tasks import _tolerance

    values = torch.tensor([-np.inf, -1.0, -0.3, -0.0, 0.0, 0.1, 0.7, 1.0, np.inf, np.nan], dtype=torch.float64)
    for margin in (0.0, 0.3, np.float64(0.7)):
        expected = _tolerance(
            values,
            bounds=(torch.tensor(-0.3, dtype=torch.float64), torch.tensor(0.1, dtype=torch.float64)),
            margin=torch.tensor(margin, dtype=torch.float64),
            sigmoid=sigmoid,
        )
        actual = _tolerance(values, (-0.3, 0.1), margin, sigmoid)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
