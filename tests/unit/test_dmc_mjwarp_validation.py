"""The step's finiteness flags and the cached tolerance constants, without running physics."""

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
wp = pytest.importorskip("warp")
pytest.importorskip("dm_control")


def test_check_kernel_flags_controls_and_each_physics_field():
    # The kernel depends only on Warp; run it on the CPU device.
    from stable_envs.envs.dmcontrol.simulation.mjwarp import _kernels

    wp.init()
    _, check_finite = _kernels()
    names = ("ctrl", "qpos", "qvel", "act", "warmstart", "sensordata")
    fields = {name: wp.zeros((3, 2), dtype=float, device="cpu") for name in names}
    flag = wp.zeros(1, dtype=int, device="cpu")

    def flags():
        flag.zero_()
        wp.launch(check_finite, dim=3, inputs=[*fields.values(), flag], device="cpu")
        return flag.numpy().item()

    assert flags() == 0
    for name, field in fields.items():
        view = wp.to_torch(field).reshape(-1)
        for value in (float("nan"), float("inf"), -float("inf")):
            view[-1] = value  # A failure outside the first environment.
            assert flags() == (1 if name == "ctrl" else 2)
        view.zero_()
        assert flags() == 0
    # Tasks with no activation or sensors still check.
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
