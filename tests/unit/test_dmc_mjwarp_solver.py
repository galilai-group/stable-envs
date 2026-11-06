"""Small numerical witnesses for Humanoid's ordered Newton reductions."""

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from stable_envs.envs.dmcontrol.simulation import mjwarp_runtime  # noqa: E402

pytestmark = pytest.mark.family("dmc")
DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA solver kernels")],
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_changed_row_order_preserves_incremental_hessian(device):
    from mujoco_warp._src import solver

    mjwarp_runtime._load_backend()
    sort_ids, _ = mjwarp_runtime._pyramidal_solver_kernels()
    # Retained Humanoid seed 91010, microstep 69, first Newton iteration:
    # identical changed-row sets arrived in these two different orders.
    order = np.array([43, 50, 52, 55, 7, 29, 31], dtype=np.int32)
    ids = np.tile(np.r_[order, [999, -999]], (3, 1)).astype(np.int32)
    changed = wp.array([7, 7, 0], dtype=int, device=device)
    done = wp.array([False, True, False], dtype=bool, device=device)
    packed = wp.array(ids, dtype=int, device=device)
    wp.launch(sort_ids, dim=3, inputs=[changed, done], outputs=[packed], device=device)
    expected_ids = ids.copy()
    expected_ids[0, :7] = np.sort(order)
    np.testing.assert_array_equal(packed.numpy(), expected_ids)
    np.testing.assert_array_equal(changed.numpy(), [7, 7, 0])

    # The same witness's J[:, 2], D and state determine this Hessian entry.
    # Retaining a different append order changes its float32 sum by one ULP.
    jacobian = np.zeros((1, 56, 1), dtype=np.float32)
    jacobian[0, order, 0] = [1, 1, 1, 1, 0, 1, 1]
    diagonal = np.zeros((1, 56), dtype=np.float32)
    diagonal[0, order] = [
        108.21866607666016,
        2.759406328201294,
        2.759406328201294,
        9.909588813781738,
        4.970590114593506,
        108.21866607666016,
        108.21866607666016,
    ]
    state = np.zeros((1, 56), dtype=np.int32)
    state[0, order] = int(solver.types.ConstraintState.QUADRATIC)
    inputs = [wp.array(value, device=device) for value in (jacobian, diagonal, state)]

    def advance_hessian(changed_ids):
        hessian = wp.array(np.array([[[40.84402]]], dtype=np.float32), device=device)
        wp.launch(
            solver._update_gradient_h_incremental,
            dim=(1, 1),
            inputs=[
                *inputs,
                wp.array(changed_ids[None, :], dtype=int, device=device),
                wp.array([7], dtype=int, device=device),
            ],
            outputs=[hessian],
            device=device,
        )
        return hessian.numpy()

    reference = advance_hessian(np.sort(order))
    assert not np.array_equal(advance_hessian(order), reference), "witness must detect omitted row ordering"
    np.testing.assert_array_equal(advance_hessian(packed.numpy()[0, :7]), reference)


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_norm_preserves_completed_and_unchanged_worlds(device):
    mjwarp_runtime._load_backend()
    _, norm = mjwarp_runtime._pyramidal_solver_kernels()
    # The initial gradient at the same retained microstep. Two atomic sums of
    # these identical values were 124928760 and 124928736 on the failing run.
    witness = np.array(
        [
            580.3350219726562,
            1566.0443115234375,
            9045.5478515625,
            2564.616943359375,
            3730.5830078125,
            2191.806396484375,
            1534.5321044921875,
            2905.971435546875,
            -245.6632080078125,
            850.9754028320312,
            57.05607604980469,
            441.85400390625,
            187.67861938476562,
            -1400.049560546875,
            100.74884033203125,
            38.65321731567383,
            68.47366333007812,
            983.3436279296875,
            16.843978881835938,
            5.088052749633789,
            -1.798999547958374,
            39.035118103027344,
            -36.44234848022461,
            -65.60667419433594,
            208.62318420410156,
            439.9317321777344,
            -115.54176330566406,
        ],
        dtype=np.float32,
    )
    reference = np.float32(0)
    for value in witness:
        reference = np.float32(reference + np.float32(value * value))
    assert reference == np.float32(124928736)
    assert reference != np.float32(124928760), "witness must detect omitted norm correction"
    gradient = np.full((3, 32), 32768, dtype=np.float32)
    gradient[:, :27] = witness
    grad = wp.array(gradient, device=device)
    changed = wp.array([1, 1, 0], dtype=int, device=device)
    done = wp.array([False, True, False], dtype=bool, device=device)
    for stable_fast in (False, True):
        grad_dot = wp.array([124928760.0, 13.0, 17.0], dtype=float, device=device)
        wp.launch(norm, dim=3, inputs=[27, stable_fast, changed, done, grad], outputs=[grad_dot], device=device)
        np.testing.assert_array_equal(grad_dot.numpy(), [reference, 13, 17 if stable_fast else reference])
        np.testing.assert_array_equal(grad.numpy(), gradient)
