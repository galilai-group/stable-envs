"""The small-batch and large-batch CUDA backends of a class start from the same reset State for a seed."""

import pytest
import torch

import stable_envs as se
from tests.helpers.backend_parity import check_backend_parity

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.family("dmc")
def test_reacher_native_and_mjwarp():
    pytest.importorskip("mujoco_warp")
    check_backend_parity("DMCReacher", ("native", "mjwarp"), size=32, exclude=se.DMCReacher.state_solver_caches)


@pytest.mark.family("ogbench")
def test_cube_native_and_mjwarp():
    pytest.importorskip("mujoco_warp")
    check_backend_parity("OGBCube", ("native", "mjwarp"), num_envs=2, exclude=se.OGBCube.state_solver_caches)


@pytest.mark.family("pusht")
def test_pusht_numba_and_triton():
    from stable_envs.envs.pusht import raster

    if not raster.HAVE_TRITON:
        pytest.skip("Triton required")
    check_backend_parity("PushT", ("numba", "triton"), size=32, exclude=se.PushT.state_solver_caches)


@pytest.mark.family("navigation")
def test_tworoom_numba_and_jax():
    pytest.importorskip("jax")
    check_backend_parity("TwoRoom", ("numba", "jax", "torch"), reset_atol=0, step_atol=0)


@pytest.mark.family("classic")
@pytest.mark.parametrize("name", ["CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"])
def test_classic_gym_numpy_and_torch(name):
    pytest.importorskip("pygame")
    check_backend_parity(name, ("gym", "numpy", "torch"), seeds=(0, 7, 2**32 - 2), reset_atol=0, step_atol=1e-7)


@pytest.mark.family("rocket")
def test_rocket_numba_and_numba_cuda():
    pytest.importorskip("PyFlyt")
    from stable_envs.envs.rocket.simulation.execution import require_nvvm

    try:
        require_nvvm()
    except RuntimeError as e:
        pytest.skip(str(e))
    check_backend_parity("PyFlytRocketLanding", ("numba", "numba_cuda"), reset_atol=1e-9)


@pytest.mark.family("miniworld")
@pytest.mark.parametrize("preset", ["flowm", "upstream"])
def test_miniworld_numba_and_triton(preset):
    from stable_envs.envs.miniworld.simulation.adapter import configure_pyglet

    configure_pyglet()
    pytest.importorskip("miniworld.envs.moving_blockworld")
    check_backend_parity(
        "MiniWorldMovingBlockWorld", ("numba", "triton"), size=32, preset=preset, reset_atol=1e-12, step_atol=1e-9
    )
