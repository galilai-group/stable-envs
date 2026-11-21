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
@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("name", ["OGBCube", "OGBScene"])
def test_manipulation_native_and_mjwarp(name, num_envs):
    pytest.importorskip("mujoco_warp")
    cls = getattr(se, name)
    check_backend_parity(name, ("native", "mjwarp"), num_envs=num_envs, exclude=cls.state_solver_caches)


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
    from stable_envs.envs.rocket.simulation.execution import require_numba_cuda

    try:
        require_numba_cuda()
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


@pytest.mark.family("ogbench")
@pytest.mark.parametrize("name", ["PointMazeMediumNavigate", "AntMazeTeleportNavigate", "HumanoidMazeMediumNavigate"])
def test_maze_native_and_mjwarp(name):
    pytest.importorskip("mujoco_warp")
    # MuJoCo Warp integrates PointMaze with Euler where OGBench uses RK4, and its contacts differ in AntMaze.
    steps = 3 if name.startswith("Humanoid") else 0
    exclude = se.registry.REGISTRY[name].state_solver_caches
    check_backend_parity(name, ("native", "mjwarp"), steps=steps, exclude=exclude)


@pytest.mark.family("fetch")
@pytest.mark.parametrize("name", ["FetchReach", "FetchPickAndPlace"])
def test_fetch_native_and_mjwarp(name):
    pytest.importorskip("mujoco_warp")
    # The MuJoCo Warp model of Fetch changes the contact settings (docs/environments/fetch.md), so only resets agree.
    exclude = se.registry.REGISTRY[name].state_solver_caches
    check_backend_parity(name, ("native", "mjwarp"), steps=0, size=32, exclude=exclude)


@pytest.mark.family("mshuman")
def test_mshuman_native_and_mjwarp():
    pytest.importorskip("mujoco_warp")
    exclude = se.MSHumanLocomotion.state_solver_caches
    check_backend_parity("MSHumanLocomotion", ("native", "mjwarp"), size=32, exclude=exclude)


@pytest.mark.family("roadster")
@pytest.mark.parametrize("name", ["RoadsterTrafficJam", "RoadsterCarHop"])
def test_roadster_native_and_mjwarp(name):
    pytest.importorskip("mujoco_warp")
    # Mocap positions are a few hundred metres, where float32 rounding is about 2e-5.
    exclude = se.registry.REGISTRY[name].state_solver_caches
    check_backend_parity(name, ("native", "mjwarp"), size=32, reset_atol=5e-5, exclude=exclude)
