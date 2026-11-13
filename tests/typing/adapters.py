"""Static assignments checked by mypy; this module is not executed."""

from stable_envs._execution.protocol import SimulatorAdapter, StepPipeline
from stable_envs.envs.classic_control.simulation.batched import BatchedClassicAdapter
from stable_envs.envs.classic_control.simulation.device import DeviceClassicAdapter
from stable_envs.envs.classic_control.simulation.local import ClassicControlAdapter
from stable_envs.envs.classic_control.simulation.process import AcrobotProcessAdapter
from stable_envs.envs.dmcontrol.simulation.batch import DMCBatchAdapter
from stable_envs.envs.dmcontrol.simulation.factory import DMCAdapter
from stable_envs.envs.dmcontrol.simulation.mjwarp import DMCWarpAdapter
from stable_envs.envs.fetch.simulation.adapter import FetchAdapter
from stable_envs.envs.fetch.simulation.process import FetchProcessAdapter
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter
from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter
from stable_envs.envs.ogbench.simulation.maze_process import OGBMazeProcessAdapter
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter
from stable_envs.envs.rocket.simulation.adapter import RocketAdapter
from stable_envs.envs.rocket.simulation.numba_batch import RocketNumbaAdapter
from stable_envs.envs.rocket.simulation.numba_cuda_batch import RocketCudaAdapter


def local_adapters(
    classic: ClassicControlAdapter,
    batched: BatchedClassicAdapter,
    device: DeviceClassicAdapter,
    fetch: FetchAdapter,
    dmc: DMCAdapter,
    dmc_batch: DMCBatchAdapter,
    dmc_warp: DMCWarpAdapter,
    maze: OGBMazeAdapter,
    manipulation: OGBManipulationAdapter,
    pusht: PushTReplayAdapter,
    rocket: RocketAdapter,
    rocket_numba: RocketNumbaAdapter,
    rocket_cuda: RocketCudaAdapter,
) -> tuple[SimulatorAdapter, ...]:
    return (
        classic,
        batched,
        device,
        fetch,
        dmc,
        dmc_batch,
        dmc_warp,
        maze,
        manipulation,
        pusht,
        rocket,
        rocket_numba,
        rocket_cuda,
    )


def process_adapters(
    classic: AcrobotProcessAdapter,
    fetch: FetchProcessAdapter,
    maze: OGBMazeProcessAdapter,
    manipulation: OGBManipulationProcessAdapter,
) -> tuple[tuple[SimulatorAdapter, StepPipeline], ...]:
    return (classic, classic), (fetch, fetch), (maze, maze), (manipulation, manipulation)
