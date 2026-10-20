"""Static assignments checked by mypy; this module is not executed."""

from stable_envs._execution.protocol import SimulatorAdapter, StepPipeline
from stable_envs.envs.classic_control.simulation.local import ClassicControlAdapter
from stable_envs.envs.classic_control.simulation.process import AcrobotProcessAdapter
from stable_envs.envs.dmcontrol.simulation.factory import DMCAdapter
from stable_envs.envs.dmcontrol.simulation.process import DMCProcessAdapter
from stable_envs.envs.fetch.simulation.adapter import FetchAdapter
from stable_envs.envs.fetch.simulation.process import FetchProcessAdapter
from stable_envs.envs.object_permanence.simulation.g34 import OcclusionBallAdapter
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter
from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter
from stable_envs.envs.ogbench.simulation.maze_process import OGBMazeProcessAdapter
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter
from stable_envs.envs.rocket.simulation.adapter import RocketAdapter


def local_adapters(
    classic: ClassicControlAdapter,
    fetch: FetchAdapter,
    dmc: DMCAdapter,
    maze: OGBMazeAdapter,
    manipulation: OGBManipulationAdapter,
    pusht: PushTReplayAdapter,
    rocket: RocketAdapter,
    occlusion: OcclusionBallAdapter,
) -> tuple[SimulatorAdapter, ...]:
    return classic, fetch, dmc, maze, manipulation, pusht, rocket, occlusion


def process_adapters(
    classic: AcrobotProcessAdapter,
    fetch: FetchProcessAdapter,
    dmc: DMCProcessAdapter,
    maze: OGBMazeProcessAdapter,
    manipulation: OGBManipulationProcessAdapter,
) -> tuple[tuple[SimulatorAdapter, StepPipeline], ...]:
    return (classic, classic), (fetch, fetch), (dmc, dmc), (maze, maze), (manipulation, manipulation)
