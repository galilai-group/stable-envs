"""Static assignments checked by mypy; this module is not executed."""

from stable_envs._adapters.classic.local import ClassicControlAdapter
from stable_envs._adapters.classic.process import AcrobotProcessAdapter
from stable_envs._adapters.dmc.factory import DMCAdapter
from stable_envs._adapters.dmc.process import DMCProcessAdapter
from stable_envs._adapters.fetch.adapter import FetchAdapter
from stable_envs._adapters.fetch.process import FetchProcessAdapter
from stable_envs._adapters.ogbench.manipulation import OGBManipulationAdapter
from stable_envs._adapters.ogbench.manipulation_process import OGBManipulationProcessAdapter
from stable_envs._adapters.ogbench.maze import OGBMazeAdapter
from stable_envs._adapters.ogbench.maze_process import OGBMazeProcessAdapter
from stable_envs._adapters.pusht.replay_batch import PushTReplayAdapter
from stable_envs._adapters.rocket.adapter import RocketAdapter
from stable_envs._execution.protocol import SimulatorAdapter, StepPipeline
from stable_envs.envs.craftax.simulation import CraftaxAdapter


def local_adapters(
    classic: ClassicControlAdapter, fetch: FetchAdapter, dmc: DMCAdapter,
    maze: OGBMazeAdapter, manipulation: OGBManipulationAdapter,
    pusht: PushTReplayAdapter, rocket: RocketAdapter, craftax: CraftaxAdapter,
) -> tuple[SimulatorAdapter, ...]:
    return classic, fetch, dmc, maze, manipulation, pusht, rocket, craftax


def process_adapters(
    classic: AcrobotProcessAdapter, fetch: FetchProcessAdapter, dmc: DMCProcessAdapter,
    maze: OGBMazeProcessAdapter, manipulation: OGBManipulationProcessAdapter,
) -> tuple[tuple[SimulatorAdapter, StepPipeline], ...]:
    return (classic, classic), (fetch, fetch), (dmc, dmc), (maze, maze), (manipulation, manipulation)
