"""Static assignments checked by mypy; this module is not executed."""

from stable_envs._adapters import SimulatorAdapter, StepPipeline
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


def local_adapters(
    classic: ClassicControlAdapter, fetch: FetchAdapter, dmc: DMCAdapter,
    maze: OGBMazeAdapter, manipulation: OGBManipulationAdapter,
) -> tuple[SimulatorAdapter, ...]:
    return classic, fetch, dmc, maze, manipulation


def process_adapters(
    classic: AcrobotProcessAdapter, fetch: FetchProcessAdapter, dmc: DMCProcessAdapter,
    maze: OGBMazeProcessAdapter, manipulation: OGBManipulationProcessAdapter,
) -> tuple[tuple[SimulatorAdapter, StepPipeline], ...]:
    return (classic, classic), (fetch, fetch), (dmc, dmc), (maze, maze), (manipulation, manipulation)
