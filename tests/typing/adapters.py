"""Static assignments checked by mypy; this module is not executed."""

from stable_envs._adapters import SimulatorAdapter, StepPipeline
from stable_envs._adapters.acrobot_process import AcrobotProcessAdapter
from stable_envs._adapters.dmcontrol_factory import DMCAdapter
from stable_envs._adapters.dmcontrol_process import DMCProcessAdapter
from stable_envs._adapters.fetch_pickandplace import FetchAdapter
from stable_envs._adapters.fetch_process import FetchProcessAdapter
from stable_envs._adapters.ogbench import OGBMazeAdapter
from stable_envs._adapters.ogbench_manipulation import OGBManipulationAdapter
from stable_envs._adapters.ogbench_manipulation_process import OGBManipulationProcessAdapter
from stable_envs._adapters.ogbench_maze_process import OGBMazeProcessAdapter
from stable_envs.envs._classic_semantics import ClassicControlAdapter


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
