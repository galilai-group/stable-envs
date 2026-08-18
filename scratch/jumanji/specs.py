"""Per-id adapters: what the grid IS, and how to render one through the OFFICIAL viewer.

Kept in scratch/ rather than in the env module while the gates are still being measured. Each spec is
the minimum an id needs for the per-pixel-LUT construction (probe 5) to run against it:

    rows, cols, n_values   the grid and its alphabet
    viewer(size)           the official viewer, in rgb_array mode, pinned to ``size`` px
    render(viewer, grid)   the OFFICIAL frame for a raw grid
    grid_of(state)         the reduction a real jumanji State goes through to become a grid
    base                   the grid the LUT is built against
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


def _pin(viewer, size: int, render_once):
    """Pin the viewer's figure to ``size`` px square (figsize is 10x10 in, so dpi = size/10).

    The first render creates the figure; ``_get_fig_ax`` then reuses it by NAME, so the dpi set after
    that first call is the one every later frame is drawn at. Measured rather than assumed: probe 5
    reports the frame shape and it is exactly (size, size, 3) at 200, 500 and 1000.
    """
    import matplotlib.pyplot as plt
    plt.close(viewer._name)
    render_once()
    plt.figure(viewer._name).set_dpi(size / 10.0)
    render_once()


def _rgb(a) -> np.ndarray:
    a = np.asarray(a)
    return a[..., :3] if a.shape[-1] == 4 else a


class Spec:
    id: str = ""
    rows: int = 0
    cols: int = 0
    n_values: int = 0
    value_names: tuple = ()

    @property
    def P(self) -> int:
        return self.rows * self.cols

    def viewer(self, size: int):
        raise NotImplementedError

    def render(self, viewer, grid) -> np.ndarray:
        raise NotImplementedError

    def grid_of(self, state) -> np.ndarray:
        raise NotImplementedError

    def base(self) -> np.ndarray:
        return np.zeros(self.P, np.int32)

    def sampler(self, rng) -> np.ndarray:
        return rng.integers(0, self.n_values, self.P)


# ------------------------------------------------------------------ MazeViewer family

class MazeSpec(Spec):
    """``MazeEnvViewer.render(state)`` IS ``MazeViewer.render(overlay(state))`` -- the overlay is a pure
    reduction of the State to a 4-symbol grid, so the grid is the render state."""

    id = "Maze-v0"
    rows = cols = 10
    n_values = 4
    value_names = ("empty", "wall", "agent", "target")

    def viewer(self, size):
        from jumanji.environments.routing.maze.viewer import MazeEnvViewer
        v = MazeEnvViewer("Maze", render_mode="rgb_array")
        g = self.base().reshape(self.rows, self.cols)
        _pin(v, size, lambda: self._raw(v, g))
        return v

    def _raw(self, v, grid):
        import jax.numpy as jnp
        from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer
        return _rgb(MazeViewer.render(v, jnp.asarray(grid)))

    def render(self, viewer, grid):
        return self._raw(viewer, np.asarray(grid).reshape(self.rows, self.cols))

    def grid_of(self, state):
        g = np.asarray(state.walls).astype(np.int32)
        g[int(state.agent_position.row), int(state.agent_position.col)] = 2
        g[int(state.target_position.row), int(state.target_position.col)] = 3
        return g.reshape(-1)

    def sampler(self, rng):
        g = (rng.random(self.P) < 0.35).astype(np.int32)
        g[rng.integers(0, self.P)] = 2
        g[rng.integers(0, self.P)] = 3
        return g


class CleanerSpec(Spec):
    """``CleanerViewer._create_grid_image`` paints COLORS[grid] and then overwrites every agent cell
    with AGENT_COLOR, so the picture is a 4-symbol grid: clean, wall, dirty, agent."""

    id = "Cleaner-v0"
    rows = cols = 10
    n_values = 4
    value_names = ("clean", "wall", "dirty", "agent")

    def viewer(self, size):
        from jumanji.environments.routing.cleaner.viewer import CleanerViewer
        v = CleanerViewer("Cleaner", render_mode="rgb_array")
        v._display = v._display_rgb_array
        _pin(v, size, lambda: _rgb(v.render(self._state(self.base()))))
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.routing.cleaner.types import State
        g = np.asarray(grid, np.int32).reshape(self.rows, self.cols)
        ag = np.argwhere(g == 3)
        if len(ag) == 0:
            ag = np.array([[0, 0]])
        locs = np.stack([ag[i % len(ag)] for i in range(3)])
        base = np.where(g == 3, 0, g).astype(np.int8)
        return State(grid=jnp.asarray(base), agents_locations=jnp.asarray(locs, jnp.int32),
                     action_mask=jnp.zeros((3, 4), bool), key=jnp.zeros(2, jnp.uint32),
                     step_count=jnp.int32(0))

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        g = np.asarray(state.grid).astype(np.int32)
        for r, c in np.asarray(state.agents_locations):
            g[int(r), int(c)] = 3
        return g.reshape(-1)

    def sampler(self, rng):
        g = rng.integers(0, 3, self.P)
        g[rng.integers(0, self.P, 3)] = 3
        return g


# ------------------------------------------------------------------ logic family

class RubiksCubeSpec(Spec):
    """Six 3x3 ``imshow`` panels through one ListedColormap: 54 stickers, 6 colours."""

    id = "RubiksCube-v0"
    rows, cols = 6, 9
    n_values = 6
    value_names = ("white", "yellow", "green", "blue", "red", "orange")

    def viewer(self, size):
        from jumanji.environments.logic.rubiks_cube.constants import DEFAULT_STICKER_COLORS
        from jumanji.environments.logic.rubiks_cube.viewer import RubiksCubeViewer
        v = RubiksCubeViewer(sticker_colors=DEFAULT_STICKER_COLORS, cube_size=3,
                             render_mode="rgb_array")
        _pin(v, size, lambda: _rgb(v.render(self._state(self.base()))))
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.rubiks_cube.types import State
        cube = np.asarray(grid, np.int8).reshape(6, 3, 3)
        return State(cube=jnp.asarray(cube), step_count=jnp.int32(0),
                     key=jnp.zeros(2, jnp.uint32))

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return np.asarray(state.cube).astype(np.int32).reshape(-1)

    def base(self):
        return np.repeat(np.arange(6, dtype=np.int32), 9)


class SudokuSpec(Spec):
    """81 ``Text`` artists, one per cell; ``-1`` draws the empty string. 10 symbols."""

    id = "Sudoku-v0"
    rows = cols = 9
    n_values = 10
    value_names = ("empty",) + tuple(str(d) for d in range(1, 10))

    def viewer(self, size):
        from jumanji.environments.logic.sudoku.viewer import SudokuViewer
        v = SudokuViewer("Sudoku", render_mode="rgb_array")
        v._display = v._display_rgb_array
        _pin(v, size, lambda: _rgb(v.render(self._state(self.base()))))
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.sudoku.types import State
        b = np.asarray(grid, np.int32).reshape(9, 9) - 1
        return State(board=jnp.asarray(b), action_mask=jnp.zeros((9, 9, 9), bool),
                     key=jnp.zeros(2, jnp.uint32))

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return (np.asarray(state.board).astype(np.int32) + 1).reshape(-1)


class MinesweeperSpec(Spec):
    id = "Minesweeper-v0"
    rows = cols = 10
    n_values = 10
    value_names = ("hidden",) + tuple(str(d) for d in range(9))

    def viewer(self, size):
        from jumanji.environments.logic.minesweeper.viewer import MinesweeperViewer
        v = MinesweeperViewer(num_rows=10, num_cols=10, render_mode="rgb_array")
        v._display = v._display_rgb_array
        _pin(v, size, lambda: _rgb(v.render(self._state(self.base()))))
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.minesweeper.types import State
        b = np.asarray(grid, np.int32).reshape(10, 10) - 1
        return State(board=jnp.asarray(b), step_count=jnp.int32(0),
                     flat_mine_locations=jnp.zeros(10, jnp.int32),
                     key=jnp.zeros(2, jnp.uint32))

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return (np.asarray(state.board).astype(np.int32) + 1).reshape(-1)


class SlidingTilePuzzleSpec(Spec):
    """``ax.imshow(state.puzzle, cmap=...)`` with NO vmin/vmax -- the colormap normalises to the data
    range, so a cell's colour is a function of the whole board. Measured here rather than assumed."""

    id = "SlidingTilePuzzle-v0"
    rows = cols = 5
    n_values = 25
    value_names = tuple(str(i) for i in range(25))

    def viewer(self, size):
        from jumanji.environments.logic.sliding_tile_puzzle.viewer import SlidingTilePuzzleViewer
        v = SlidingTilePuzzleViewer("SlidingTilePuzzle", render_mode="rgb_array")
        v._display = v._display_rgb_array
        _pin(v, size, lambda: _rgb(v.render(self._state(self.base()))))
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.sliding_tile_puzzle.types import State
        p = np.asarray(grid, np.int32).reshape(5, 5)
        z = np.argwhere(p == 0)
        pos = z[0] if len(z) else np.array([0, 0])
        return State(puzzle=jnp.asarray(p), empty_tile_position=jnp.asarray(pos, jnp.int32),
                     key=jnp.zeros(2, jnp.uint32), step_count=jnp.int32(0))

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return np.asarray(state.puzzle).astype(np.int32).reshape(-1)

    def base(self):
        return np.arange(25, dtype=np.int32)

    def sampler(self, rng):
        return rng.permutation(25).astype(np.int32)


class ConnectorSpec(Spec):
    """``ConnectorViewer.render(grid)`` takes a raw grid. 10 agents x (path, position, target) + empty.

    Paths are drawn at alpha 0.25 over a live canvas and every cell carries a black 1pt edge shared
    with its neighbours, so this is the id most likely to break separability -- which is why it is in
    the set rather than assumed out of it."""

    id = "Connector-v3"
    rows = cols = 10
    n_values = 31
    value_names = ("empty",) + tuple(
        f"agent{i}_{k}" for i in range(10) for k in ("path", "position", "target"))

    def viewer(self, size):
        from jumanji.environments.routing.connector.viewer import ConnectorViewer
        v = ConnectorViewer("Connector", num_agents=10, render_mode="rgb_array")
        v._display = v._display_rgb_array
        _pin(v, size, lambda: self.render(v, self.base()))
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp
        return _rgb(viewer.render(jnp.asarray(np.asarray(grid, np.int32).reshape(10, 10))))

    def grid_of(self, state):
        return np.asarray(state.grid).astype(np.int32).reshape(-1)


class SokobanSpec(Spec):
    """``BoxViewer`` draws one PNG sprite per cell from a 5-entry set, via ``grid_combine``."""

    id = "Sokoban-v0"
    rows = cols = 10
    n_values = 5
    value_names = ("empty", "wall", "target", "box", "agent")

    def viewer(self, size):
        import jumanji
        env = jumanji.make("Sokoban-v0")
        v = env._viewer
        v._display = v._display_rgb_array
        self._env = env
        _pin(v, size, lambda: self.render(v, self.base()))
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp
        g = jnp.asarray(np.asarray(grid, np.uint8).reshape(10, 10))
        # BoxViewer.render takes the state and calls self._grid_combine(variable, fixed); the combined
        # grid is what _draw_grid consumes, so it is handed one directly.
        return _rgb(viewer.render(_Combined(g)))

    def grid_of(self, state):
        return np.asarray(
            self._env.grid_combine(state.variable_grid, state.fixed_grid)).astype(np.int32).reshape(-1)


class _Combined:
    """Stands in for the Sokoban State: ``BoxViewer._add_grid_image`` only calls ``grid_combine``."""

    def __init__(self, g):
        self.variable_grid = g
        self.fixed_grid = g


ALL = [MazeSpec(), CleanerSpec(), RubiksCubeSpec(), SudokuSpec(), MinesweeperSpec(),
       SlidingTilePuzzleSpec(), ConnectorSpec(), SokobanSpec()]
BY_ID = {s.id: s for s in ALL}
