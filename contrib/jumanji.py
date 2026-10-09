from __future__ import annotations

import importlib.metadata as _md
import importlib.util as _ilu
import os

# Select headless plotting and avoid JAX reserving memory needed by Torch.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Detect optional packages without importing JAX or changing plotting state.
_VENV = "/mnt/vast/home/randall.balestriero/venvs/se-jumanji"
_missing = [m for m in ("jumanji", "jax", "matplotlib") if _ilu.find_spec(m) is None]
if _missing:  # pragma: no cover
    raise ImportError(
        f"the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so "
        f"it cannot render without the reference package: missing {', '.join(_missing)}. The envs were "
        f"measured in {_VENV}, built with a CLEAN `python -m venv` — a --system-site-packages venv "
        f"breaks jax's CUDA plugin discovery."
    )
try:
    _JVER = _md.version("jumanji")
except Exception:  # pragma: no cover
    _JVER = "unknown"

import numpy as np
import torch

from stable_envs.env import Env
from stable_envs.registry import register
from stable_envs.spaces import Box, Dict, RGBBox

# Built atlases persist under this root, keyed by source ID and image size.
_CACHE_ROOT = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
#: In-process atlas cache, keyed (id, size, device).
_ATLAS: dict = {}
# Viewers with the same name share a global figure; cache one per atlas ID.
_VIEWERS: dict = {}
#: Reset-distribution state pools, keyed by id. Built once per process from the official env.
_POOL: dict = {}
#: Sentinel label meaning "this pixel's colour is not a flat palette colour; keep the official one".
RAW = 255
# Bound gather scratch memory when rendering large batches.
_SCRATCH_BUDGET = 256 << 20


def _rgb(a) -> np.ndarray:
    a = np.asarray(a)
    return a[..., :3] if a.shape[-1] == 4 else a


def _official_frame(spec: _Spec, size: int, grid) -> np.ndarray:
    """Draw through the shared official viewer at the requested pixel size.

    Keep the viewer name unchanged: some viewers draw it in the image. Change
    figure DPI and settle the layout before reading a frame at a new size.
    """
    import matplotlib.pyplot as plt

    v = _viewer(spec)
    want = float(size) / 10.0
    if getattr(v, "_se_dpi", None) != want:
        spec.render(v, spec.base())  # ensure the figure exists
        plt.figure(v._name).set_dpi(want)
        spec.render(v, spec.base())  # let the layout settle at the new dpi
        v._se_dpi = want
    out = spec.render(v, grid)
    if out.shape != (int(size), int(size), 3):
        raise RuntimeError(
            f"{spec.env_id}: the official viewer produced {out.shape}, not "
            f"{(int(size), int(size), 3)} — the figure dpi is not being honoured, and every number "
            f"measured against it would be a resample rather than the reference"
        )
    return out


class _Spec:
    env_id: str = ""
    cls_name: str = ""
    ds_name: str = ""
    rows: int = 0
    cols: int = 0
    n_values: int = 0
    value_names: tuple = ()
    doc: str = ""

    # Resolutions with validated per-cell separability.
    verified_sizes: tuple = (200, 400)

    # Reset-only variants must share the viewer to avoid competing for its named figure.
    atlas_id: str = ""

    @property
    def key_id(self) -> str:
        return self.atlas_id or self.env_id

    @property
    def P(self) -> int:
        return self.rows * self.cols

    def viewer(self):
        raise NotImplementedError

    def render(self, viewer, grid) -> np.ndarray:
        raise NotImplementedError

    def grid_of(self, state) -> np.ndarray:
        raise NotImplementedError

    def base(self) -> np.ndarray:
        return np.zeros(self.P, np.int32)

    def cell_names(self) -> tuple:
        return tuple(f"cell_r{r}c{c}" for r in range(self.rows) for c in range(self.cols))


class _Maze(_Spec):
    env_id, cls_name, ds_name = "Maze-v0", "JumanjiMaze", "jumanji-maze"
    rows = cols = 10
    n_values = 4
    value_names = ("empty", "wall", "agent", "target")
    doc = (
        "Jumanji's 10x10 Maze. The render state is the OVERLAID grid — "
        "``MazeEnvViewer.render`` is ``MazeViewer.render(overlay(state))``, so the agent and the "
        "target are cell symbols rather than coordinates."
    )

    def viewer(self):
        from jumanji.environments.routing.maze.viewer import MazeEnvViewer

        v = MazeEnvViewer("Maze", render_mode="rgb_array")
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp
        from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer

        g = np.asarray(grid, np.int32).reshape(self.rows, self.cols)
        return _rgb(MazeViewer.render(viewer, jnp.asarray(g)))

    def grid_of(self, state):
        g = np.asarray(state.walls).astype(np.int32)
        g[int(state.agent_position.row), int(state.agent_position.col)] = 2
        g[int(state.target_position.row), int(state.target_position.col)] = 3
        return g.reshape(-1)


class _RubiksCube(_Spec):
    env_id, cls_name, ds_name = "RubiksCube-v0", "JumanjiRubiksCube", "jumanji-rubiks-cube"
    rows, cols = 6, 9
    n_values = 6
    value_names = ("white", "yellow", "green", "blue", "red", "orange")
    doc = (
        "Jumanji's 3x3x3 Rubik's Cube, drawn as six 3x3 ``imshow`` panels through one "
        "``ListedColormap``. The state is the 54 stickers."
    )

    def viewer(self):
        from jumanji.environments.logic.rubiks_cube.constants import DEFAULT_STICKER_COLORS
        from jumanji.environments.logic.rubiks_cube.viewer import RubiksCubeViewer

        v = RubiksCubeViewer(sticker_colors=DEFAULT_STICKER_COLORS, cube_size=3, render_mode="rgb_array")
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.rubiks_cube.types import State

        return State(
            cube=jnp.asarray(np.asarray(grid, np.int8).reshape(6, 3, 3)),
            step_count=jnp.int32(0),
            key=jnp.zeros(2, jnp.uint32),
        )

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return np.asarray(state.cube).astype(np.int32).reshape(-1)

    def base(self):
        return np.repeat(np.arange(6, dtype=np.int32), 9)

    def cell_names(self):
        faces = ("up", "front", "right", "back", "left", "down")
        return tuple(f"{f}_r{r}c{c}" for f in faces for r in range(3) for c in range(3))


class _Sudoku(_Spec):
    env_id, cls_name, ds_name = "Sudoku-v0", "JumanjiSudoku", "jumanji-sudoku"
    rows = cols = 9
    n_values = 10
    value_names = ("empty",) + tuple(str(d) for d in range(1, 10))
    doc = (
        "Jumanji's Sudoku, drawn as 81 ``Text`` artists and nothing else — the viewer's "
        "``_draw_board`` only turns the axis off, so 2.4% of the frame carries the whole board."
    )

    def viewer(self):
        from jumanji.environments.logic.sudoku.viewer import SudokuViewer

        v = SudokuViewer("Sudoku", render_mode="rgb_array")
        v._display = v._display_rgb_array
        return v

    def _state(self, grid):
        import jax.numpy as jnp
        from jumanji.environments.logic.sudoku.types import State

        return State(
            board=jnp.asarray(np.asarray(grid, np.int32).reshape(9, 9) - 1),
            action_mask=jnp.zeros((9, 9, 9), bool),
            key=jnp.zeros(2, jnp.uint32),
        )

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return (np.asarray(state.board).astype(np.int32) + 1).reshape(-1)


class _Connector(_Spec):
    env_id, cls_name, ds_name = "Connector-v3", "JumanjiConnector", "jumanji-connector"
    rows = cols = 10
    n_values = 31
    value_names = ("empty",) + tuple(f"agent{i}_{k}" for i in range(10) for k in ("path", "position", "target"))
    doc = (
        "Jumanji's Connector, 10 agents on a 10x10 grid. Its cells are alpha-0.25 rectangles with "
        "shared black borders drawn against a live canvas, and at 200 px they separate exactly "
        "anyway — but only at 200 px."
    )
    # At higher DPI, antialiased borders blend neighboring cells and break separability.
    verified_sizes = (200,)

    def viewer(self):
        from jumanji.environments.routing.connector.viewer import ConnectorViewer

        v = ConnectorViewer("Connector", num_agents=10, render_mode="rgb_array")
        v._display = v._display_rgb_array
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp

        return _rgb(viewer.render(jnp.asarray(np.asarray(grid, np.int32).reshape(10, 10))))

    def grid_of(self, state):
        return np.asarray(state.grid).astype(np.int32).reshape(-1)


class _Sokoban(_Spec):
    env_id, cls_name, ds_name = "Sokoban-v0", "JumanjiSokoban", "jumanji-sokoban"
    rows = cols = 10
    n_values = 5
    value_names = ("empty", "wall", "target", "box", "agent")
    doc = (
        "Jumanji's Sokoban (Boxoban levels), drawn as one PNG sprite per cell. The state is the "
        "COMBINED grid the viewer consumes, i.e. ``grid_combine(variable, fixed)``."
    )

    def viewer(self):
        import jumanji

        env = jumanji.make("Sokoban-v0")
        v = env._viewer
        v._display = v._display_rgb_array
        self._env = env
        return v

    def render(self, viewer, grid):
        import jax.numpy as jnp

        g = jnp.asarray(np.asarray(grid, np.uint8).reshape(10, 10))
        return _rgb(viewer.render(_CombinedGrid(g)))

    def grid_of(self, state):
        env = getattr(self, "_env", None)
        if env is None:
            import jumanji

            env = self._env = jumanji.make("Sokoban-v0")
        return np.asarray(env.grid_combine(state.variable_grid, state.fixed_grid)).astype(np.int32).reshape(-1)


class _CombinedGrid:
    """Supply an already combined Sokoban grid; grid_combine(g, g) preserves it."""

    def __init__(self, g):
        self.variable_grid = g
        self.fixed_grid = g


class _Minesweeper(_Spec):
    env_id, cls_name, ds_name = "Minesweeper-v0", "JumanjiMinesweeper", "jumanji-minesweeper"
    rows = cols = 10
    n_values = 10
    value_names = ("hidden",) + tuple(str(d) for d in range(9))
    doc = (
        "Jumanji's 10x10 Minesweeper. ``-1`` is an unexplored square and ``0-8`` the neighbouring "
        "mine count; ``flat_mine_locations`` reaches no pixel and is not in the state."
    )

    def viewer(self):
        from jumanji.environments.logic.minesweeper.viewer import MinesweeperViewer

        v = MinesweeperViewer(num_rows=10, num_cols=10, render_mode="rgb_array")
        v._display = v._display_rgb_array
        return v

    def _state(self, grid, mines=None):
        import jax.numpy as jnp
        from jumanji.environments.logic.minesweeper.types import State

        return State(
            board=jnp.asarray(np.asarray(grid, np.int32).reshape(10, 10) - 1),
            step_count=jnp.int32(0),
            flat_mine_locations=jnp.asarray(np.zeros(10, np.int32) if mines is None else mines, jnp.int32),
            key=jnp.zeros(2, jnp.uint32),
        )

    def render(self, viewer, grid):
        return _rgb(viewer.render(self._state(grid)))

    def grid_of(self, state):
        return (np.asarray(state.board).astype(np.int32) + 1).reshape(-1)


class _RubiksCubePartlyScrambled(_RubiksCube):
    """Use the short-scramble reset distribution with the standard cube atlas."""

    env_id = "RubiksCube-partly-scrambled-v0"
    cls_name = "JumanjiRubiksCubePartlyScrambled"
    ds_name = "jumanji-rubiks-cube-partly-scrambled"
    atlas_id = "RubiksCube-v0"


class _SudokuVeryEasy(_Sudoku):
    """Use the easier puzzle generator with the standard Sudoku atlas."""

    env_id = "Sudoku-very-easy-v0"
    cls_name = "JumanjiSudokuVeryEasy"
    ds_name = "jumanji-sudoku-very-easy"
    atlas_id = "Sudoku-v0"


SPECS = [
    _Maze(),
    _RubiksCube(),
    _Sudoku(),
    _Connector(),
    _Sokoban(),
    _Minesweeper(),
    _RubiksCubePartlyScrambled(),
    _SudokuVeryEasy(),
]


def _viewer(spec: _Spec):
    """Cache one viewer per atlas ID; sizes share its named Matplotlib figure."""
    key = spec.key_id
    hit = _VIEWERS.get(key)
    if hit is None:
        hit = _VIEWERS[key] = spec.viewer()
    return hit


def _atlas_path(env_id: str, size: int) -> str:
    return os.path.join(_CACHE_ROOT, "jumanji", f"{env_id.replace('-', '_')}_s{size}.npz")


def _build_atlas(spec: _Spec, size: int) -> dict:
    """Build a background plate and cell/symbol lookup from official frames.

    Reject pixels changed by multiple cells: a per-cell lookup cannot represent
    their interactions.
    """
    base = spec.base()
    f0 = _official_frame(spec, size, base)
    P, V = spec.P, spec.n_values
    frames = np.empty((P, V, size, size, 3), np.uint8)
    for p in range(P):
        for val in range(V):
            g = base.copy()
            g[p] = val
            frames[p, val] = _official_frame(spec, size, g)

    changed = np.zeros((P, size, size), bool)
    for p in range(P):
        ref = frames[p, base[p]].astype(np.int16)
        changed[p] = (frames[p].astype(np.int16) - ref != 0).any(axis=(0, 3))
    n_own = changed.sum(0)
    conflicts = int((n_own > 1).sum())
    if conflicts:
        raise RuntimeError(
            f"{spec.env_id}: {conflicts} pixels are moved by MORE THAN ONE cell, so the frame is not a "
            f"per-cell lookup and this construction is unsound for it."
        )
    owner = np.where(n_own == 1, changed.argmax(0), -1).astype(np.int32)
    ys, xs = np.nonzero(owner >= 0)
    lut = frames[owner[ys, xs], :, ys, xs]  # (K, V, 3) uint8

    # Recolor only each symbol's modal color; preserve blended fringes from the source.
    palette = np.zeros((V, 3), np.uint8)
    label = np.full(lut.shape[:2], RAW, np.uint8)
    for val in range(V):
        cols, counts = np.unique(lut[:, val], axis=0, return_counts=True)
        palette[val] = cols[counts.argmax()]
        label[:, val] = np.where((lut[:, val] == palette[val]).all(-1), val, RAW)
    return {
        "plate": f0,
        "ys": ys.astype(np.int32),
        "xs": xs.astype(np.int32),
        "cells": owner[ys, xs].astype(np.int32),
        "lut": lut,
        "label": label,
        "palette": palette,
        "meta": np.array([P, V, size, len(ys)], np.int64),
    }


def _background(plate: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> tuple:
    """Find the dominant color and matching mask among pixels not owned by cells."""
    H, W, _ = plate.shape
    owned = np.zeros(H * W, bool)
    owned[ys.astype(np.int64) * W + xs.astype(np.int64)] = True
    unowned = ~owned.reshape(H, W)
    cols, counts = np.unique(plate[unowned], axis=0, return_counts=True)
    bg = cols[counts.argmax()].astype(np.uint8)
    return bg, (plate == bg).all(-1) & unowned


def _load_atlas(spec: _Spec, size: int, device) -> dict:
    key = (spec.key_id, int(size), str(device))
    hit = _ATLAS.get(key)
    if hit is not None:
        return hit
    path = _atlas_path(spec.key_id, size)
    data = None
    if os.path.exists(path):
        try:
            data = {k: v for k, v in np.load(path).items()}
        except Exception:  # noqa: BLE001  partial/corrupt cache
            data = None
    if data is None:
        data = _build_atlas(spec, size)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:  # a handle, not a name: savez appends .npz to names
            np.savez_compressed(fh, **data)
        os.replace(tmp, path)

    dev = torch.device(device)
    t = lambda a, d=None: torch.as_tensor(a, device=dev, dtype=d)
    K = int(data["meta"][3])
    bg, bg_mask = _background(data["plate"], data["ys"], data["xs"])
    out = {
        "plate": t(data["plate"]),  # (H,W,3) uint8
        "pix": t(data["ys"].astype(np.int64)) * int(size) + t(data["xs"].astype(np.int64)),
        "cells": t(data["cells"].astype(np.int64)),  # (K,)
        "lut": t(data["lut"]).reshape(-1, 3),  # (K*V, 3) uint8
        "label": t(data["label"]).reshape(-1),  # (K*V,) uint8
        "palette": data["palette"],
        "bg": bg,
        "K": K,
        "V": int(data["meta"][1]),
    }
    # Precompute pixel offsets into the flattened symbol lookup.
    out["kbase"] = torch.arange(K, device=dev, dtype=torch.long) * out["V"]
    out["bg_mask"] = t(bg_mask)  # (H,W) bool
    _ATLAS[key] = out
    return out


class JumanjiGrid(Env):
    """Render cell grids using an atlas sampled from the official Jumanji viewer.

    Atlas construction requires Jumanji and Matplotlib; rendering gathers cached
    colors on the chosen Torch device. Each pixel must depend on at most one
    cell, so supported resolutions are restricted per task.
    """

    family = "jumanji"

    SPEC: _Spec = None
    default_size = 200
    parity_interior_min = 1.0

    def __init__(
        self,
        num_envs: int = 1,
        size: int | None = None,
        supersample: int = 1,
        device: str = "cuda",
        backend: str = "auto",
        seed: int = 0,
    ):
        spec = self.SPEC
        self.state_dim = spec.P
        self.state_names = spec.cell_names()
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        self._check_size(self.size)
        self._a = _load_atlas(spec, self.size, self.device)

    def _check_size(self, size: int) -> None:
        if int(size) not in self.SPEC.verified_sizes:
            raise ValueError(
                f"{type(self).__name__} renders at {self.SPEC.verified_sizes} px, not {int(size)}. The "
                f"atlas is exact only where the official viewer's cells provably separate, and that was "
                f"measured per size rather than assumed — at an unverified size there is no atlas that "
                f"is right, only one that would be silently wrong."
            )

    def make_variation_space(self) -> Dict:
        """Expose the background color and one ``(V, 3)`` palette leaf.

        Palette changes affect each symbol's modal color. Other colors, including
        antialiasing fringes, retain their source values.
        """
        spec = self.SPEC
        a = _load_atlas(spec, self.size, self.device)
        # RGBBox fixes shape=(3,); the palette needs (V, 3).
        return Dict(
            {
                "palette": Box(
                    low=0, high=255, shape=(spec.n_values, 3), dtype=np.uint8, init_value=a["palette"].astype(np.uint8)
                ),
                "background": RGBBox(init_value=a["bg"].astype(np.uint8)),
            }
        )

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = int(s.shape[0])
        size = int(size or self.size)
        if size == self.size:
            a = self._a
        else:
            self._check_size(size)
            a = _load_atlas(self.SPEC, size, self.device)
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        pal = sty["palette"].reshape(B, a["V"], 3).round().clamp(0, 255).to(torch.uint8)
        bg = sty["background"].reshape(B, 1, 3).round().clamp(0, 255).to(torch.uint8)
        buf.copy_(a["plate"].unsqueeze(0).expand(B, size, size, 3))
        buf[:, a["bg_mask"]] = bg  # (B,1,3) broadcasts over the masked pixels

        # Wrap symbols consistently with reference_render and chunk the gather to bound scratch memory.
        flatbuf = buf.view(B, -1, 3)
        rows = max(1, int(_SCRATCH_BUDGET // max(1, a["K"] * 8)))
        codes = s.round().long() % a["V"]  # (B, P)
        for lo in range(0, B, rows):
            hi = min(lo + rows, B)
            sym = codes[lo:hi][:, a["cells"]]  # (n, K)
            flat = a["kbase"].unsqueeze(0) + sym
            col = a["lut"][flat]  # (n, K, 3)
            lab = a["label"][flat]  # (n, K) uint8
            # Clamp the RAW sentinel to a valid gather index; its gathered color is discarded.
            idx = lab.long().clamp_(max=a["V"] - 1).unsqueeze(-1).expand(-1, -1, 3)
            recol = torch.gather(pal[lo:hi], 1, idx)
            flatbuf[lo:hi, a["pix"]] = torch.where((lab == RAW).unsqueeze(-1), col, recol)
        return buf.view(*lead, size, size, 3) if lead else buf

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample render grids from a cached, deterministically seeded source rollout pool."""
        rng = rng or self.rng
        pool = _state_pool(self.SPEC)
        return pool[rng.integers(0, len(pool), int(batch))].astype(np.float32)

    @classmethod
    def reference_env(cls, size: int = 200):
        """Return a source spec/size handle; official viewers are shared and remain open."""
        return (cls.SPEC, int(size))

    @classmethod
    def reference_render(cls, ref, states) -> np.ndarray:
        """Draw source frames for rounded grid symbols, wrapping values modulo the alphabet."""
        spec, size = ref
        out = []
        for row in np.asarray(states, np.float32):
            g = np.rint(row).astype(np.int32) % spec.n_values
            out.append(np.ascontiguousarray(_official_frame(spec, size, g)))
        return np.stack(out)


def _state_pool(spec: _Spec, n: int = 256) -> np.ndarray:
    """Cache ``(n, P)`` grids from seeded source resets and short random rollouts."""
    hit = _POOL.get(spec.env_id)
    if hit is not None:
        return hit
    import jax
    import jumanji

    env = jumanji.make(spec.env_id)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = np.random.default_rng(0)
    out = np.empty((n, spec.P), np.float32)
    for i in range(n):
        key = jax.random.PRNGKey(int(i))
        state, _ = reset(key)
        for _ in range(int(rng.integers(0, 12))):
            key, sub = jax.random.split(key)
            try:
                mask = getattr(state, "action_mask", None)
                if mask is not None:
                    m = np.asarray(mask)
                    idx = np.flatnonzero(m.reshape(-1))
                    if not len(idx):
                        break
                    a = np.asarray(np.unravel_index(int(rng.choice(idx)), m.shape)).squeeze()
                    a = jax.numpy.asarray(a)
                else:
                    spc = env.action_spec
                    nv = np.asarray(getattr(spc, "num_values", 4))
                    a = jax.numpy.asarray(rng.integers(0, np.maximum(nv, 1)))
                state, _ = step(state, a)
            except Exception:  # noqa: BLE001  terminal / bad action
                break
        out[i] = spec.grid_of(state)
    _POOL[spec.env_id] = out
    return out


def _make_classes() -> dict:
    made = {}
    for spec in SPECS:
        made[spec.cls_name] = register(
            type(
                spec.cls_name,
                (JumanjiGrid,),
                dict(
                    SPEC=spec,
                    name=spec.ds_name,
                    reference_id=spec.env_id,
                    __doc__=f"{spec.doc}\n\nOfficial id ``{spec.env_id}`` (jumanji {_JVER}); the frame is a "
                    f"per-pixel lookup over {spec.P} cells and {spec.n_values} symbols.",
                ),
            )
        )
    return made


globals().update(_make_classes())
