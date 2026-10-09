# contrib: renderer-only ports (not released)

This directory holds environment families that re-render another project's environment from a state
vector. They are kept in the repository for in-lab use and are **not** part of the `stable-envs`
package:

- They are not checked against their upstream State. The released families (classic control, Fetch,
  PushT, Rocket, navigation, DM Control, OGBench, Craftax) are.
- Most of them cannot `step()`: they render states and raise `NotImplementedError` from `step` and
  `transition`. Only `MiniGridTopDown` (`minigrid.py`) and `Sokoban` (`sokoban.py`) carry a batched
  simulator.
- They are not in the wheel or the sdist (`MANIFEST.in` prunes `contrib/`, and setuptools packages only
  `stable_envs*`), and the package declares no extras for them.

| module | classes (examples) | requirements file |
|---|---|---|
| `carracing.py` | `CarRacing` | `requirements/box2d.txt` |
| `lunarlander.py` | `LunarLander` | `requirements/box2d.txt` |
| `crafter.py` | `Crafter` | `requirements/crafter.txt` |
| `griddly.py` | `GriddlySokoban`, … | `requirements/griddly.txt` |
| `highway.py` | `Highway`, `Merge`, `Parking`, … | `requirements/highway.txt` |
| `humanoidbench.py` | `HumanoidBenchH1Walk`, … | `requirements/humanoidbench.txt` |
| `jumanji.py` | `JumanjiMaze`, … | `requirements/jumanji.txt` |
| `maze2d.py` | `PointMazeMedium`, … | none; uses `gymnasium-robotics` and `mujoco` (the package's `robotics` extra) |
| `metaworld.py` | Meta-World tasks | `requirements/metaworld.txt` |
| `minigrid.py` | `MiniGridEmpty`, `BabyAIGoToLocal`, … | `requirements/minigrid.txt` |
| `minihack.py` | `MiniHack` | `requirements/minihack.txt` |
| `mujoco_locomotion.py` | `MuJoCoReacher`, `MuJoCoAnt`, … | `requirements/mujoco-locomotion.txt` |
| `overcooked.py` | `OvercookedCrampedRoom`, … | `requirements/overcooked.txt` |
| `pgx.py` | PGX board games | `requirements/pgx.txt` |
| `procgen.py` | `ProcgenMaze`, … | `requirements/procgen.txt` |
| `robocasa.py` | `RoboCasaCounterToSink`, … | none; see the module docstring and `examples/robocasa_visual/` |
| `sokoban.py` | `Sokoban` | `requirements/sokoban.txt` |

`pusht_torch/` is a batched torch renderer for the released PushT classes whose frames equal
`env.render(state)`; see `pusht_torch/README.md`.

Render engines used only by these families are in `render/`: `rendernet.py` (the RoboCasa learned
corrector, see `examples/rendernet/`) and `lightspec.py` (MJWarp light specialization for
HumanoidBench, Meta-World and RoboCasa).

## Use from a checkout

Run from the repository root (or put it on `PYTHONPATH`) with `stable-envs` installed, e.g.
`pip install -e .`:

```python
import contrib
import stable_envs as snv

errors = contrib.load()   # {module: "ImportError: ..."} for modules whose dependencies are missing
env = snv.make("MiniGridEmpty", num_envs=8)
```

`contrib.load()` imports every module and records failures under the module name, as
`snv.import_errors()` does for the package; `contrib.import_errors()` returns the same mapping later.
Importing a single module (`import contrib.minigrid`) registers only its classes. `stable_envs` itself
never imports `contrib`.

## Dependencies

```bash
pip install -r contrib/requirements/minigrid.txt
```

Each file holds the pins of the extra the family used to have. Notes carried over from those extras:

- `box2d.txt` builds Box2D from source; install `swig` first.
- `metaworld.txt`: install `metaworld` itself with `pip install --no-deps metaworld` (it pins
  `mujoco==3.3.0` on paper and runs on 3.11), then the file.
- `humanoidbench.txt`: `humanoid_bench` is not on PyPI; install it from git with `--no-deps`. The
  family registers only when `STABLE_ENVS_HUMANOIDBENCH=1` is set.
- `overcooked.txt`: `jaxmarl`'s pins conflict with the rest of the stack; use a dedicated interpreter.
- The visual-tier cells of `mujoco_locomotion.py` render through nvdiffrast, which is installed from
  source (see `docs/install.md`).

## Tests

```bash
python -m pytest contrib/tests -m "not gpu" -q   # CPU
python -m pytest contrib/tests/gpu                 # needs CUDA
```

Run them from the repository root; the root `conftest.py` and its safety plugin apply as they do to
`tests/`. `contrib/tests/conftest.py` calls `contrib.load()` before collection. Plain `pytest` runs
only `tests/`. CI runs the CPU tests in the `contrib` job of `.github/workflows/testing.yml`, with the
requirements files that install on a stock Ubuntu runner.
