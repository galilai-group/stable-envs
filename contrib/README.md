# contrib: renderer-only ports (not released)

This directory holds families that re-render another project's environment from a state vector.
They are **not** part of the `stable-envs` package:

- They are not checked against their upstream State.
- Most cannot `step()`. Only `MiniGridTopDown` (`minigrid.py`) and `Sokoban` (`sokoban.py`) have a
  batched simulator.
- They are not in the wheel or the sdist, and the package has no extras for them.

| module | classes (examples) | requirements file |
|---|---|---|
| `carracing.py` | `CarRacing` | `requirements/box2d.txt` |
| `lunarlander.py` | `LunarLander` | `requirements/box2d.txt` |
| `crafter.py` | `Crafter` | `requirements/crafter.txt` |
| `griddly.py` | `GriddlySokoban`, … | `requirements/griddly.txt` |
| `highway.py` | `Highway`, `Merge`, `Parking`, … | `requirements/highway.txt` |
| `humanoidbench.py` | `HumanoidBenchH1Walk`, … | `requirements/humanoidbench.txt` |
| `jumanji.py` | `JumanjiMaze`, … | `requirements/jumanji.txt` |
| `maze2d.py` | `PointMazeMedium`, … | none; uses `gymnasium-robotics` and `mujoco` (the `robotics` extra) |
| `metaworld.py` | Meta-World tasks | `requirements/metaworld.txt` |
| `minigrid.py` | `MiniGridEmpty`, `BabyAIGoToLocal`, … | `requirements/minigrid.txt` |
| `minihack.py` | `MiniHack` | `requirements/minihack.txt` |
| `mujoco_locomotion.py` | `MuJoCoReacher`, `MuJoCoAnt`, … | `requirements/mujoco-locomotion.txt` |
| `overcooked.py` | `OvercookedCrampedRoom`, … | `requirements/overcooked.txt` |
| `pgx.py` | PGX board games | `requirements/pgx.txt` |
| `procgen.py` | `ProcgenMaze`, … | `requirements/procgen.txt` |
| `robocasa.py` | `RoboCasaCounterToSink`, … | none; see the module docstring and `robocasa_dump.py` |
| `sokoban.py` | `Sokoban` | `requirements/sokoban.txt` |

Other contents:

- `pusht_torch/`: a batched torch renderer for the released PushT classes. See
  [pusht_torch/README.md](pusht_torch/README.md).
- `robocasa_dump.py`: builds the kitchen cache that `robocasa.py` renders from. Run it with RoboCasa
  installed.
- `render/`: render engines used only by these families.

## Use from a checkout

Run from the repository root with `stable-envs` installed (`pip install -e .`):

```python
import contrib
import stable_envs as snv

errors = contrib.load()   # {module: "ImportError: ..."} for modules with missing dependencies
env = snv.make("MiniGridEmpty", num_envs=8)
```

`contrib.load()` imports every module and records failures. `contrib.import_errors()` returns the
same mapping later. `import contrib.minigrid` registers only that module's classes. `stable_envs`
never imports `contrib`.

## Dependencies

```bash
pip install -r contrib/requirements/minigrid.txt
```

- `box2d.txt` builds Box2D from source. Install `swig` first.
- `metaworld.txt`: first run `pip install --no-deps metaworld`, then install the file.
- `humanoidbench.txt`: install `humanoid_bench` from git with `--no-deps`. The family registers only
  when `STABLE_ENVS_HUMANOIDBENCH=1` is set.
- `overcooked.txt`: `jaxmarl` conflicts with the rest of the stack. Use a separate interpreter.
- The visual-tier cells of `mujoco_locomotion.py` need nvdiffrast, built from source (see
  `docs/install.md`).

## Tests

```bash
python -m pytest contrib/tests -m "not gpu" -q   # CPU
python -m pytest contrib/tests/gpu                 # needs CUDA
```

Run them from the repository root. Plain `pytest` runs only `tests/`. CI runs the CPU tests in the
`contrib` job of `.github/workflows/ci.yml`.
