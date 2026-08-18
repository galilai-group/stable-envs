# Three ways to get a frame: 8 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [5.65087890625, 6.14111328125]

All three columns are frames/s for the same transformation, `state -> uint8 frame`.

* **default** — the env's own official renderer, one frame at a time, which is the only entry
  point it offers. Blank where no reference package is installed in this interpreter.
* **ours** — the GPU path at its best batch.
* **stored** — one CPU core decoding this env's own frame as JPEG q90 from memory.

The last column is `ours / stored`: **CPU cores of decode that one GPU replaces.**

`ss` is the supersample each env RESOLVED to, not the one the flag asked for. It belongs in
the table because 44 of the classes declare `default_supersample = 2` and a sweep that forces
1 on them reports a cheaper picture than `se.make(name)` produces: 1.16x on
DMCBallInCupCatch, 1.26x on AntmazeLargeNavigate and 3.3x on MujocoPusher, measured idle.

`cpu%` is the process CPU seconds per wall second of the timed loop. Near 0 the device is
the limit and the row travels between nodes; at or above 100% the host is busy for the whole
render, and that row is a measurement of this node's processor under this node's load —
DMCBallInCupCatch measures 196% and has produced 3,664 f/s idle against 271 contended.

| env | frame | ss | default f/s | ours f/s (batch) | cpu% | stored f/s/core | **cores replaced** | state B | JPEG B |
|---|---|---|---|---|---|---|---|---|---|
| JumanjiMinesweeper | 200x200 | 1 | 11 | 1,196,463 (16,384) | 100% | 3,405 | **351.3x** | 400 | 9,588 |
| JumanjiSudoku | 200x200 | 1 | 12 | 1,282,419 (16,384) | 100% | 3,708 | **345.8x** | 324 | 9,726 |
| JumanjiSudokuVeryEasy | 200x200 | 1 | 10 | 1,282,433 (16,384) | 100% | 3,984 | **321.9x** | 324 | 9,856 |
| JumanjiRubiksCubePartlyScrambled | 200x200 | 1 | 8 | 593,505 (16,384) | 100% | 3,562 | **166.6x** | 216 | 11,149 |
| JumanjiRubiksCube | 200x200 | 1 | 8 | 593,746 (16,384) | 100% | 3,612 | **164.4x** | 216 | 11,065 |
| JumanjiMaze | 200x200 | 1 | 82 | 564,824 (16,384) | 100% | 3,892 | **145.1x** | 400 | 7,939 |
| JumanjiSokoban | 200x200 | 1 | 6 | 403,573 (16,384) | 100% | 2,829 | **142.7x** | 400 | 18,244 |
| JumanjiConnector | 200x200 | 1 | 4 | 455,587 (16,384) | 100% | 3,768 | **120.9x** | 400 | 9,895 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| JumanjiMinesweeper | 3,405 | 3,241 | 0.952 |
| JumanjiSudoku | 3,708 | 3,527 | 0.951 |
| JumanjiSudokuVeryEasy | 3,984 | 3,757 | 0.943 |
| JumanjiRubiksCubePartlyScrambled | 3,562 | 3,339 | 0.938 |
| JumanjiRubiksCube | 3,612 | 3,402 | 0.942 |
| JumanjiMaze | 3,892 | 3,677 | 0.945 |
| JumanjiSokoban | 2,829 | 2,701 | 0.955 |
| JumanjiConnector | 3,768 | 3,573 | 0.948 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ModuleNotFoundError: No module named 'dm_control'
* `fetch` — ModuleNotFoundError: No module named 'mujoco'
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `humanoidbench` — ModuleNotFoundError: No module named 'mujoco'
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `metaworld` — ModuleNotFoundError: No module named 'mujoco'
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `procgen` — ImportError: procgen is not installed; `pip install procgen` provides both the assets this renderer packs and the reference env parity is measured against
* `robocasa` — ModuleNotFoundError: No module named 'mujoco'
* `sokoban` — ModuleNotFoundError: No module named 'imageio'