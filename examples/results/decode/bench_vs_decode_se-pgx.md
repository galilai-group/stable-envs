# Three ways to get a frame: 4 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [5.83935546875, 7.05517578125]

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
| PgxTicTacToe | 240x240 | 1 | 97 | 107,858 (4,096) | 100% | 2,493 | **43.3x** | 36 | 6,729 |
| PgxGo9x9 | 250x250 | 1 | 40 | 92,855 (4,096) | 100% | 2,336 | **39.8x** | 324 | 27,233 |
| PgxOthello | 270x270 | 1 | 21 | 81,162 (4,096) | 100% | 3,130 | **25.9x** | 256 | 13,577 |
| PgxConnectFour | 280x280 | 1 | 67 | 77,336 (4,096) | 100% | 3,101 | **24.9x** | 168 | 12,393 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| PgxTicTacToe | 2,493 | 2,567 | 1.030 |
| PgxGo9x9 | 2,336 | 2,253 | 0.965 |
| PgxOthello | 3,130 | 2,952 | 0.943 |
| PgxConnectFour | 3,101 | 2,940 | 0.948 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ModuleNotFoundError: No module named 'dm_control'
* `fetch` — ModuleNotFoundError: No module named 'mujoco'
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `humanoidbench` — ModuleNotFoundError: No module named 'mujoco'
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji, matplotlib. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `metaworld` — ModuleNotFoundError: No module named 'mujoco'
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `procgen` — ImportError: procgen is not installed; `pip install procgen` provides both the assets this renderer packs and the reference env parity is measured against
* `robocasa` — ModuleNotFoundError: No module named 'mujoco'
* `sokoban` — ModuleNotFoundError: No module named 'imageio'