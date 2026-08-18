# Three ways to get a frame: 10 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 8 cores visible · load [7.17724609375, 8.4873046875]

All three columns are frames/s for the same transformation, `state -> uint8 frame`.

* **default** — the env's own official renderer, one frame at a time, which is the only entry
  point it offers. Blank where no reference package is installed in this interpreter.
* **ours** — the GPU path at its best batch.
* **stored** — one CPU core decoding this env's own frame as JPEG q90 from memory.

The last column is `ours / stored`: **CPU cores of decode that one GPU replaces.**

`ss` is the supersample each env RESOLVED to, not the one the flag asked for. It belongs in
the table because 44 of the classes declare `default_supersample = 2` and a sweep that forces
1 on them reports a four-times-cheaper picture than `se.make(name)` produces — measured, 3.2x
on DMCBallInCupCatch.

`cpu%` is the process CPU seconds per wall second of the timed loop. Near 0 the device is
the limit and the row travels between nodes; at or above 100% the host is busy for the whole
render, and that row is a measurement of this node's processor under this node's load —
DMCBallInCupCatch measures 196% and has produced 3,664 f/s idle against 271 contended.

| env | frame | ss | default f/s | ours f/s (batch) | cpu% | stored f/s/core | **cores replaced** | state B | JPEG B |
|---|---|---|---|---|---|---|---|---|---|
| MiniHack | 336x1264 | ? | — | 244,456 (4,096) | — | 698 | **350.0x** | 6636 | 26,890 |
| Piecewise | 224x224 | ? | 772 | 187,629 (4,096) | — | 2,163 | **86.7x** | 8 | 3,030 |
| Pendulum | 500x500 | ? | 312 | 39,843 (4,096) | — | 1,120 | **35.6x** | 8 | 8,171 |
| MountainCar | 400x600 | ? | 485 | 33,216 (4,096) | — | 1,150 | **28.9x** | 4 | 11,008 |
| MountainCarContinuous | 400x600 | ? | 541 | 33,236 (4,096) | — | 1,493 | **22.3x** | 4 | 10,998 |
| Parking | 300x600 | ? | — | 7,398 (4,096) | — | 1,914 | **3.9x** | 28 | 10,196 |
| MujocoReacher | 240x240 | 2 | — | 7,756 (16,384) | 100% | 2,751 | **2.8x** | 16 | 4,971 |
| MujocoInvertedPendulum | 240x240 | 2 | — | 7,652 (16,384) | 100% | 3,389 | **2.3x** | 8 | 3,143 |
| MujocoPusher | 240x240 | 2 | — | 4,404 (16,384) | 100% | 2,390 | **1.8x** | 44 | 3,379 |
| MujocoInvertedDoublePendulum | 240x240 | 2 | — | 7,527 (4,096) | 100% | 4,834 | **1.6x** | 12 | 2,012 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| MiniHack | 698 | 640 | 0.916 |
| Piecewise | 2,163 | 2,145 | 0.992 |
| Pendulum | 1,120 | 1,102 | 0.983 |
| MountainCar | 1,150 | 1,450 | 1.260 |
| MountainCarContinuous | 1,493 | 1,468 | 0.983 |
| Parking | 1,914 | 1,247 | 0.652 |
| MujocoReacher | 2,751 | 2,667 | 0.970 |
| MujocoInvertedPendulum | 3,389 | 3,262 | 0.963 |
| MujocoPusher | 2,390 | 2,327 | 0.974 |
| MujocoInvertedDoublePendulum | 4,834 | 4,590 | 0.950 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.