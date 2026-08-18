# Three ways to get a frame: 27 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [12.64990234375, 15.82958984375]

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
| ProcgenMaze | 64x64 | 1 | 896 | 1,576,433 (4,096) | 100% | 8,138 | **193.7x** | 2516 | 3,288 |
| PushT | 96x96 | 1 | 476 | 1,181,957 (4,096) | 100% | 7,699 | **153.5x** | 20 | 2,355 |
| ProcgenMiner | 64x64 | 1 | 974 | 1,156,551 (4,096) | 100% | 9,483 | **122.0x** | 1628 | 3,304 |
| PointmazeTeleportStitch | 224x224 | 2 | 454 | 241,443 (4,096) | 100% | 2,504 | **96.4x** | 8 | 10,237 |
| PointmazeTeleportNavigate | 224x224 | 2 | 439 | 241,667 (4,096) | 100% | 3,078 | **78.5x** | 8 | 10,237 |
| PointmazeGiantNavigate | 224x224 | 2 | 406 | 242,264 (4,096) | 100% | 3,120 | **77.6x** | 8 | 9,528 |
| PointmazeGiantStitch | 224x224 | 2 | 414 | 242,479 (4,096) | 100% | 3,525 | **68.8x** | 8 | 9,528 |
| PointmazeMediumStitch | 224x224 | 2 | 488 | 242,276 (4,096) | 100% | 3,522 | **68.8x** | 8 | 7,970 |
| PointmazeLargeNavigate | 224x224 | 2 | 436 | 242,332 (4,096) | 100% | 4,080 | **59.4x** | 8 | 9,434 |
| PointmazeLargeStitch | 224x224 | 2 | 450 | 242,672 (4,096) | 100% | 4,086 | **59.4x** | 8 | 9,434 |
| PointmazeMediumNavigate | 224x224 | 2 | 469 | 241,664 (4,096) | 100% | 4,218 | **57.3x** | 8 | 7,970 |
| SimplePointMaze | 500x500 | 1 | 53 | 70,746 (4,096) | 100% | 1,317 | **53.7x** | 8 | 10,773 |
| ProcgenChaser | 64x64 | 1 | 873 | 347,350 (4,096) | 100% | 7,511 | **46.2x** | 796 | 3,490 |
| PointMazeOpen | 224x224 | 2 | 2,131 | 94,813 (4,096) | 100% | 2,355 | **40.3x** | 16 | 6,653 |
| PiecewiseZones | 224x224 | 1 | 501 | 141,119 (4,096) | 100% | 4,075 | **34.6x** | 8 | 4,181 |
| PointMazeMedium | 224x224 | 2 | 1,988 | 94,870 (4,096) | 100% | 3,104 | **30.6x** | 16 | 8,710 |
| ProcgenHeist | 64x64 | 1 | 952 | 316,300 (4,096) | 100% | 10,815 | **29.2x** | 784 | 3,017 |
| Racetrack | 600x600 | 1 | — | 24,573 (1,024) | 100% | 860 | **28.6x** | 36 | 21,860 |
| PointMazeLarge | 224x224 | 2 | 1,803 | 95,050 (4,096) | 100% | 3,857 | **24.6x** | 16 | 10,249 |
| TwoRoom | 224x224 | 1 | 668 | 117,007 (4,096) | 100% | 4,967 | **23.6x** | 8 | 4,114 |
| PointMazeUMaze | 224x224 | 2 | 2,255 | 94,990 (4,096) | 100% | 4,379 | **21.7x** | 16 | 6,837 |
| Roundabout | 600x600 | 1 | — | 16,538 (1,024) | 100% | 781 | **21.2x** | 80 | 30,343 |
| TwoRoomTarget | 224x224 | 1 | 477 | 76,785 (4,096) | 100% | 4,922 | **15.6x** | 16 | 4,383 |
| RoboCasaCounterToSinkWrist | 256x256 | 1 | 546 | 13,326 (1,024) | 100% | 2,710 | **4.9x** | 52 | 6,352 |
| RoboCasaCounterToSinkRight | 256x256 | 1 | 437 | 5,688 (4,096) | 100% | 1,833 | **3.1x** | 176 | 16,023 |
| RoboCasaCounterToSinkL45S33 | 256x256 | 1 | 394 | 5,991 (4,096) | 100% | 2,904 | **2.1x** | 148 | 14,997 |
| RoboCasaCounterToSink | 256x256 | 1 | 442 | 5,759 (4,096) | 100% | 2,990 | **1.9x** | 160 | 14,278 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| ProcgenMaze | 8,138 | 7,545 | 0.927 |
| PushT | 7,699 | 7,220 | 0.938 |
| ProcgenMiner | 9,483 | 8,693 | 0.917 |
| PointmazeTeleportStitch | 2,504 | 2,426 | 0.969 |
| PointmazeTeleportNavigate | 3,078 | 2,951 | 0.959 |
| PointmazeGiantNavigate | 3,120 | 3,006 | 0.963 |
| PointmazeGiantStitch | 3,525 | 3,378 | 0.958 |
| PointmazeMediumStitch | 3,522 | 3,396 | 0.964 |
| PointmazeLargeNavigate | 4,080 | 3,849 | 0.943 |
| PointmazeLargeStitch | 4,086 | 3,880 | 0.950 |
| PointmazeMediumNavigate | 4,218 | 4,044 | 0.959 |
| SimplePointMaze | 1,317 | 1,276 | 0.969 |
| ProcgenChaser | 7,511 | 6,986 | 0.930 |
| PointMazeOpen | 2,355 | 2,271 | 0.964 |
| PiecewiseZones | 4,075 | 3,949 | 0.969 |
| PointMazeMedium | 3,104 | 3,002 | 0.967 |
| ProcgenHeist | 10,815 | 9,813 | 0.907 |
| Racetrack | 860 | 847 | 0.985 |
| PointMazeLarge | 3,857 | 2,340 | 0.607 |
| TwoRoom | 4,967 | 2,875 | 0.579 |
| PointMazeUMaze | 4,379 | 4,173 | 0.953 |
| Roundabout | 781 | 762 | 0.976 |
| TwoRoomTarget | 4,922 | 4,686 | 0.952 |
| RoboCasaCounterToSinkWrist | 2,710 | 2,620 | 0.967 |
| RoboCasaCounterToSinkRight | 1,833 | 2,242 | 1.223 |
| RoboCasaCounterToSinkL45S33 | 2,904 | 1,829 | 0.630 |
| RoboCasaCounterToSink | 2,990 | 2,890 | 0.966 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.