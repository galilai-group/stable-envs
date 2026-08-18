# Three ways to get a frame: 40 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [12.18408203125, 17.51806640625]

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
| MiniGridFourRooms | 608x608 | 1 | 238 | 711,102 (4,096) | 100% | 800 | **889.1x** | 1456 | 40,795 |
| MiniGridPOVFetch | 224x224 | 1 | 447 | 4,420,748 (4,096) | 100% | 5,096 | **867.4x** | 196 | 2,853 |
| MiniGridDoorKey | 256x256 | 1 | 392 | 1,211,344 (4,096) | 100% | 2,237 | **541.4x** | 268 | 6,601 |
| MiniGridKeyCorridor | 320x320 | 1 | 206 | 1,153,969 (4,096) | 100% | 2,654 | **434.8x** | 412 | 10,897 |
| MiniGridLavaCrossing | 288x288 | 1 | 938 | 1,179,476 (4,096) | 100% | 2,906 | **405.9x** | 336 | 12,861 |
| MiniHack | 336x1264 | 1 | — | 244,515 (4,096) | 100% | 721 | **339.0x** | 6636 | 26,890 |
| MiniGridEmpty | 256x256 | 1 | 854 | 1,303,556 (4,096) | 100% | 4,027 | **323.7x** | 268 | 6,311 |
| Piecewise | 224x224 | 1 | 830 | 187,975 (4,096) | 100% | 2,494 | **75.4x** | 8 | 3,030 |
| Parking | 300x600 | 1 | — | 62,138 (4,096) | 100% | 1,507 | **41.2x** | 28 | 10,196 |
| Pendulum | 500x500 | 1 | 283 | 38,396 (4,096) | 100% | 1,318 | **29.1x** | 8 | 8,171 |
| MountainCarContinuous | 400x600 | 1 | 452 | 31,137 (4,096) | 100% | 1,148 | **27.1x** | 4 | 10,998 |
| MountainCar | 400x600 | 1 | 415 | 31,111 (4,096) | 100% | 1,364 | **22.8x** | 4 | 11,008 |
| MujocoReacher | 240x240 | 1 | 2,037 | 52,901 (4,096) | 100% | 4,351 | **12.2x** | 16 | 5,027 |
| MujocoInvertedPendulum | 240x240 | 1 | 2,502 | 52,503 (4,096) | 100% | 4,645 | **11.3x** | 8 | 3,179 |
| MujocoInvertedDoublePendulum | 240x240 | 1 | 2,115 | 43,658 (4,096) | 100% | 4,757 | **9.2x** | 12 | 2,069 |
| MujocoPusher | 240x240 | 1 | 461 | 36,628 (4,096) | 1124% | 4,495 | **8.1x** | 44 | 3,482 |
| MetaWorldPushBack | 480x480 | 1 | 219 | 2,961 (4,096) | 100% | 633 | **4.7x** | 72 | 34,711 |
| MetaWorldPush | 480x480 | 1 | 217 | 2,960 (4,096) | 100% | 689 | **4.3x** | 72 | 34,820 |
| MetaWorldPegUnplugSide | 480x480 | 1 | 212 | 2,829 (4,096) | 100% | 663 | **4.3x** | 88 | 35,260 |
| MetaWorldReachWall | 480x480 | 1 | 218 | 2,940 (4,096) | 100% | 690 | **4.3x** | 76 | 35,284 |
| MetaWorldShelfPlace | 480x480 | 1 | 218 | 2,739 (4,096) | 100% | 691 | **4.0x** | 76 | 38,072 |
| MetaWorldPushWall | 480x480 | 1 | 219 | 2,942 (4,096) | 100% | 767 | **3.8x** | 72 | 35,158 |
| MetaWorldSweep | 480x480 | 1 | 220 | 2,966 (4,096) | 100% | 818 | **3.6x** | 64 | 34,608 |
| MetaWorldPegInsertSide | 480x480 | 1 | 221 | 2,829 (4,096) | 100% | 795 | **3.6x** | 80 | 35,599 |
| MetaWorldPickPlace | 480x480 | 1 | 217 | 2,959 (4,096) | 100% | 843 | **3.5x** | 76 | 34,953 |
| MetaWorldPickOutOfHole | 480x480 | 1 | 213 | 2,848 (4,096) | 100% | 822 | **3.5x** | 76 | 36,019 |
| MetaWorldPlateSlideBackSide | 480x480 | 1 | 219 | 2,589 (4,096) | 100% | 810 | **3.2x** | 48 | 38,598 |
| MetaWorldPlateSlide | 480x480 | 1 | 215 | 2,569 (4,096) | 100% | 808 | **3.2x** | 60 | 37,372 |
| MetaWorldLeverPull | 480x480 | 1 | 221 | 2,770 (4,096) | 100% | 873 | **3.2x** | 56 | 36,770 |
| MetaWorldSweepInto | 480x480 | 1 | 220 | 2,977 (4,096) | 100% | 948 | **3.1x** | 64 | 36,048 |
| MetaWorldPickPlaceWall | 480x480 | 1 | 217 | 2,940 (4,096) | 100% | 945 | **3.1x** | 76 | 35,280 |
| MetaWorldReach | 480x480 | 1 | 216 | 2,959 (4,096) | 100% | 953 | **3.1x** | 76 | 34,940 |
| MetaWorldSoccer | 480x480 | 1 | 220 | 2,407 (4,096) | 100% | 782 | **3.1x** | 80 | 37,488 |
| MetaWorldPlateSlideSide | 480x480 | 1 | 217 | 2,578 (4,096) | 100% | 852 | **3.0x** | 52 | 38,897 |
| MetaWorldStickPush | 480x480 | 1 | 217 | 2,824 (4,096) | 100% | 947 | **3.0x** | 80 | 35,318 |
| MetaWorldStickPull | 480x480 | 1 | 214 | 2,831 (4,096) | 100% | 949 | **3.0x** | 80 | 35,484 |
| MetaWorldHandlePullSide | 480x480 | 1 | 218 | 2,713 (4,096) | 100% | 953 | **2.8x** | 48 | 35,091 |
| MetaWorldPlateSlideBack | 480x480 | 1 | 217 | 2,553 (4,096) | 100% | 950 | **2.7x** | 48 | 37,236 |
| MetaWorldWindowOpen | 480x480 | 1 | 212 | 1,172 (4,096) | 100% | 776 | **1.5x** | 56 | 38,344 |
| MetaWorldWindowClose | 480x480 | 1 | 214 | 1,176 (4,096) | 100% | 859 | **1.4x** | 48 | 38,672 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| MiniGridFourRooms | 800 | 793 | 0.991 |
| MiniGridPOVFetch | 5,096 | 4,845 | 0.951 |
| MiniGridDoorKey | 2,237 | 2,199 | 0.983 |
| MiniGridKeyCorridor | 2,654 | 2,540 | 0.957 |
| MiniGridLavaCrossing | 2,906 | 2,779 | 0.956 |
| MiniHack | 721 | 631 | 0.875 |
| MiniGridEmpty | 4,027 | 3,858 | 0.958 |
| Piecewise | 2,494 | 2,600 | 1.042 |
| Parking | 1,507 | 1,458 | 0.967 |
| Pendulum | 1,318 | 1,291 | 0.980 |
| MountainCarContinuous | 1,148 | 1,131 | 0.985 |
| MountainCar | 1,364 | 1,327 | 0.973 |
| MujocoReacher | 4,351 | 4,168 | 0.958 |
| MujocoInvertedPendulum | 4,645 | 4,444 | 0.957 |
| MujocoInvertedDoublePendulum | 4,757 | 2,672 | 0.562 |
| MujocoPusher | 4,495 | 2,571 | 0.572 |
| MetaWorldPushBack | 633 | 614 | 0.971 |
| MetaWorldPush | 689 | 677 | 0.983 |
| MetaWorldPegUnplugSide | 663 | 674 | 1.016 |
| MetaWorldReachWall | 690 | 663 | 0.961 |
| MetaWorldShelfPlace | 691 | 650 | 0.940 |
| MetaWorldPushWall | 767 | 760 | 0.991 |
| MetaWorldSweep | 818 | 809 | 0.989 |
| MetaWorldPegInsertSide | 795 | 779 | 0.980 |
| MetaWorldPickPlace | 843 | 811 | 0.963 |
| MetaWorldPickOutOfHole | 822 | 804 | 0.979 |
| MetaWorldPlateSlideBackSide | 810 | 798 | 0.986 |
| MetaWorldPlateSlide | 808 | 788 | 0.974 |
| MetaWorldLeverPull | 873 | 846 | 0.969 |
| MetaWorldSweepInto | 948 | 927 | 0.978 |
| MetaWorldPickPlaceWall | 945 | 923 | 0.977 |
| MetaWorldReach | 953 | 929 | 0.975 |
| MetaWorldSoccer | 782 | 787 | 1.007 |
| MetaWorldPlateSlideSide | 852 | 846 | 0.992 |
| MetaWorldStickPush | 947 | 928 | 0.980 |
| MetaWorldStickPull | 949 | 926 | 0.976 |
| MetaWorldHandlePullSide | 953 | 936 | 0.982 |
| MetaWorldPlateSlideBack | 950 | 930 | 0.979 |
| MetaWorldWindowOpen | 776 | 776 | 0.999 |
| MetaWorldWindowClose | 859 | 842 | 0.980 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.