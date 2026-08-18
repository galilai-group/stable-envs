# Three ways to get a frame: 40 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [6.7734375, 4.412109375]

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
| Merge | 150x600 | 1 | — | 198,437 (4,096) | 100% | 2,862 | **69.3x** | 80 | 8,798 |
| Intersection | 600x600 | 1 | — | 25,311 (1,024) | 100% | 794 | **31.9x** | 160 | 14,279 |
| HumanoidmazeLargeNavigate | 224x224 | 2 | 438 | 79,578 (4,096) | 267% | 2,864 | **27.8x** | 112 | 9,382 |
| HumanoidmazeMediumStitch | 224x224 | 2 | 461 | 81,013 (4,096) | 271% | 3,666 | **22.1x** | 112 | 7,905 |
| HumanoidmazeLargeStitch | 224x224 | 2 | 450 | 79,658 (4,096) | 266% | 4,088 | **19.5x** | 112 | 9,382 |
| HumanoidmazeGiantStitch | 224x224 | 2 | 399 | 77,507 (4,096) | 263% | 4,060 | **19.1x** | 112 | 9,325 |
| HumanoidmazeMediumNavigate | 224x224 | 2 | 443 | 80,933 (4,096) | 268% | 4,251 | **19.0x** | 112 | 7,905 |
| HumanoidmazeGiantNavigate | 224x224 | 2 | 391 | 77,494 (4,096) | 263% | 4,154 | **18.7x** | 112 | 9,325 |
| HumanoidBenchH1Slide | 256x256 | 1 | 60 | 17,112 (4,096) | 134% | 2,085 | **8.2x** | 104 | 7,211 |
| HumanoidBenchH1Stair | 256x256 | 1 | 59 | 13,214 (4,096) | 126% | 2,329 | **5.7x** | 104 | 6,486 |
| HumanoidBenchH1Walk | 256x256 | 1 | 39 | 19,441 (4,096) | 139% | 3,856 | **5.0x** | 104 | 6,834 |
| HumanoidBenchH1Stand | 256x256 | 1 | 39 | 19,462 (4,096) | 139% | 3,903 | **5.0x** | 104 | 6,834 |
| MetaWorldAssembly | 480x480 | 1 | 214 | 2,879 (4,096) | 100% | 601 | **4.8x** | 80 | 35,052 |
| MetaWorldBasketball | 480x480 | 1 | 215 | 2,808 (4,096) | 100% | 643 | **4.4x** | 72 | 36,655 |
| MetaWorldFaucetClose | 480x480 | 1 | 213 | 2,839 (4,096) | 100% | 690 | **4.1x** | 56 | 36,666 |
| MetaWorldHandInsert | 480x480 | 1 | 173 | 2,954 (4,096) | 100% | 731 | **4.0x** | 76 | 35,852 |
| MetaWorldDoorUnlock | 480x480 | 1 | 215 | 2,616 (4,096) | 100% | 655 | **4.0x** | 48 | 35,601 |
| MetaWorldDrawerClose | 480x480 | 1 | 219 | 2,831 (4,096) | 100% | 713 | **4.0x** | 48 | 35,655 |
| MetaWorldDrawerOpen | 480x480 | 1 | 221 | 2,845 (4,096) | 100% | 720 | **4.0x** | 48 | 35,657 |
| MetaWorldBoxClose | 480x480 | 1 | 216 | 2,858 (4,096) | 100% | 741 | **3.9x** | 80 | 35,844 |
| MetaWorldHandlePressSide | 480x480 | 1 | 216 | 2,707 (4,096) | 100% | 735 | **3.7x** | 52 | 35,227 |
| MetaWorldCoffeePush | 480x480 | 1 | 213 | 2,748 (4,096) | 100% | 753 | **3.7x** | 92 | 38,358 |
| MetaWorldDialTurn | 480x480 | 1 | 217 | 2,927 (4,096) | 100% | 803 | **3.6x** | 56 | 34,812 |
| MetaWorldCoffeePull | 480x480 | 1 | 214 | 2,754 (4,096) | 100% | 758 | **3.6x** | 92 | 38,263 |
| HumanoidBenchH1SitHard | 256x256 | 1 | 38 | 13,405 (4,096) | 127% | 3,704 | **3.6x** | 132 | 8,703 |
| HumanoidBenchH1SitSimple | 256x256 | 1 | 39 | 13,392 (4,096) | 127% | 3,729 | **3.6x** | 104 | 7,753 |
| MetaWorldButtonPressWall | 480x480 | 1 | 214 | 2,744 (4,096) | 100% | 781 | **3.5x** | 52 | 38,046 |
| MetaWorldCoffeeButton | 480x480 | 1 | 213 | 2,749 (4,096) | 100% | 790 | **3.5x** | 80 | 37,758 |
| MetaWorldHammer | 480x480 | 1 | 214 | 2,779 (4,096) | 100% | 803 | **3.5x** | 68 | 37,619 |
| MetaWorldButtonPressTopdown | 480x480 | 1 | 214 | 2,804 (4,096) | 100% | 882 | **3.2x** | 48 | 37,503 |
| MetaWorldBinPicking | 480x480 | 1 | 213 | 2,893 (4,096) | 100% | 921 | **3.1x** | 64 | 35,833 |
| MetaWorldButtonPressTopdownWall | 480x480 | 1 | 215 | 2,792 (4,096) | 100% | 894 | **3.1x** | 48 | 37,828 |
| MetaWorldFaucetOpen | 480x480 | 1 | 178 | 2,838 (4,096) | 100% | 912 | **3.1x** | 56 | 36,700 |
| MetaWorldButtonPress | 480x480 | 1 | 216 | 2,767 (4,096) | 100% | 901 | **3.1x** | 48 | 37,638 |
| MetaWorldDisassemble | 480x480 | 1 | 213 | 2,837 (4,096) | 100% | 926 | **3.1x** | 88 | 36,577 |
| MetaWorldDoorClose | 480x480 | 1 | 213 | 2,616 (4,096) | 100% | 897 | **2.9x** | 56 | 38,851 |
| MetaWorldDoorOpen | 480x480 | 1 | 214 | 2,681 (4,096) | 100% | 925 | **2.9x** | 56 | 35,681 |
| MetaWorldHandlePull | 480x480 | 1 | 216 | 2,702 (4,096) | 100% | 953 | **2.8x** | 52 | 35,673 |
| MetaWorldHandlePress | 480x480 | 1 | 165 | 2,697 (4,096) | 100% | 951 | **2.8x** | 52 | 35,585 |
| MetaWorldDoorLock | 480x480 | 1 | 212 | 2,616 (4,096) | 100% | 930 | **2.8x** | 48 | 35,532 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| Merge | 2,862 | 2,777 | 0.970 |
| Intersection | 794 | 784 | 0.988 |
| HumanoidmazeLargeNavigate | 2,864 | 2,787 | 0.973 |
| HumanoidmazeMediumStitch | 3,666 | 3,511 | 0.958 |
| HumanoidmazeLargeStitch | 4,088 | 3,926 | 0.960 |
| HumanoidmazeGiantStitch | 4,060 | 3,930 | 0.968 |
| HumanoidmazeMediumNavigate | 4,251 | 4,058 | 0.955 |
| HumanoidmazeGiantNavigate | 4,154 | 3,058 | 0.736 |
| HumanoidBenchH1Slide | 2,085 | 2,759 | 1.323 |
| HumanoidBenchH1Stair | 2,329 | 2,249 | 0.966 |
| HumanoidBenchH1Walk | 3,856 | 3,692 | 0.958 |
| HumanoidBenchH1Stand | 3,903 | 3,735 | 0.957 |
| MetaWorldAssembly | 601 | 790 | 1.314 |
| MetaWorldBasketball | 643 | 898 | 1.397 |
| MetaWorldFaucetClose | 690 | 684 | 0.991 |
| MetaWorldHandInsert | 731 | 719 | 0.984 |
| MetaWorldDoorUnlock | 655 | 638 | 0.975 |
| MetaWorldDrawerClose | 713 | 701 | 0.983 |
| MetaWorldDrawerOpen | 720 | 709 | 0.984 |
| MetaWorldBoxClose | 741 | 725 | 0.979 |
| MetaWorldHandlePressSide | 735 | 724 | 0.985 |
| MetaWorldCoffeePush | 753 | 744 | 0.988 |
| MetaWorldDialTurn | 803 | 786 | 0.979 |
| MetaWorldCoffeePull | 758 | 744 | 0.982 |
| HumanoidBenchH1SitHard | 3,704 | 3,552 | 0.959 |
| HumanoidBenchH1SitSimple | 3,729 | 3,544 | 0.950 |
| MetaWorldButtonPressWall | 781 | 764 | 0.978 |
| MetaWorldCoffeeButton | 790 | 775 | 0.981 |
| MetaWorldHammer | 803 | 788 | 0.981 |
| MetaWorldButtonPressTopdown | 882 | 880 | 0.998 |
| MetaWorldBinPicking | 921 | 900 | 0.978 |
| MetaWorldButtonPressTopdownWall | 894 | 874 | 0.977 |
| MetaWorldFaucetOpen | 912 | 897 | 0.983 |
| MetaWorldButtonPress | 901 | 878 | 0.975 |
| MetaWorldDisassemble | 926 | 905 | 0.977 |
| MetaWorldDoorClose | 897 | 877 | 0.978 |
| MetaWorldDoorOpen | 925 | 903 | 0.976 |
| MetaWorldHandlePull | 953 | 932 | 0.978 |
| MetaWorldHandlePress | 951 | 929 | 0.977 |
| MetaWorldDoorLock | 930 | 908 | 0.977 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.