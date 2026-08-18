# Three ways to get a frame: 40 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [12.18408203125, 14.83642578125]

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
| BabyAIGoToLocal | 256x256 | 1 | 395 | 1,235,817 (4,096) | 100% | 3,730 | **331.3x** | 268 | 9,142 |
| CraftaxClassic | 112x144 | 1 | — | 568,670 (4,096) | 100% | 5,544 | **102.6x** | 508 | 6,826 |
| Crafter | 64x64 | 1 | — | 661,255 (4,096) | 100% | 9,648 | **68.5x** | 320 | 2,315 |
| Highway | 150x600 | 1 | — | 199,371 (4,096) | 100% | 2,926 | **68.1x** | 192 | 12,245 |
| AntmazeTeleportExplore | 224x224 | 2 | 411 | 90,246 (4,096) | 287% | 2,368 | **38.1x** | 60 | 10,244 |
| AntmazeLargeNavigate | 224x224 | 2 | 442 | 90,635 (4,096) | 287% | 2,422 | **37.4x** | 60 | 9,437 |
| AntmazeMediumExplore | 224x224 | 2 | 448 | 92,149 (4,096) | 290% | 2,615 | **35.2x** | 60 | 7,952 |
| AntmazeLargeStitch | 224x224 | 2 | 441 | 90,808 (4,096) | 287% | 3,102 | **29.3x** | 60 | 9,437 |
| AntmazeTeleportNavigate | 224x224 | 2 | 425 | 90,294 (4,096) | 288% | 3,090 | **29.2x** | 60 | 10,244 |
| AntmazeLargeExplore | 224x224 | 2 | 429 | 90,728 (4,096) | 288% | 3,129 | **29.0x** | 60 | 9,437 |
| Acrobot | 500x500 | 1 | 408 | 31,630 (4,096) | 100% | 1,101 | **28.7x** | 8 | 10,025 |
| AntmazeTeleportStitch | 224x224 | 2 | 425 | 89,997 (4,096) | 286% | 3,487 | **25.8x** | 60 | 10,244 |
| CartPole | 400x600 | 1 | 447 | 35,973 (4,096) | 100% | 1,441 | **25.0x** | 8 | 7,393 |
| AntmazeGiantStitch | 224x224 | 2 | 399 | 87,724 (4,096) | 281% | 3,548 | **24.7x** | 60 | 9,498 |
| AntmazeMediumNavigate | 224x224 | 2 | 465 | 92,167 (4,096) | 290% | 4,194 | **22.0x** | 60 | 7,952 |
| AntmazeMediumStitch | 224x224 | 2 | 465 | 92,105 (4,096) | 291% | 4,238 | **21.7x** | 60 | 7,952 |
| AntmazeGiantNavigate | 224x224 | 2 | 391 | 87,636 (4,096) | 280% | 4,040 | **21.7x** | 60 | 9,498 |
| FetchSlide | 480x480 | 1 | 594 | 8,958 (4,096) | 100% | 850 | **10.5x** | 100 | 19,900 |
| FetchPickAndPlace | 480x480 | 1 | 615 | 8,653 (4,096) | 100% | 857 | **10.1x** | 100 | 19,130 |
| FetchPickAndPlaceDict | 480x480 | 1 | 706 | 8,655 (4,096) | 100% | 860 | **10.1x** | 100 | 19,130 |
| FetchReachDict | 480x480 | 1 | 713 | 8,682 (4,096) | 100% | 874 | **9.9x** | 72 | 19,249 |
| FetchPickAndPlaceDense | 480x480 | 1 | 702 | 8,654 (4,096) | 100% | 1,000 | **8.7x** | 100 | 19,130 |
| FetchReach | 480x480 | 1 | 621 | 8,686 (4,096) | 100% | 1,026 | **8.5x** | 72 | 19,249 |
| FetchReachDense | 480x480 | 1 | 703 | 8,678 (4,096) | 100% | 1,045 | **8.3x** | 72 | 19,249 |
| HumanoidBenchH1HandSlide | 256x256 | 1 | 57 | 15,481 (4,096) | 131% | 1,916 | **8.1x** | 304 | 7,121 |
| HumanoidBenchH1HandRun | 256x256 | 1 | 38 | 17,481 (4,096) | 135% | 2,188 | **8.0x** | 304 | 7,408 |
| FetchSlideDense | 480x480 | 1 | 591 | 8,953 (4,096) | 100% | 1,140 | **7.9x** | 100 | 19,900 |
| FetchPushDict | 480x480 | 1 | 678 | 8,749 (4,096) | 100% | 1,115 | **7.8x** | 100 | 19,326 |
| FetchSlideDict | 480x480 | 1 | 586 | 8,956 (4,096) | 100% | 1,228 | **7.3x** | 100 | 19,900 |
| FetchPushDense | 480x480 | 1 | 672 | 8,746 (4,096) | 100% | 1,250 | **7.0x** | 100 | 19,326 |
| FetchPush | 480x480 | 1 | 600 | 8,744 (4,096) | 100% | 1,255 | **7.0x** | 100 | 19,326 |
| HumanoidBenchH1HandWalk | 256x256 | 1 | 38 | 17,471 (4,096) | 135% | 2,891 | **6.0x** | 304 | 7,408 |
| HumanoidBenchH1HandStand | 256x256 | 1 | 38 | 17,459 (4,096) | 135% | 2,915 | **6.0x** | 304 | 7,408 |
| HumanoidBenchH1HandSitHard | 256x256 | 1 | 38 | 12,023 (4,096) | 124% | 2,411 | **5.0x** | 332 | 7,764 |
| HumanoidBenchH1Run | 256x256 | 1 | 39 | 19,496 (4,096) | 139% | 3,920 | **5.0x** | 104 | 6,834 |
| HumanoidBenchH1BalanceSimple | 256x256 | 1 | 39 | 12,588 (4,096) | 125% | 2,960 | **4.3x** | 132 | 7,138 |
| HumanoidBenchH1HandBalanceSimple | 256x256 | 1 | 38 | 11,280 (4,096) | 123% | 2,883 | **3.9x** | 332 | 7,579 |
| HumanoidBenchH1HandSitSimple | 256x256 | 1 | 38 | 12,118 (4,096) | 124% | 3,674 | **3.3x** | 304 | 8,618 |
| HumanoidBenchH1HandStair | 256x256 | 1 | 57 | 11,867 (4,096) | 124% | 3,898 | **3.0x** | 304 | 6,521 |
| HumanoidBenchH1HandSpoon | 256x256 | 1 | 38 | 10,489 (1,024) | 100% | 3,538 | **3.0x** | 332 | 10,107 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| BabyAIGoToLocal | 3,730 | 2,176 | 0.583 |
| CraftaxClassic | 5,544 | 5,069 | 0.914 |
| Crafter | 9,648 | 8,587 | 0.890 |
| Highway | 2,926 | 2,477 | 0.846 |
| AntmazeTeleportExplore | 2,368 | 2,300 | 0.971 |
| AntmazeLargeNavigate | 2,422 | 2,353 | 0.971 |
| AntmazeMediumExplore | 2,615 | 2,511 | 0.960 |
| AntmazeLargeStitch | 3,102 | 2,968 | 0.957 |
| AntmazeTeleportNavigate | 3,090 | 2,972 | 0.962 |
| AntmazeLargeExplore | 3,129 | 3,021 | 0.965 |
| Acrobot | 1,101 | 1,065 | 0.967 |
| AntmazeTeleportStitch | 3,487 | 3,342 | 0.958 |
| CartPole | 1,441 | 1,400 | 0.972 |
| AntmazeGiantStitch | 3,548 | 3,419 | 0.964 |
| AntmazeMediumNavigate | 4,194 | 4,001 | 0.954 |
| AntmazeMediumStitch | 4,238 | 3,984 | 0.940 |
| AntmazeGiantNavigate | 4,040 | 3,027 | 0.749 |
| FetchSlide | 850 | 837 | 0.984 |
| FetchPickAndPlace | 857 | 877 | 1.024 |
| FetchPickAndPlaceDict | 860 | 844 | 0.981 |
| FetchReachDict | 874 | 849 | 0.971 |
| FetchPickAndPlaceDense | 1,000 | 1,001 | 1.000 |
| FetchReach | 1,026 | 1,005 | 0.979 |
| FetchReachDense | 1,045 | 1,016 | 0.972 |
| HumanoidBenchH1HandSlide | 1,916 | 3,660 | 1.911 |
| HumanoidBenchH1HandRun | 2,188 | 2,131 | 0.974 |
| FetchSlideDense | 1,140 | 1,110 | 0.974 |
| FetchPushDict | 1,115 | 1,086 | 0.974 |
| FetchSlideDict | 1,228 | 1,193 | 0.972 |
| FetchPushDense | 1,250 | 1,215 | 0.972 |
| FetchPush | 1,255 | 833 | 0.664 |
| HumanoidBenchH1HandWalk | 2,891 | 2,800 | 0.969 |
| HumanoidBenchH1HandStand | 2,915 | 2,772 | 0.951 |
| HumanoidBenchH1HandSitHard | 2,411 | 2,343 | 0.972 |
| HumanoidBenchH1Run | 3,920 | 3,768 | 0.961 |
| HumanoidBenchH1BalanceSimple | 2,960 | 2,161 | 0.730 |
| HumanoidBenchH1HandBalanceSimple | 2,883 | 2,797 | 0.970 |
| HumanoidBenchH1HandSitSimple | 3,674 | 3,520 | 0.958 |
| HumanoidBenchH1HandStair | 3,898 | 3,721 | 0.955 |
| HumanoidBenchH1HandSpoon | 3,538 | 3,403 | 0.962 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.