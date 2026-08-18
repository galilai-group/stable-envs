# Three ways to get a frame: 22 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [5.4462890625, 4.18505859375]

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
| DMCCheetahRun | 84x84 | 1 | 354 | 124,345 (4,096) | 100% | 7,357 | **16.9x** | 36 | 3,531 |
| DMCHopperHop | 84x84 | 1 | 336 | 124,436 (4,096) | 100% | 9,214 | **13.5x** | 28 | 3,275 |
| DMCWalkerWalk | 84x84 | 1 | 336 | 116,662 (4,096) | 100% | 9,451 | **12.3x** | 36 | 3,141 |
| DMCHumanoidStand | 84x84 | 1 | 282 | 101,359 (4,096) | 100% | 9,556 | **10.6x** | 112 | 3,057 |
| DMCCartpoleBalance | 84x84 | 1 | 358 | 76,456 (4,096) | 252% | 8,691 | **8.8x** | 8 | 2,183 |
| DMCSwimmerSwimmer6 | 84x84 | 1 | 272 | 67,351 (4,096) | 100% | 7,734 | **8.7x** | 48 | 1,929 |
| DMCFingerSpin | 84x84 | 1 | 364 | 61,848 (4,096) | 356% | 7,194 | **8.6x** | 12 | 2,801 |
| DMCCartpoleSwingup | 84x84 | 1 | 351 | 74,802 (4,096) | 249% | 9,437 | **7.9x** | 8 | 2,213 |
| DMCPointMassEasy | 84x84 | 1 | 366 | 59,302 (4,096) | 100% | 7,763 | **7.6x** | 8 | 1,807 |
| DMCPendulumSwingup | 84x84 | 1 | 356 | 67,711 (4,096) | 370% | 10,514 | **6.4x** | 4 | 1,967 |
| DMCFingerTurnHard | 84x84 | 1 | 352 | 40,348 (4,096) | 268% | 7,466 | **5.4x** | 20 | 2,910 |
| DMCFingerTurnEasy | 84x84 | 1 | 351 | 40,033 (4,096) | 267% | 7,537 | **5.3x** | 20 | 2,926 |
| DMCAcrobotSwingup | 84x84 | 1 | 361 | 48,877 (4,096) | 292% | 10,110 | **4.8x** | 8 | 2,204 |
| DMCReacherHard | 84x84 | 1 | 355 | 48,812 (4,096) | 295% | 10,433 | **4.7x** | 16 | 2,013 |
| DMCReacherEasy | 84x84 | 1 | 358 | 47,961 (4,096) | 291% | 10,380 | **4.6x** | 16 | 2,076 |
| MujocoHalfCheetah | 240x240 | 1 | 369 | 10,445 (4,096) | 121% | 2,430 | **4.3x** | 36 | 11,097 |
| MujocoAnt | 240x240 | 1 | 352 | 9,727 (4,096) | 120% | 2,556 | **3.8x** | 60 | 11,896 |
| MujocoSwimmer | 240x240 | 1 | 352 | 10,395 (4,096) | 121% | 2,928 | **3.6x** | 20 | 10,224 |
| MujocoHumanoid | 240x240 | 1 | 325 | 9,250 (4,096) | 119% | 2,655 | **3.5x** | 96 | 13,354 |
| MujocoHopper | 240x240 | 1 | 355 | 10,465 (4,096) | 121% | 3,057 | **3.4x** | 24 | 12,249 |
| MujocoWalker2d | 240x240 | 1 | 370 | 10,499 (4,096) | 119% | 3,168 | **3.3x** | 36 | 10,528 |
| DMCBallInCupCatch | 84x84 | 1 | 349 | 4,378 (4,096) | 120% | 7,480 | **0.6x** | 16 | 2,200 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| DMCCheetahRun | 7,357 | 6,774 | 0.921 |
| DMCHopperHop | 9,214 | 8,282 | 0.899 |
| DMCWalkerWalk | 9,451 | 8,484 | 0.898 |
| DMCHumanoidStand | 9,556 | 8,507 | 0.890 |
| DMCCartpoleBalance | 8,691 | 7,903 | 0.909 |
| DMCSwimmerSwimmer6 | 7,734 | 7,098 | 0.918 |
| DMCFingerSpin | 7,194 | 6,672 | 0.927 |
| DMCCartpoleSwingup | 9,437 | 8,523 | 0.903 |
| DMCPointMassEasy | 7,763 | 7,074 | 0.911 |
| DMCPendulumSwingup | 10,514 | 9,310 | 0.885 |
| DMCFingerTurnHard | 7,466 | 6,900 | 0.924 |
| DMCFingerTurnEasy | 7,537 | 6,956 | 0.923 |
| DMCAcrobotSwingup | 10,110 | 9,103 | 0.900 |
| DMCReacherHard | 10,433 | 9,270 | 0.889 |
| DMCReacherEasy | 10,380 | 9,280 | 0.894 |
| MujocoHalfCheetah | 2,430 | 2,340 | 0.963 |
| MujocoAnt | 2,556 | 2,462 | 0.963 |
| MujocoSwimmer | 2,928 | 2,810 | 0.960 |
| MujocoHumanoid | 2,655 | 2,556 | 0.963 |
| MujocoHopper | 3,057 | 2,926 | 0.957 |
| MujocoWalker2d | 3,168 | 3,014 | 0.952 |
| DMCBallInCupCatch | 7,480 | 6,864 | 0.918 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `humanoidbench` — ImportError: the HumanoidBench family raytraces 21-50 H1 meshes with mujoco_warp, and reads its scenes from the humanoid_bench package: missing humanoid_bench; mujoco is 3.11.0 and mujoco_warp requires >= 3.11. Note that humanoid_bench itself pins mujoco == 3.1.6 and gymnasium == 0.29.1, so it must be installed with `pip install --no-deps -e .` beside a newer mujoco; `pip install stable-envs[humanoidbench]` pulls the renderer half. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-hb. See docs/research/humanoidbench.md
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji, jax, matplotlib. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `metaworld` — ImportError: the Meta-World family ray-casts the Sawyer's meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing metaworld; mujoco is 3.11.0 and mujoco_warp requires >= 3.11 (metaworld itself pins mujoco==3.3.0, which is why it is installed with --no-deps). The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch; see docs/research/metaworld.md and the `metaworld` extra in pyproject.toml
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `procgen` — ImportError: procgen is not installed; `pip install procgen` provides both the assets this renderer packs and the reference env parity is measured against
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.