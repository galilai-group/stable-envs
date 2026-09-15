# Three ways to get a frame: 1 envs

Device: NVIDIA GeForce RTX 3090 · CPU: AMD EPYC 7532 32-Core Processor · 4 cores visible · load [12.2, 11.85]

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
| DMCManipulatorBringBall | 224x224 | 1 | 262 | 8,396 (4,096) | 160% | 2,256 | **3.7x** | 60 | 12,322 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| DMCManipulatorBringBall | 2,256 | 1,544 | 0.684 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `fetch` — ImportError: the Fetch family renders its 18 robot meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.12.0 and mujoco_warp requires >= 3.11. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch, built with `python -m venv --system-site-packages` so that only mujoco is shadowed and gymnasium, gymnasium-robotics, torch, triton and scipy stay shared. See docs/research/fetch.md
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `humanoidbench` — ImportError: the HumanoidBench family is measured and complete on fidelity (17 cells, VISUAL tier, mean 2.6-8.4/255, SSIM 0.845-0.931, 437x the official renderer at batch 64) but is NOT registered by default: mujoco_warp's batched render is not bit-reproducible inside a long conformance session, so 2-3 of ~660 tests fail every run on test_determinism and test_frame_i_is_independent_of_batch_size, by 1-2/255, on a different cell each time. Set STABLE_ENVS_HUMANOIDBENCH=1 to register it anyway. See docs/research/humanoidbench.md
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji, jax, matplotlib. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `metaworld` — ImportError: the Meta-World family ray-casts the Sawyer's meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing metaworld, warp, mujoco_warp; mujoco is 3.12.0 and mujoco_warp requires >= 3.11 (metaworld itself pins mujoco==3.3.0, which is why it is installed with --no-deps). The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch; see docs/research/metaworld.md and the `metaworld` extra in pyproject.toml
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `procgen` — ImportError: procgen is not installed; `pip install procgen` provides both the assets this renderer packs and the reference env parity is measured against
* `robocasa` — ImportError: the RoboCasa family renders its kitchens with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.12.0 and mujoco_warp requires >= 3.11. Note that robocasa itself pins mujoco == 3.3.1, so the interpreter that BUILDS a kitchen is never the one that renders it; see the module docstring. See docs/research/robocasa.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.