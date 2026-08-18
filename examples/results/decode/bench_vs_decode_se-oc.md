# Three ways to get a frame: 5 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [6.0927734375, 7.05517578125]

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
| OvercookedCounterCircuit | 224x320 | 1 | 860 | 6,974,633 (16,384) | 100% | 2,511 | **2,777.8x** | 280 | 6,860 |
| OvercookedCoordRing | 224x224 | 1 | 1,189 | 6,968,629 (16,384) | 100% | 2,709 | **2,572.8x** | 196 | 5,733 |
| OvercookedCrampedRoom | 192x224 | 1 | 472 | 8,055,812 (16,384) | 100% | 3,715 | **2,168.4x** | 168 | 5,357 |
| OvercookedForcedCoord | 224x224 | 1 | 1,148 | 6,967,233 (16,384) | 100% | 3,974 | **1,753.4x** | 196 | 5,479 |
| OvercookedAsymmAdvantages | 224x352 | 1 | 27 | 3,596,911 (16,384) | 100% | 3,401 | **1,057.7x** | 308 | 8,163 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| OvercookedCounterCircuit | 2,511 | 2,459 | 0.980 |
| OvercookedCoordRing | 2,709 | 2,607 | 0.962 |
| OvercookedCrampedRoom | 3,715 | 3,527 | 0.949 |
| OvercookedForcedCoord | 3,974 | 3,745 | 0.942 |
| OvercookedAsymmAdvantages | 3,401 | 2,263 | 0.665 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ModuleNotFoundError: No module named 'dm_control'
* `fetch` — ImportError: the Fetch family renders its 18 robot meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.1.3 and mujoco_warp requires >= 3.11. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch, built with `python -m venv --system-site-packages` so that only mujoco is shadowed and gymnasium, gymnasium-robotics, torch, triton and scipy stay shared. See docs/research/fetch.md
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `humanoidbench` — ImportError: the HumanoidBench family raytraces 21-50 H1 meshes with mujoco_warp, and reads its scenes from the humanoid_bench package: missing warp, mujoco_warp, humanoid_bench; mujoco is 3.1.3 and mujoco_warp requires >= 3.11. Note that humanoid_bench itself pins mujoco == 3.1.6 and gymnasium == 0.29.1, so it must be installed with `pip install --no-deps -e .` beside a newer mujoco; `pip install stable-envs[humanoidbench]` pulls the renderer half. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-hb. See docs/research/humanoidbench.md
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `metaworld` — ImportError: the Meta-World family ray-casts the Sawyer's meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing metaworld, warp, mujoco_warp; mujoco is 3.1.3 and mujoco_warp requires >= 3.11 (metaworld itself pins mujoco==3.3.0, which is why it is installed with --no-deps). The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch; see docs/research/metaworld.md and the `metaworld` extra in pyproject.toml
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `procgen` — ImportError: procgen is not installed; `pip install procgen` provides both the assets this renderer packs and the reference env parity is measured against
* `robocasa` — ImportError: the RoboCasa family renders its kitchens with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.1.3 and mujoco_warp requires >= 3.11. Note that robocasa itself pins mujoco == 3.3.1, so the interpreter that BUILDS a kitchen is never the one that renders it; see the module docstring. See docs/research/robocasa.md
* `sokoban` — ModuleNotFoundError: No module named 'imageio'