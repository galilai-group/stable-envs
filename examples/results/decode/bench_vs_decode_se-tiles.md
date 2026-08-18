# Three ways to get a frame: 7 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [5.4462890625, 8.33203125]

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
| GriddlyCookMePasta | 264x336 | 1 | 147 | 5,898,273 (16,384) | 101% | 1,524 | **3,869.5x** | 616 | 28,558 |
| GriddlySokoban | 216x312 | 1 | 144 | 6,681,328 (16,384) | 101% | 2,111 | **3,165.4x** | 468 | 27,623 |
| GriddlyClusters | 240x312 | 1 | 148 | 6,056,032 (16,384) | 100% | 1,980 | **3,058.9x** | 520 | 22,811 |
| GriddlyButterflies | 264x672 | 1 | 102 | 2,547,779 (4,096) | 100% | 953 | **2,673.9x** | 1232 | 51,892 |
| GriddlyLabyrinth | 336x384 | 1 | 121 | 2,311,257 (16,384) | 102% | 910 | **2,539.5x** | 896 | 50,140 |
| GriddlySokoban2 | 168x192 | 1 | 124 | 8,316,502 (16,384) | 101% | 3,775 | **2,203.3x** | 224 | 13,285 |
| GriddlyBait | 144x120 | 1 | 171 | 9,473,899 (16,384) | 100% | 4,790 | **1,977.9x** | 120 | 7,799 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| GriddlyCookMePasta | 1,524 | 1,493 | 0.979 |
| GriddlySokoban | 2,111 | 2,051 | 0.972 |
| GriddlyClusters | 1,980 | 1,931 | 0.975 |
| GriddlyButterflies | 953 | 929 | 0.975 |
| GriddlyLabyrinth | 910 | 883 | 0.970 |
| GriddlySokoban2 | 3,775 | 3,622 | 0.960 |
| GriddlyBait | 4,790 | 4,575 | 0.955 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.9.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `fetch` — ImportError: the Fetch family renders its 18 robot meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.9.0 and mujoco_warp requires >= 3.11. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch, built with `python -m venv --system-site-packages` so that only mujoco is shadowed and gymnasium, gymnasium-robotics, torch, triton and scipy stay shared. See docs/research/fetch.md
* `humanoidbench` — ImportError: the HumanoidBench family raytraces 21-50 H1 meshes with mujoco_warp, and reads its scenes from the humanoid_bench package: missing warp, mujoco_warp, humanoid_bench; mujoco is 3.9.0 and mujoco_warp requires >= 3.11. Note that humanoid_bench itself pins mujoco == 3.1.6 and gymnasium == 0.29.1, so it must be installed with `pip install --no-deps -e .` beside a newer mujoco; `pip install stable-envs[humanoidbench]` pulls the renderer half. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-hb. See docs/research/humanoidbench.md
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji, jax. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `metaworld` — ImportError: the Meta-World family ray-casts the Sawyer's meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.9.0 and mujoco_warp requires >= 3.11 (metaworld itself pins mujoco==3.3.0, which is why it is installed with --no-deps). The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch; see docs/research/metaworld.md and the `metaworld` extra in pyproject.toml
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `robocasa` — ImportError: the RoboCasa family renders its kitchens with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.9.0 and mujoco_warp requires >= 3.11. Note that robocasa itself pins mujoco == 3.3.1, so the interpreter that BUILDS a kitchen is never the one that renders it; see the module docstring. See docs/research/robocasa.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.