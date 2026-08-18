# Three ways to get a frame: 3 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [5.03271484375, 6.19482421875]

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
| Sokoban | 160x160 | 1 | 256 | 9,572,010 (16,384) | 100% | 4,186 | **2,286.6x** | 400 | 13,225 |
| LunarLander | 400x600 | 1 | 339 | 18,825 (4,096) | 100% | 1,399 | **13.5x** | 152 | 10,482 |
| CarRacing | 96x96 | 1 | 132 | 26,927 (4,096) | 100% | 8,118 | **3.3x** | 112 | 2,565 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| Sokoban | 4,186 | 3,996 | 0.955 |
| LunarLander | 1,399 | 1,359 | 0.972 |
| CarRacing | 8,118 | 7,596 | 0.936 |

## Modules absent in this interpreter

* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.9.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `fetch` — ImportError: the Fetch family renders its 18 robot meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.9.0 and mujoco_warp requires >= 3.11. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch, built with `python -m venv --system-site-packages` so that only mujoco is shadowed and gymnasium, gymnasium-robotics, torch, triton and scipy stay shared. See docs/research/fetch.md
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `humanoidbench` — ImportError: the HumanoidBench family raytraces 21-50 H1 meshes with mujoco_warp, and reads its scenes from the humanoid_bench package: missing warp, mujoco_warp, humanoid_bench; mujoco is 3.9.0 and mujoco_warp requires >= 3.11. Note that humanoid_bench itself pins mujoco == 3.1.6 and gymnasium == 0.29.1, so it must be installed with `pip install --no-deps -e .` beside a newer mujoco; `pip install stable-envs[humanoidbench]` pulls the renderer half. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-hb. See docs/research/humanoidbench.md
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji, jax. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `metaworld` — ImportError: the Meta-World family ray-casts the Sawyer's meshes with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.9.0 and mujoco_warp requires >= 3.11 (metaworld itself pins mujoco==3.3.0, which is why it is installed with --no-deps). The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-fetch; see docs/research/metaworld.md and the `metaworld` extra in pyproject.toml
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `robocasa` — ImportError: the RoboCasa family renders its kitchens with mujoco_warp's batch raytracer, which this interpreter cannot run: missing warp, mujoco_warp; mujoco is 3.9.0 and mujoco_warp requires >= 3.11. Note that robocasa itself pins mujoco == 3.3.1, so the interpreter that BUILDS a kitchen is never the one that renders it; see the module docstring. See docs/research/robocasa.md