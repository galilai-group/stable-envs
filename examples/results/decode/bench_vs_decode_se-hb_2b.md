# Three ways to get a frame: 10 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [4.13818359375, 18.4326171875]

All three columns are frames/s for the same transformation, `state -> uint8 frame`.

* **default** — the env's own official renderer, one frame at a time, which is the only entry
  point it offers. Blank where no reference package is installed in this interpreter.
* **ours** — the GPU path at its best batch.
* **stored** — one CPU core decoding this env's own frame as JPEG q90 from memory.

The last column is `ours / stored`: **CPU cores of decode that one GPU replaces.**

| env | frame | default f/s | ours f/s (batch) | stored f/s/core | **cores replaced** | state B | JPEG B |
|---|---|---|---|---|---|---|---|
| MetaWorldPushWall | 480x480 | 83 | 2,512 (4,096) | 655 | **3.8x** | 72 | 35,162 |
| MetaWorldReachWall | 480x480 | 8 | 2,508 (4,096) | 664 | **3.8x** | 76 | 35,306 |
| MetaWorldStickPush | 480x480 | 212 | 2,198 (4,096) | 723 | **3.0x** | 80 | 35,305 |
| MetaWorldPlateSlideSide | 480x480 | 216 | 2,228 (4,096) | 801 | **2.8x** | 52 | 38,878 |
| MetaWorldPush | 480x480 | 135 | 2,522 (4,096) | 957 | **2.6x** | 72 | 34,824 |
| MetaWorldReach | 480x480 | 216 | 2,515 (4,096) | 957 | **2.6x** | 76 | 34,944 |
| MetaWorldPushBack | 480x480 | 84 | 2,525 (4,096) | 974 | **2.6x** | 72 | 34,746 |
| MetaWorldShelfPlace | 480x480 | 38 | 2,369 (4,096) | 915 | **2.6x** | 76 | 38,088 |
| MetaWorldStickPull | 480x480 | 20 | 2,202 (4,096) | 957 | **2.3x** | 80 | 35,504 |
| MetaWorldSoccer | 480x480 | 216 | 2,101 (4,096) | 930 | **2.3x** | 80 | 37,527 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| MetaWorldPushWall | 655 | 636 | 0.971 |
| MetaWorldReachWall | 664 | 646 | 0.973 |
| MetaWorldStickPush | 723 | 708 | 0.979 |
| MetaWorldPlateSlideSide | 801 | 705 | 0.880 |
| MetaWorldPush | 957 | 810 | 0.847 |
| MetaWorldReach | 957 | 934 | 0.976 |
| MetaWorldPushBack | 974 | 947 | 0.972 |
| MetaWorldShelfPlace | 915 | 892 | 0.975 |
| MetaWorldStickPull | 957 | 807 | 0.843 |
| MetaWorldSoccer | 930 | 787 | 0.846 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.