# Three ways to get a frame: 10 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [4.13818359375, 26.9609375]

All three columns are frames/s for the same transformation, `state -> uint8 frame`.

* **default** — the env's own official renderer, one frame at a time, which is the only entry
  point it offers. Blank where no reference package is installed in this interpreter.
* **ours** — the GPU path at its best batch.
* **stored** — one CPU core decoding this env's own frame as JPEG q90 from memory.

The last column is `ours / stored`: **CPU cores of decode that one GPU replaces.**

| env | frame | default f/s | ours f/s (batch) | stored f/s/core | **cores replaced** | state B | JPEG B |
|---|---|---|---|---|---|---|---|
| MiniGridPOVFetch | 224x224 | 439 | 4,479,313 (4,096) | 3,654 | **1,225.8x** | 196 | 2,853 |
| MiniGridFourRooms | 608x608 | 117 | 711,991 (4,096) | 608 | **1,171.6x** | 1456 | 40,795 |
| MiniGridKeyCorridor | 320x320 | 206 | 1,157,515 (4,096) | 1,581 | **732.2x** | 412 | 10,897 |
| MiniGridDoorKey | 256x256 | 239 | 1,210,788 (4,096) | 1,987 | **609.5x** | 268 | 6,601 |
| MiniGridLavaCrossing | 288x288 | 942 | 1,183,728 (4,096) | 2,894 | **409.0x** | 336 | 12,861 |
| MiniGridEmpty | 256x256 | 852 | 1,322,277 (4,096) | 4,011 | **329.7x** | 268 | 6,311 |
| MetaWorldSweep | 480x480 | 21 | 2,534 (1,024) | 829 | **3.1x** | 64 | 34,631 |
| MetaWorldSweepInto | 480x480 | 29 | 2,554 (1,024) | 950 | **2.7x** | 64 | 36,051 |
| MetaWorldWindowOpen | 480x480 | 90 | 1,039 (4,096) | 697 | **1.5x** | 56 | 38,418 |
| MetaWorldWindowClose | 480x480 | 200 | 1,050 (4,096) | 909 | **1.2x** | 48 | 38,743 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| MiniGridPOVFetch | 3,654 | 3,532 | 0.966 |
| MiniGridFourRooms | 608 | 591 | 0.972 |
| MiniGridKeyCorridor | 1,581 | 1,535 | 0.971 |
| MiniGridDoorKey | 1,987 | 2,392 | 1.204 |
| MiniGridLavaCrossing | 2,894 | 2,769 | 0.957 |
| MiniGridEmpty | 4,011 | 3,812 | 0.951 |
| MetaWorldSweep | 829 | 729 | 0.879 |
| MetaWorldSweepInto | 950 | 804 | 0.847 |
| MetaWorldWindowOpen | 697 | 686 | 0.984 |
| MetaWorldWindowClose | 909 | 695 | 0.764 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.