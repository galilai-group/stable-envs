# Three ways to get a frame: 10 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [4.13818359375, 20.33447265625]

All three columns are frames/s for the same transformation, `state -> uint8 frame`.

* **default** — the env's own official renderer, one frame at a time, which is the only entry
  point it offers. Blank where no reference package is installed in this interpreter.
* **ours** — the GPU path at its best batch.
* **stored** — one CPU core decoding this env's own frame as JPEG q90 from memory.

The last column is `ours / stored`: **CPU cores of decode that one GPU replaces.**

| env | frame | default f/s | ours f/s (batch) | stored f/s/core | **cores replaced** | state B | JPEG B |
|---|---|---|---|---|---|---|---|
| MetaWorldPickPlaceWall | 480x480 | 149 | 2,508 (4,096) | 672 | **3.7x** | 76 | 35,295 |
| MetaWorldPegUnplugSide | 480x480 | 66 | 2,395 (4,096) | 659 | **3.6x** | 88 | 35,258 |
| MetaWorldPegInsertSide | 480x480 | 101 | 2,424 (4,096) | 812 | **3.0x** | 80 | 35,576 |
| MetaWorldPlateSlideBack | 480x480 | 217 | 2,207 (4,096) | 805 | **2.7x** | 48 | 37,222 |
| MetaWorldPickPlace | 480x480 | 9 | 2,519 (4,096) | 961 | **2.6x** | 76 | 34,959 |
| MetaWorldPickOutOfHole | 480x480 | 209 | 2,440 (4,096) | 939 | **2.6x** | 76 | 36,028 |
| MetaWorldLeverPull | 480x480 | 162 | 2,348 (4,096) | 938 | **2.5x** | 56 | 36,849 |
| MetaWorldHandlePullSide | 480x480 | 215 | 2,326 (4,096) | 937 | **2.5x** | 48 | 35,111 |
| MetaWorldPlateSlideBackSide | 480x480 | 215 | 2,236 (4,096) | 916 | **2.4x** | 48 | 38,627 |
| MetaWorldPlateSlide | 480x480 | 214 | 2,221 (4,096) | 933 | **2.4x** | 60 | 37,378 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| MetaWorldPickPlaceWall | 672 | 662 | 0.985 |
| MetaWorldPegUnplugSide | 659 | 642 | 0.974 |
| MetaWorldPegInsertSide | 812 | 795 | 0.979 |
| MetaWorldPlateSlideBack | 805 | 917 | 1.139 |
| MetaWorldPickPlace | 961 | 811 | 0.844 |
| MetaWorldPickOutOfHole | 939 | 618 | 0.658 |
| MetaWorldLeverPull | 938 | 708 | 0.755 |
| MetaWorldHandlePullSide | 937 | 915 | 0.976 |
| MetaWorldPlateSlideBackSide | 916 | 889 | 0.971 |
| MetaWorldPlateSlide | 933 | 892 | 0.956 |

## Modules absent in this interpreter

* `carracing` — ImportError: CarRacing-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `dmcontrol` — ImportError: dm_control is installed but cannot compile a model with mujoco 3.11.0: 'MjModel' object has no attribute 'flex_bandwidth'. This is a version skew, not a missing package — no dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with dm_control==1.0.41. See docs/research/dmcontrol.md.
* `griddly` — ImportError: griddly is not importable (ModuleNotFoundError: No module named 'griddly'). The atlas is composed of tiles griddly's own Vulkan renderer draws, so there is no degraded mode. `pip install griddly==1.6.7` — the only release, from 2023-03-15; upstream's last commit is 2024-04-07 and there is no successor.
* `jumanji` — ImportError: the Jumanji family builds its per-pixel atlas by CALLING the official matplotlib viewer, so it cannot render without the reference package: missing jumanji. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-jumanji, built with a CLEAN `python -m venv` — a --system-site-packages venv breaks jax's CUDA plugin discovery. See docs/research/jumanji.md
* `lunarlander` — ImportError: LunarLander-v3 cannot be reset and rendered: DependencyNotInstalled: Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`. This env needs BOTH `box2d-py` (for the simulator that produces its states) and `pygame` (for the reference renderer parity is measured against); `pip install swig` then `pip install "gymnasium[box2d]"`.
* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxmarl'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.
* `pgx` — ImportError: the PGX family needs pgx (which brings jax), resvg_py and Pillow, and this interpreter cannot provide them: ModuleNotFoundError: No module named 'pgx'. The envs were measured in /mnt/vast/home/randall.balestriero/venvs/se-pgx, built with a CLEAN `python -m venv` -- NOT --system-site-packages, which breaks jax's CUDA plugin discovery. See docs/research/pgx.md
* `sokoban` — ImportError: gym-sokoban is not installed. The seven 16x16 tile images ARE this env -- the frame is a gather from them and nothing here re-derives them -- so there is no degraded mode to fall back to. `pip install gym-sokoban`.