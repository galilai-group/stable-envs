# Third-party notices

## envX TwoRoom

`stable_envs/envs/navigation/tworoom_jax.py` adapts the collision rules and integration
logic in `envx/tworooms/geometry.py` and `envx/tworooms/environment.py` from
[fracapuano/envX](https://github.com/fracapuano/envX) at commit
`d26c817c1124229c819be574f137cd68bda4380a` (Copyright 2026 Francesco Capuano, Apache-2.0).
Modifications support batched Variation, boundary rounding and DLPack exchange with Torch.
Shared navigation code handles masked steps, observations and reset/State/task contracts.
envX is not a runtime dependency.
The Apache license is included in `licenses/APACHE-2.0.txt` and distributed with the package.

## envX Reacher

`stable_envs/envs/dmcontrol/simulation/reacher_mjx.py` adapts `src/envx/reacher/env.py`
from the envX commit above under Apache-2.0.

## DeepMind Control Suite

`stable_envs/envs/dmcontrol/simulation/mjwarp_tasks.py`, `batch.py`, `first_family.py` and `humanoid.py`
adapt task observations and reward rules from the
[DeepMind Control Suite](https://github.com/google-deepmind/dm_control/tree/main/dm_control/suite) of
dm-control 1.0.50 (Copyright 2017 The dm_control Authors, Apache-2.0) to batched NumPy and Torch arrays.
dm_control is a runtime dependency of the `dmcontrol` extra; these adaptations do not import its task code.
The Apache-2.0 license text is distributed in `licenses/APACHE-2.0.txt`; the copyright notices
above apply to these adaptations.

## envX PushT

`stable_envs/envs/pusht/simulation/jax_physics.py` adapts
`src/envx/pusht/physics.py`
from the envX commit above under Apache-2.0. `numba_physics.py` and `triton_physics.py` in the same
directory port that adaptation to Numba and Triton.

## Munk2D

The PushT physics ports in `stable_envs/envs/pusht/simulation/` port parts of
[Munk2D](https://github.com/viblo/Munk2D) at commit `ade7ed72849e60289eefb7a41e79ae6322fefaf3`, the
Chipmunk2D fork that pymunk 7.2.0 builds: the wall contacts (`SegmentToPoly` and `ContactPoints` from
`src/cpCollision.c`), the arbiter contact cache (`src/cpArbiter.c`, `src/cpSpaceStep.c`) and the
bounding-box tree broadphase (`src/cpBBTree.c`, in `numba_broadphase.py` and `triton_broadphase.py`).
Munk2D is MIT-licensed ("Copyright (c) 2025 Victor Blomqvist, Copyright (c) 2007-2024 Scott Lembcke and
Howling Moon Software"). The license text is distributed in `licenses/Munk2D-MIT.txt`.

## Craftax

The `craftax` family (`stable_envs/envs/craftax/`) runs
[Craftax](https://github.com/MichaelTMatthews/Craftax), which is MIT-licensed ("Copyright (c) 2024
Michael Matthews"), as an external dependency. Its simulator, renderer and textures are not copied
here; frames are the pixel observations Craftax computes. The per-lane key handling and the uint8
frame scaling follow stable-worldmodel's `stable_worldmodel/envs/craftax/craftax.py` at commit
`21446f1ede6d5284e981bd7b47f432b994e6d812` (MIT).

## Miniworld

The `miniworld` family (`stable_envs/envs/miniworld/`) runs `MovingBlockWorld` from
[hlillemark/Miniworld](https://github.com/hlillemark/Miniworld), a fork of Farama's
[Miniworld](https://github.com/Farama-Foundation/Miniworld) (Apache-2.0), as an external dependency, at
the head of its `flowm_release` branch, the code frozen for the Flow Equivariant World Models paper. Its
renderer and textures are not copied here. `stable_envs/envs/miniworld/simulation/kernel_cpu.py`,
`kernel_torch.py` and `kernel_triton.py` port the step rules of `MiniWorldEnv.step` and
`MovingBlockWorld.step` to Numba, Torch and Triton. Modifications: batched over environments, and the reset
draws from a counter-based generator instead of the source's NumPy generator. The Apache-2.0 license text is
distributed in `licenses/APACHE-2.0.txt`. The `flowm` preset and the heading handling follow the
fork's `scripts/generate_videos.py` at commit `bc7972050155b67e242c07bfdccfe1eeddbf9882`.

## pygame drawing routines (LGPL)

`stable_envs/envs/pusht/raster.py`, `raster_cpu.py` and `raster_device.py` port the
`draw_fillpoly`, `draw_circle_filled`, `draw_line_width` and `draw_aaline` routines of
`src_c/draw.c` from [pygame](https://github.com/pygame/pygame) 2.6.1
(Copyright (C) 2000-2001 Pete Shinners, GNU Library General Public License version 2 or later).
Modifications: the routines are batched over frames, evaluated only at the pixels that the resize reads,
and run in Numba and Torch. These three files are licensed under LGPL-2.1-or-later, not MIT; each file
starts with an SPDX line that says so. The license text is in `licenses/LGPL-2.1.txt`.
pygame is not a runtime dependency of these files.

## SDL_gfx drawing rules

`stable_envs/render/gfxdraw/` reimplements in Torch the integer rasterization and
packed-channel alpha-blending rules of SDL_gfx by Andreas Schiffler, which pygame's
`gfxdraw` module uses. Upstream [SDL2_gfx](https://www.ferzkopp.net/wordpress/2016/01/02/sdl_gfx-sdl2_gfx/)
is licensed under the zlib license; the copy bundled in pygame (`src_c/SDL_gfx/`) is
marked LGPL. No SDL_gfx source is included.

## stable-worldmodel

Some modules are adapted from
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel/tree/21446f1ede6d5284e981bd7b47f432b994e6d812)
at commit `21446f1ede6d5284e981bd7b47f432b994e6d812`. Its authors are Lucas Maes,
Quentin Le Lidec, Randall Balestriero and contributors. It is MIT-licensed
("Copyright (c) 2026 GalilAI-group").

| Module in stable-envs | Source in stable-worldmodel |
|---|---|
| `stable_envs/spaces.py` | `stable_worldmodel/spaces.py` |
| `stable_envs/envs/pusht/simulation/task.py` | `stable_worldmodel/envs/pusht/env.py`, `env_discrete.py` |
| `stable_envs/envs/pusht/simulation/quantizer.py` | `stable_worldmodel/envs/quantizer.py` |
| `stable_envs/envs/pusht/draw.py` | `stable_worldmodel/envs/utils.py` (drawing subset) |
| `stable_envs/envs/rocket/simulation/task.py` | `stable_worldmodel/envs/rocket_landing/pyflyt_rocketlanding.py` |

The adapted code has no runtime import of stable-worldmodel. Physics stays in the
external dependencies under their own licenses: Pymunk/Munk2D (MIT) for PushT, and
PyFlyt (MIT) with PyBullet (zlib) for Rocket. No native physics code is copied here.

MIT permission notice for the adapted code follows:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MS-Human-700

The `mshuman` family (`stable_envs/envs/mshuman/`) loads the locomotion model of
[MS-Human-700](https://github.com/LNSGroup/MS-Human-700) at commit
`2d686957aefd5739cf4d2859a6acfd94c8f84400` (Apache-2.0). The MJCF and its assets are fetched at
runtime, not vendored here; `MSHumanLocomotion` steps and draws them through MuJoCo Warp and MuJoCo.
The Apache license is included in `licenses/APACHE-2.0.txt` and distributed with the package.

## MuJoCo

`stable_envs/render/mujoco_scene.py` and `stable_envs/render/mujoco_raster.py` follow MuJoCo's classic
fixed-function OpenGL renderer (`src/render/render_gl3.c`, `src/render/render_context.c`) of
[MuJoCo](https://github.com/google-deepmind/mujoco) 3.17.0 (Copyright 2021 DeepMind Technologies Limited,
Apache-2.0). `stable_envs/envs/fetch/raster.py` and `stable_envs/envs/fetch/transparency.py` follow the same
renderer at MuJoCo 3.11.0. Modifications: the scene extraction, lighting, shadow map and translucent-geom
ordering are reimplemented in NumPy, Numba and Warp to draw batches of frames on CPU or CUDA. MuJoCo is a
runtime dependency of the MuJoCo families; no MuJoCo C source is included. The Apache-2.0 license text is
distributed in `licenses/APACHE-2.0.txt`.

## MuJoCo Warp

[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 3.12.0 and 3.17.0 (Apache-2.0; "Copyright 2025 The
Newton Developers" in `smooth.py`, "Copyright 2026 The Newton Developers" in `render.py` and `render_util.py`) is
adapted in these files:

| Module in stable-envs | Source in MuJoCo Warp |
|---|---|
| `stable_envs/_execution/mjwarp_fused.py` | `mujoco_warp/_src/smooth.py` (`com_pos`, `crb`, `_rne_cfrc_backward`) |
| `stable_envs/_execution/mjwarp_tree.py` | `mujoco_warp/_src/smooth.py` (kinematics, `com_pos`, `com_vel`, `crb`, RNE) |
| `stable_envs/render/mjwarp/__init__.py` | `mujoco_warp/_src/render_util.py` (`create_warp_texture`), `mujoco_warp/_src/render.py` (`pack_rgba_to_uint32`) |
| `stable_envs/envs/dmcontrol/trackcom_reflection.py` | `mujoco_warp/_src/render.py` (ray cast, lighting and texture sampling of the reflected rays) |
| `stable_envs/render/cpu3d.py`, `stable_envs/render/cpu_raster/scene3d.py` | `mujoco_warp/_src/render.py` (shading rules) |

Modifications: the tree passes are fused into one launch per pass or one thread block per world, the texture
upload is shared between textures and can be downsampled, the RGB packing rounds instead of truncating, and the
shading rules are reimplemented in Torch and Numba for CPU. MuJoCo Warp is a runtime dependency of the CUDA
backends. The Apache-2.0 license text is distributed in `licenses/APACHE-2.0.txt`.

## PyFlyt

`stable_envs/envs/rocket/simulation/numba_physics.py` and `numba_cuda_physics.py` port the Rocket landing
physics and task of [PyFlyt](https://github.com/jjshoots/PyFlyt) 0.29.0 (`PyFlyt/core/drones/rocket.py`,
`PyFlyt/gym_envs/rocket_envs/rocket_base_env.py`, `rocket_landing_env.py`, MIT). Modifications: the forces
and the task logic are rewritten as Numba CPU and CUDA kernels that step many environments in parallel.
PyFlyt is a runtime dependency of the `rocket` extra for `simulation_backend="pybullet"`. The PyFlyt 0.29.0
license file has no copyright line; its text is distributed in `licenses/PyFlyt-MIT.txt`.

## Bullet, PyBullet and TinyRenderer

`stable_envs/envs/rocket/tiny_raster.py` is an altered Numba port of the TinyRenderer of
[Bullet](https://github.com/bulletphysics/bullet3) as built in pybullet 3.2.7
(`examples/TinyRenderer/our_gl.cpp`, `TinyRenderer.cpp`, `model.cpp`,
`TinyRendererVisualShapeConverter.cpp`) and of the convex
hull that Bullet builds for URDF cylinders (`btShapeHull`, `HullLibrary`). `stable_envs/envs/rocket/raster.py`,
`stable_envs/envs/rocket/render.py` and the TinyRenderer lighting in `stable_envs/render/nvdiff.py` follow the
same renderer. `stable_envs/envs/rocket/simulation/numba_physics.py` and `numba_cuda_physics.py` follow
Bullet's single-body Featherstone integration and multibody contact solver. These files are altered versions:
they are not the original Bullet software. Modifications: the arithmetic is rewritten in Numba and
CUDA to draw and step many environments in parallel. Bullet and pybullet are licensed under the zlib license
(pybullet authors: Erwin Coumans, Yunfei Bai, Jasmine Hsu). pybullet is a runtime dependency of the `rocket`
extra. The license text is distributed in `licenses/Bullet-zlib.txt`.

## Gymnasium

`stable_envs/envs/classic_control/simulation/batched.py`, `triton_step.py` and `device.py` port the `step`
functions of the classic-control environments of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
1.3.0 (`gymnasium/envs/classic_control/`, MIT, "Copyright (c) 2016 OpenAI, Copyright (c) 2022 Farama
Foundation"). `stable_envs/render/gfxdraw/` draws the frames of their `render` functions. Modifications: the
steps run on batches in NumPy, Torch and Triton in the source's operation order. Gymnasium is a runtime
dependency, and CPU resets stay in Gymnasium. The license text is distributed in `licenses/Gymnasium-MIT.txt`.

## Gymnasium-Robotics

`stable_envs/envs/fetch/simulation/batch.py`, `mjwarp.py` and `observations.py` port the step, reset and
observation code of the Fetch environments of
[Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) 1.4.2
(`gymnasium_robotics/envs/fetch/`, MIT, "Copyright (c) 2022 Farama Foundation"). Modifications: the step and
reset run on arrays of environments, with MuJoCo in one batched call or MuJoCo Warp on CUDA.
Gymnasium-Robotics is a runtime dependency of the `robotics` extra. The license text is distributed in
`licenses/Gymnasium-Robotics-MIT.txt`.

## OGBench

`stable_envs/envs/ogbench/simulation/manipulation_control.py`, `manipulation_batch.py`,
`manipulation_warp.py` and `maze_batch.py` port the controller, step, observation and task code of
[OGBench](https://github.com/seohongpark/ogbench) 1.2.1 (`ManipSpaceEnv`, the locomotion maze
environments, MIT, "Copyright (c) 2024 OGBench Authors"). Modifications: the code runs on batches of
environments, with MuJoCo in one batched call or MuJoCo Warp and Warp kernels on CUDA. OGBench is a runtime
dependency of the `ogbench` extras. The license text is distributed in `licenses/OGBench-MIT.txt`.

## NumPy random

`stable_envs/envs/rocket/simulation/numba_physics.py` ports NumPy's `PCG64.next_uint64` and the ziggurat
`Generator.standard_normal`, and `stable_envs/envs/ogbench/simulation/maze_batch.py` ports the MT19937
`RandomState` seeding, generation, `uniform` and `randint`, from [NumPy](https://github.com/numpy/numpy)
(`numpy/random/`). NumPy is BSD-3-Clause ("Copyright (c) 2005-2025, NumPy Developers"), and its
`numpy/random` module is dual-licensed NCSA and BSD-3-Clause ("Copyright (c) 2019 Kevin Sheppard");
MT19937 is by Makoto Matsumoto and Takuji Nishimura, PCG64 by Melissa O'Neill. Modifications: the generators
are rewritten in Numba and NumPy to draw the same numbers for many environments at once. NumPy is a runtime
dependency. The license texts are distributed in `licenses/NumPy-BSD-3-Clause.txt` and
`licenses/NumPy-random-NCSA-BSD-3-Clause.txt`.

## OpenCV

`stable_envs/envs/pusht/raster.py` and `raster_cpu.py` reimplement the fixed-point
`cv2.resize` `INTER_LINEAR` arithmetic of [OpenCV](https://github.com/opencv/opencv) 5.0
(opencv-python-headless 5.0.0.93, Apache-2.0). Modifications: only the offsets and 11-bit weights are computed,
in integer Torch and Numba code. OpenCV is a runtime dependency of the `pusht` extra; no OpenCV
source is included. The Apache-2.0 license text is distributed in `licenses/APACHE-2.0.txt`.
