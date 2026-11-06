# Third-party notices

## envX TwoRoom

`stable_envs/envs/navigation/tworoom_jax.py` adapts the collision rules and integration
logic in `envx/tworooms/geometry.py` and `envx/tworooms/environment.py` from
[fracapuano/envX](https://github.com/fracapuano/envX) at commit
`d26c817c1124229c819be574f137cd68bda4380a` (Copyright 2026 Francesco Capuano, Apache-2.0).
Modifications support batched Variation, boundary rounding and DLPack exchange with Torch.
Shared navigation code handles masked steps, observations and reset/State/task contracts.
envX is not a runtime dependency.
The Apache license is included in `licenses/envX-APACHE-2.0.txt` and distributed with the package.

## envX Reacher

`stable_envs/envs/dmcontrol/simulation/reacher_mjx.py` adapts `src/envx/reacher/env.py`
from the envX commit above under Apache-2.0.

## DeepMind Control Suite and MuJoCo

`stable_envs/envs/dmcontrol/simulation/mjwarp_tasks.py` adapts task observations and reward rules
from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control/tree/main/dm_control/suite)
(Copyright 2017 The dm_control Authors, Apache-2.0) to batched Torch arrays.
The contact compatibility routines in `simulation/mjwarp_runtime.py` adapt the primitive collision
and frame construction rules of [MuJoCo 3.17.0](https://github.com/google-deepmind/mujoco/tree/3.17.0/src/engine)
(Copyright 2021 DeepMind Technologies Limited, Apache-2.0) to runtime-local Warp kernels.
The DMC-local Newton loops in the same module adapt the solver control flow from
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 3.17.0's `_src/solver.py`
(Copyright 2025 The Newton Developers, Apache-2.0), adding ordered reductions while retaining
upstream solver kernels, convergence and warmstart behavior.
The Apache-2.0 license text is distributed in `licenses/envX-APACHE-2.0.txt`; the copyright notices
above apply to these adaptations.

## envX PushT

`stable_envs/envs/pusht/simulation/jax_physics.py` adapts
`src/envx/pusht/physics.py`
from the envX commit above under Apache-2.0.

## Craftax

The `craftax` family (`stable_envs/envs/craftax/`) runs
[Craftax](https://github.com/MichaelTMatthews/Craftax), which is MIT-licensed ("Copyright (c) 2024
Michael Matthews"), as an external dependency. Its simulator, renderer and textures are not copied
here; frames are the pixel observations Craftax computes. The per-lane key handling and the uint8
frame scaling follow stable-worldmodel's `stable_worldmodel/envs/craftax/craftax.py` at commit
`21446f1ede6d5284e981bd7b47f432b994e6d812` (MIT).

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
The Apache license is included in `licenses/ms-human-700-APACHE-2.0.txt` and distributed with the package.
