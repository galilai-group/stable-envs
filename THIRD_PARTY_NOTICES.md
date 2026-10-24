# Third-party notices

## envX TwoRoom

`stable_envs/envs/navigation/tworoom_jax.py` adapts the collision rules and state-step/observation
logic in `envx/tworooms/geometry.py` and `envx/tworooms/environment.py` from
[fracapuano/envX](https://github.com/fracapuano/envX) at commit
`d26c817c1124229c819be574f137cd68bda4380a` (Copyright 2026 Francesco Capuano, Apache-2.0).
Modifications support stable-envs' batched Variation, masked steps, reset/State/task contracts,
boundary rounding and DLPack exchange with Torch. envX is not a runtime dependency.
The Apache license is included in `licenses/envX-APACHE-2.0.txt` and distributed with the package.

## envX Reacher

`stable_envs/envs/dmcontrol/simulation/reacher_mjx.py` adapts the mocap target and MJX
transition/observation logic in `src/envx/reacher/env.py` from the same envX commit above.
Modifications preserve stable-envs' two-substep actions, native reset sampling, full State
and physics Variation. The same Apache-2.0 notice and license apply. envX is not a runtime dependency.

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
