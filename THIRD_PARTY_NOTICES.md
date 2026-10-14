# Third-party notices

## Craftax textures

`stable_envs/envs/craftax/_textures.py` and the texture tables in
`stable_envs/envs/craftax/classic.py` hold texture data baked from
[Craftax](https://github.com/MichaelTMatthews/Craftax) 1.6.1, which is MIT-licensed
("Copyright (c) 2024 Michael Matthews"). Craftax's Classic texture images are the
image files of [Crafter](https://github.com/danijar/crafter), also MIT-licensed
("Copyright (c) 2021 Danijar Hafner"). The data is resized and alpha-thresholded
as Craftax does at load time; it is otherwise unchanged. The MIT permission notice
at the end of this file applies to it.

## SDL_gfx drawing rules

`stable_envs/render/gfxdraw/` reimplements in Torch the integer rasterization and
packed-channel alpha-blending rules of SDL_gfx by Andreas Schiffler, which pygame's
`gfxdraw` module uses. Upstream [SDL2_gfx](https://www.ferzkopp.net/wordpress/2016/01/02/sdl_gfx-sdl2_gfx/)
is licensed under the zlib license; the copy bundled in pygame (`src_c/SDL_gfx/`) is
marked LGPL. No SDL_gfx source is included.

## Migrated environment tasks

The private Rocket task and Variation helpers in
`stable_envs/envs/rocket/simulation/task.py` and `spaces.py` are adapted from
stable-worldmodel, commit `21446f1ede6d5284e981bd7b47f432b994e6d812`:

- `stable_worldmodel/envs/rocket_landing/pyflyt_rocketlanding.py`
- the Rocket-used Box, RGBBox and Dict behaviors in `stable_worldmodel/spaces.py`

The pinned project is MIT-licensed (its `LICENSE` reads "Copyright (c) 2026
GalilAI-group"), and its `pyproject.toml` lists Lucas Maes, Quentin Le Lidec and
Randall Balestriero as authors. The task source SHA256 is
`5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23`.
Source: https://github.com/galilai-group/stable-worldmodel/tree/21446f1ede6d5284e981bd7b47f432b994e6d812

The adaptation removes an unused URDF helper and adds reset-anchor hooks,
local source-compatible Variation helpers and temporary-file cleanup. It does
not contain the PyFlyt physics engine or PyBullet native implementation. Those
remain external dependencies under their own licenses (PyFlyt MIT; Bullet zlib).

The private PushT task modules `task.py`, `spaces.py` and `quantizer.py`
under `stable_envs/envs/pusht/simulation/`, and `stable_envs/envs/pusht/draw.py`, are
adapted from the same pinned project and authors. Their sources are
`stable_worldmodel/envs/pusht/env.py`, `env_discrete.py`,
`stable_worldmodel/spaces.py`, `stable_worldmodel/envs/utils.py`, and
`stable_worldmodel/envs/quantizer.py`. Source hashes are recorded in the module
headers.
The adaptation replaces package imports with private local imports, combines
the continuous/discrete task definitions, omits the unused source
MultiDiscrete Variation class, and reads the finished canvas with
`pygame.image.tobytes`, which gives the same pixels without a strided copy.
Pymunk/Munk2D remains an external dependency; no native physics implementation
is copied here. This private task migration
does not establish acceptance of the public PushT registrations.

MIT permission notice for the adapted code and the Craftax textures follows:

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
