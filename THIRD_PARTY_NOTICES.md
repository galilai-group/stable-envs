# Third-party notices

## MuJoCo Warp renderer specialization

`stable_envs/render/mjwarp/lightspec.py` contains modified code from
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), version 3.11.0,
source path `mujoco_warp/_src/render.py`.

Copyright 2026 The Newton Developers. Licensed under the Apache License, Version
2.0; the full text is included in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
The modification removes shadow traversals at compile time for lights that cannot
cast shadows, retaining upstream light evaluation order and arithmetic.

## Migrated environment tasks

The private Rocket task and Variation helpers in
`stable_envs/_adapters/rocket_task.py` and `rocket_spaces.py` are adapted from
stable-worldmodel, commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`:

- `stable_worldmodel/envs/rocket_landing/pyflyt_rocketlanding.py`
- the Rocket-used Box, RGBBox and Dict behaviors in `stable_worldmodel/spaces.py`

The pinned project's `pyproject.toml` declares the MIT license and lists Lucas
Maes, Quentin Le Lidec and Randall Balestriero as authors. That source tree has
no separate LICENSE file. The task source SHA256 is
`5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23`.
Source: https://github.com/galilai-group/stable-worldmodel/tree/176d54532b3c0a251a4f509b74f2fdacc143a63d

The adaptation removes an unused URDF helper and adds reset-anchor hooks,
local source-compatible Variation helpers and temporary-file cleanup. It does
not contain the PyFlyt physics engine or PyBullet native implementation. Those
remain external dependencies under their own licenses (PyFlyt MIT; Bullet zlib).

The private PushT task modules `pusht_task.py`, `pusht_spaces.py`,
`pusht_draw.py`, and `pusht_quantizer.py` under `stable_envs/_adapters/` are
adapted from the same pinned project and authors. Their sources are
`stable_worldmodel/envs/pusht/env.py`, `env_discrete.py`,
`stable_worldmodel/spaces.py`, `stable_worldmodel/envs/utils.py`, and
`stable_worldmodel/envs/quantizer.py`. Source hashes are recorded in the module
headers.
The adaptation replaces package imports with private local imports, combines
the continuous/discrete task definitions, and omits the unused source
MultiDiscrete Variation class. Pymunk/Munk2D remains an external dependency;
no native physics implementation is copied here. This private task migration
does not establish acceptance of the public PushT registrations.

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
