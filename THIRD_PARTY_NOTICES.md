# Third-party attribution: Rocket task candidate

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
