# Three ways to get a frame: 6 envs

Device: NVIDIA H200 · CPU: INTEL(R) XEON(R) PLATINUM 8558 · 16 cores visible · load [6.701171875, 6.8193359375]

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
| MiniGridEmpty | 256x256 | 1 | — | 1,474,043 (4,096) | 100% | 3,326 | **443.2x** | 268 | 6,311 |
| PushT | 96x96 | 1 | — | 1,336,179 (4,096) | 100% | 11,154 | **119.8x** | 20 | 2,355 |
| PointmazeMediumNavigate | 224x224 | 2 | — | 263,446 (4,096) | 100% | 2,744 | **96.0x** | 8 | 7,970 |
| DMCWalkerWalk | 84x84 | 1 | — | 130,060 (4,096) | 100% | 8,356 | **15.6x** | 36 | 3,141 |
| MetaWorldAssembly | 480x480 | 1 | — | 2,889 (4,096) | 100% | 798 | **3.6x** | 80 | 35,052 |
| CarRacing | 96x96 | 1 | — | 27,221 (4,096) | 100% | 9,276 | **2.9x** | 112 | 2,565 |

## Load-and-decode (warm local file) against decode-from-memory

| env | memory f/s | file f/s | file/memory |
|---|---|---|---|
| MiniGridEmpty | 3,326 | 2,454 | 0.738 |
| PushT | 11,154 | 9,973 | 0.894 |
| PointmazeMediumNavigate | 2,744 | 2,634 | 0.960 |
| DMCWalkerWalk | 8,356 | 9,311 | 1.114 |
| MetaWorldAssembly | 798 | 779 | 0.977 |
| CarRacing | 9,276 | 8,391 | 0.905 |

## Modules absent in this interpreter

* `overcooked` — ImportError: jaxmarl is not importable (ModuleNotFoundError: No module named 'jaxopt'). The atlas IS jaxmarl's own `_render_tile` output — nothing here re-derives it — so there is no degraded mode. `pip install jaxmarl` (measured against jaxmarl 0.1.0 / jax 0.4.38) INTO A CLEAN VENV: a `python -m venv --system-site-packages` venv breaks jax's CUDA plugin discovery.