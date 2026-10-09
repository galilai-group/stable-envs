# PushT torch renderer (not released)

`TorchRenderer` renders the public `PushT` / `PushTDiscrete` State as a batch on any torch device.
Its frames are equal, byte for byte, to `env.render(state)` (the source's pygame renderer).

```python
import stable_envs as snv
from contrib.pusht_torch import TorchRenderer

env = snv.PushT(num_envs=1024)
state = env.reset(seed=0)
frames = TorchRenderer("cuda").render(env, state)  # (1024, 224, 224, 3) uint8 on CUDA
```

## How it works

1. `display_list.py` computes each frame's draw calls from the render rows with the source's
   arithmetic: Chipmunk body transforms and vertices, `round` in `to_pygame`, `light_color`, the
   fat-segment outline, the goal outline and the contact segments. Body geometry and draw order
   come once per geometry from a pymunk scene.
2. `pgdraw.py` turns pygame's filled polygons, filled circles and thick lines into per-row spans in
   batched torch, paints them in draw order, and blends antialiased contact lines.
3. `cvresize.py` applies OpenCV's fixed-point `INTER_LINEAR` 512-to-`size` resize.
4. Action markers (`render_action=True`) are drawn with `cv2.drawMarker` on CPU.

## Checked

- Primitives against pygame 2.6.1 on random inputs, including clipped lines and antialiased lines
  over random backgrounds (`contrib/tests/unit/test_pusht_torch.py`).
- Frames against `env.render` for both classes, with and without targets and action markers,
  default and sampled Variations, and contacts: CPU at 96 px on macOS; CUDA at 224 and 96 px on an
  RTX A5000 (`contrib/tests/gpu/test_pusht_torch_cuda.py`).

## Limits

- The reference is pygame 2.6.1 (not pygame-ce) and OpenCV's generic resize path. Linux x86-64
  OpenCV wheels take that path. ARM builds dispatch some sizes (224 among them) to a HAL and can differ
  by a few levels there.
- Per-frame Python work remains (row extraction, display-list assembly), so small batches gain
  little. See the measurements below.

## Measurements

One RTX A5000 node (Oscar `gpu2603`), one CPU thread, 224 px, States from four episodes. Both
columns are State to frames, `env.render` on CPU and `TorchRenderer` on CUDA:

| batch | `env.render` ms/frame | `TorchRenderer` ms/frame | speedup |
|---:|---:|---:|---:|
| 1 | 20.8 | 30.7 | 0.7x |
| 16 | 5.39 | 2.24 | 2.4x |
| 64 | 4.69 | 0.88 | 5.3x |
| 256 | 4.54 | 0.56 | 8.1x |
| 1,024 | 4.47 | 0.45 | 10.0x |
| 2,048 | 4.51 | 0.45 | 10.0x |

The `env.render` column depends on the node's CPU; another A5000 node measured 2.8 ms/frame for it.

## Licensing

`pgdraw.py` reproduces pygame's drawing code and is distributed under the GNU LGPL 2.1 or later,
like pygame. `cvresize.py` reproduces OpenCV's resize arithmetic (Apache-2.0). This is one reason the
renderer is not in the MIT-licensed package.
