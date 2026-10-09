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

One RTX A5000 node, one CPU thread, 224 px, States from four episodes. Both columns are State to
frames on the same node, `env.render` on CPU and `TorchRenderer` on CUDA:

| batch | `env.render` ms/frame | `TorchRenderer` ms/frame | speedup |
|---:|---:|---:|---:|
| 1 | 12.7 | 20.3 | 0.6x |
| 16 | 2.24 | 1.52 | 1.5x |
| 64 | 1.72 | 0.65 | 2.6x |
| 256 | 1.58 | 0.42 | 3.7x |
| 1,024 | 1.59 | 0.32 | 5.0x |
| 2,048 | 1.63 | 0.31 | 5.2x |

`env.render` depends on the node's CPU. On a Xeon Platinum 8268 core it takes 3.7 to 4.0 ms/frame
(`docs/benchmarks.md`).

## Licensing

`pgdraw.py` reproduces pygame's drawing code and is distributed under the GNU LGPL 2.1 or later,
like pygame. `cvresize.py` reproduces OpenCV's resize arithmetic (Apache-2.0). This is one reason the
renderer is not in the MIT-licensed package.
