# PushT torch renderer (not released)

`TorchRenderer` renders the public `PushT` / `PushTDiscrete` State as a batch on any torch device.
Its frames are byte-for-byte equal to `env.render(state)`, which uses the source's pygame renderer.

```python
import stable_envs as snv
from contrib.pusht_torch import TorchRenderer

env = snv.PushT(num_envs=1024)
state = env.reset(seed=0)
frames = TorchRenderer("cuda").render(env, state)  # (1024, 224, 224, 3) uint8 on CUDA
```

## How it works

- `display_list.py` computes each frame's draw calls with the source's arithmetic.
- `pgdraw.py` reproduces pygame's polygon, circle and line fills in batched torch.
- `cvresize.py` reproduces OpenCV's `INTER_LINEAR` resize from 512 px. Action markers are drawn with
  OpenCV on CPU.

## Tests

- Primitives against pygame 2.6.1: `contrib/tests/unit/test_pusht_torch.py`.
- Frames against `env.render` for both classes, with targets, action markers, sampled Variations and
  contacts: `contrib/tests/gpu/test_pusht_torch_cuda.py`.

## Limits

- The reference is pygame 2.6.1 (not pygame-ce) and OpenCV's generic resize. Linux x86-64 OpenCV
  wheels use that path. ARM builds can route some sizes (224 among them) elsewhere and differ by a few
  levels.
- Per-frame Python work remains, so small batches gain little.

## Measurements

One RTX A5000 node, one CPU thread, 224 px, States from four episodes. `env.render` runs on that
node's CPU; `TorchRenderer` runs on CUDA.

| batch | `env.render` ms/frame | `TorchRenderer` ms/frame | speedup |
|---:|---:|---:|---:|
| 1 | 12.7 | 20.3 | 0.6x |
| 16 | 2.24 | 1.52 | 1.5x |
| 64 | 1.72 | 0.65 | 2.6x |
| 256 | 1.58 | 0.42 | 3.7x |
| 1,024 | 1.59 | 0.32 | 5.0x |
| 2,048 | 1.63 | 0.31 | 5.2x |

`env.render` speed depends on the CPU. See `docs/benchmarks.md` for other CPUs.

## License

`pgdraw.py` reproduces pygame's drawing code and is under the GNU LGPL 2.1 or later. `cvresize.py`
reproduces OpenCV's resize arithmetic (Apache-2.0). For this reason the renderer is not in the
MIT-licensed package.
