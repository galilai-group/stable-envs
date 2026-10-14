# Gallery

Frames straight from the renderers, at native resolution. Nothing here is upscaled except the videos, and
those use nearest-neighbour, so the pixels on screen are the pixels the renderer produced.

## PushT

State: a complete replay State (234 numeric leaves: the current bodies and goal, the action history
and the Variation it replays from), not a pose vector. See [State](concepts/state.md#pusht-checkpoints).
Reference: `gym_pusht/PushT-v0`.

The released PushT draws its frames with the source's own pygame code on CPU, so there is no second
renderer to put beside it: a frame is the source frame for the same State. `device="cuda"` places the
returned tensors on the GPU and leaves the drawing on CPU. `contrib/pusht_torch/` holds an unreleased
batched torch renderer; see [Benchmarks](benchmarks.md#pusht) for its agreement and throughput.

### States from the reset distribution

24 lanes after `reset(seed=0)`, rendered at 96 px:

![24 PushT reset states](assets/media/PushT/grid.png)

### Appearance variation

The same 24 lanes with the four colour leaves (`agent`, `background`, `block`, `goal`) sampled
independently per lane through `set_variation` before the reset:

![Per-lane colour variation](assets/media/PushT/variation_batch.png)

## Reproducing

The PushT images come from the released env, on CPU:

```python
import numpy as np
import stable_envs as snv

with snv.PushT(num_envs=24, device="cpu", size=96) as env:
    env.reset(seed=0)
    grid = env.render()
    colors = [p for p in env.variation_names if p.endswith(".color")]
    env.set_variation(env.sample_variation(paths=colors, rng=np.random.default_rng(0)))
    env.reset(seed=0)
    varied = env.render()
```
