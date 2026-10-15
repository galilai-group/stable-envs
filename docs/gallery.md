# Gallery

This page shows frames from the renderers at native resolution, with no upscaling.

## PushT

State: a full replay State (234 numeric leaves: bodies, goal, action history and the Variation it
replays from). See [State](concepts/state.md#pusht-checkpoints). Reference: `gym_pusht/PushT-v0`.

PushT draws its frames with the source's pygame code on CPU, so a frame is the source frame for the
same State. `device="cuda"` places the output tensors on the GPU.

### Reset states

24 lanes after `reset(seed=0)`, at 96 px:

![24 PushT reset states](assets/media/PushT/grid.png)

### Colour variation

The same 24 lanes with the colour leaves (`agent`, `background`, `block`, `goal`) sampled per lane
with `set_variation` before the reset:

![Per-lane colour variation](assets/media/PushT/variation_batch.png)

## Reproduce

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
