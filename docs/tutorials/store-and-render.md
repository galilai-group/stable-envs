# Store states, render later

A [state](../terms.md#state) is small and a [frame](../terms.md#frame) is big, so you can collect and store states and render the frames only when a training step needs them. This tutorial collects states, saves them, renders them in a fresh environment under any [variation](../terms.md#variation), resumes from one, and builds a model batch.

## Prerequisites

You need the core install (`pip install stable-envs`) and a CPU. Each code block continues from the one before it, so the page runs top to bottom as one script. It writes one file, `/tmp/tworoom.pt`.

## Collect states without rendering

`step(actions, frames=None)` runs every action and returns all the states without drawing a single frame:

```python
import numpy as np
import torch
import stable_envs as snv

env = snv.TwoRoom(num_envs=4, device="cpu")
env.reset(seed=0)
actions = torch.rand(4, 32, 2) * 2 - 1  # (B, T, 2) in [-1, 1]
out = env.step(actions, frames=None)    # States only, nothing rendered
print(out["states"].batch_size)         # torch.Size([4, 32])
```

Physics still runs for every action, and no frame is drawn. If a [batch row](../terms.md#batch-row) ends inside the sequence, its later states repeat the last one and `out["state_valid"]` marks them false, so keep that mask with the states.

## Save and load

A state is a `TensorDict`, so `torch.save` and `torch.load` store it as it is:

```python
torch.save({"states": out["states"], "valid": out["state_valid"], "actions": actions}, "/tmp/tworoom.pt")
episodes = torch.load("/tmp/tworoom.pt")
print(episodes["states"].batch_size)  # torch.Size([4, 32])
```

`TensorDict.memmap` writes it as memory-mapped files if your dataset doesn't fit in memory.

## Render stored states

`render(states)` draws a state with a one-dimensional batch of any size and doesn't touch the live simulation, so you don't need `set_state` first. Flatten `(B, T)` states with `reshape(-1)`, render, and reshape the frames back:

```python
viewer = snv.TwoRoom(num_envs=1, device="cpu")
frames = viewer.render(episodes["states"].reshape(-1))  # 128 states in one call
frames = frames.reshape(4, 32, *frames.shape[1:])
print(frames.shape, frames.dtype)  # torch.Size([4, 32, 224, 224, 3]) torch.uint8
```

A few classes need exactly `num_envs` rows, which [Env](../api/env.md#rendering) lists.

## Store the variation too

A frame depends on the state and the variation, so store the variation rows you collected under and pass them back when you render. Each stored state needs the row of the batch row it came from, which `repeat_interleave` gives you after flattening:

```python
looks = [name for name in env.variation_names if name not in env.dynamics_variation_leaves]
env.set_variation(env.sample_variation(4, paths=looks, rng=np.random.default_rng(0)))
env.reset(seed=0)
out = env.step(torch.rand(4, 8, 2) * 2 - 1, frames=None)

states = out["states"].reshape(-1)                         # 32 states
rows = env.get_variation().repeat_interleave(8, dim=0)     # (32, variation_dim)
frames = viewer.render(states, variation=rows)
print(frames.shape)                                        # torch.Size([32, 224, 224, 3])
print(bool((frames != viewer.render(states)).any()))       # True, the default look differs
```

Rendering the same states under other rows gives new pictures of one trajectory, which [Randomize appearance and physics](variation.md) covers.

## Resume from a stored state

Rendering is enough for training on pixels, but to continue an episode you need `set_state`. It takes a state with exactly `num_envs` batch rows, under a compatible variation, because a state doesn't carry every configuration value:

```python
midpoint = out["states"][:, 3]  # the state after 4 actions, batch (4,)
env.set_state(midpoint)
branch = env.step(torch.zeros(4, 2))
print(branch["state"]["agent", "position"].shape)  # torch.Size([4, 2])
```

To branch several futures from one state, restore it into a second environment and step that one.

## Hand a rollout to a model

stable-envs stops at frames and states, and you pick the [observation](../terms.md#observation) a model sees. Here the batch keeps the pixels, the agent position and the actions, and leaves out `state["target", "position"]` on purpose: TwoRoom never draws its target, so its position is privileged information that a pixel model doesn't see.

```python
start = env.reset(seed=0)
actions = torch.rand(4, 16, 2) * 2 - 1
out = env.step(actions, frames=slice(None))  # every state and frame
pixels = torch.cat([env.render(start)[:, None], out["frames"]], dim=1)
agent = torch.cat([start["agent", "position"][:, None], out["states"]["agent", "position"]], dim=1)
first = torch.full((4, 1, 2), float("nan"))  # no action comes before reset
batch = {"pixels": pixels.permute(0, 1, 4, 2, 3).float() / 255, "proprio": agent, "action": torch.cat([first, actions], dim=1)}
print(batch["pixels"].shape)   # torch.Size([4, 17, 3, 224, 224])
print(batch["proprio"].shape)  # torch.Size([4, 17, 2])
print(batch["action"].shape)   # torch.Size([4, 17, 2])
env.close()
viewer.close()
```

stable-worldmodel's models read dicts like this one. In stable-worldmodel 0.1.1, `LeWM.encode` takes `pixels` as `(B, T, C, H, W)` floats and `action` as `(B, T, A)`. The first action is NaN, as in stable-worldmodel datasets, so replace it before you compute a loss, for example with `torch.nan_to_num(batch["action"], 0.0)`.

## Checklist

- Store the variation rows with the states. `render(states)` without `variation=` draws the default look.
- Keep `state_valid` with the states. After a batch row ends, its later states repeat the last one.
- Flatten `(B, T)` states with `reshape(-1)` for `render`. `set_state` needs exactly `num_envs` batch rows, so pass one time step, such as `states[:, 3]`.
- Restore a state under the variation it was collected with. A state doesn't carry every configuration value.

Next: [Randomize appearance and physics](variation.md) gives every batch row its own look and physics.
