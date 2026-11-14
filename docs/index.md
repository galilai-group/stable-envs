---
title: Stable-Envs
summary: Fast and Reproducible Environments with Controllable Variations
sidebar_title: Home
---

stable-envs is a library of batched simulation environments that keep the full simulator state, so you can restore any step exactly and render images from it later, on CPU or GPU.

Store the [state](terms.md#state), replay or branch from any step, and draw [frames](terms.md#frame) from it later. 80 environment classes in 11 families share the same API. Policies, datasets and training live in [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), which builds on stable-envs.

## Install

=== "uv"

    ```bash
    uv add stable-envs
    ```

=== "pip"

    ```bash
    pip install stable-envs
    ```

The core install runs CartPole, Acrobot and most navigation classes. [Install](install.md) lists the extras for the other families.

## Example

```python
import torch
import stable_envs as snv

with snv.CartPole(num_envs=8, device="cpu") as env:
    env.reset(seed=0)
    actions = torch.randint(0, 2, (8, 10, 1))  # (B, T, A): ten steps for each environment
    out = env.step(actions, frames=None)  # step without rendering
    print(out["states"].batch_size)  # torch.Size([8, 10])
    frames = env.render(out["states"][:, 4])  # render the state after five steps
    print(frames.shape, frames.dtype)  # torch.Size([8, 400, 600, 3]) torch.uint8
```

`step` returns the state after every step, and `render` draws a batch of frames from any of them.

## Next steps

| If you want to | Read |
|---|---|
| run your first environment | [Quickstart](quickstart.md) |
| install the extras for a family, or set up a GPU | [Install](install.md) |
| store states in a dataset and render them while you train | [Store states, render later](tutorials/store-and-render.md) |
| randomize colors, lighting or physics across a batch | [Randomize appearance and physics](tutorials/variation.md) |
| write an environment of your own | [Write your own environment](tutorials/custom-env.md) |
| find an environment and see what it looks like | [Environments](environments.md) |
| look up a function or class | [API](api/index.md) |
