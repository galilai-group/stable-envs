# Quickstart

Take one environment from construction to close: its state, frames, steps, save and restore, and variation. Each code block continues from the one before it, and everything runs on CPU with the core install (`pip install stable-envs`).

## Environment

Every environment is a class on `snv`, and `num_envs` is how many copies run side by side, so think of it as the batch size:

```python
import stable_envs as snv

env = snv.CartPole(num_envs=4, device="cpu")
print(env.action_spec)  # Spec(shape=(1,), dtype=int64, n=2)
print(env.frame_spec)   # Spec(shape=(400, 600, 3), dtype=uint8, bounds=[0, 255])
```

Every tensor the environment returns lives on `device`. The [specs](terms.md#spec) describe one environment, without the batch axis. `snv.make("CartPole", num_envs=4)` builds the same thing from a string, and `snv.list_envs()` lists the classes that registered on your interpreter. [Env](api/env.md#construction) has the constructor arguments every class shares and the default device of each class.

## State

`reset(seed=...)` starts every environment and returns the [state](terms.md#state), a `tensordict.TensorDict` with one row per environment:

```python
state = env.reset(seed=0)
print(state.batch_size)  # torch.Size([4])
print(list(state.keys()))
# ['x', 'x_dot', 'theta', 'theta_dot', 'lifecycle', 'task', 'rng', 'physics']
print(state["x"].shape)  # torch.Size([4, 1])
```

The state also holds things an agent shouldn't see, like TwoRoom's invisible target, so don't feed it to a model as-is. Picking what a model sees is stable-worldmodel's job. The keys differ from family to family, and `env.state_spec` lists them.

## Render a frame

`render` draws one [frame](terms.md#frame) for each row of the state you pass:

```python
frame = env.render(state)
print(frame.shape, frame.dtype)  # torch.Size([4, 400, 600, 3]) torch.uint8
```

Rendering doesn't advance the simulation, so you can render a state you stored long ago. Classic control renders only at Gymnasium's native size (400 × 600 for CartPole), and most other families take a `size` argument when you construct them. To look at a frame, save it as an image:

```python
# needs Pillow: pip install pillow
from PIL import Image

Image.fromarray(frame[0].cpu().numpy()).save("cartpole.png")
```

## Step

`step` takes one action per environment, shaped `(B, A)`, or a sequence of actions shaped `(B, T, A)`:

```python
import torch

action = env.action_spec.sample(4, env.rng)
out = env.step(action)
print(out["reward"])        # tensor([1., 1., 1., 1.], dtype=torch.float64)
print(out["frames"].shape)  # torch.Size([4, 1, 400, 600, 3])

actions = torch.ones(4, 3, 1, dtype=torch.int64)  # three steps of "push right"
out = env.step(actions, frames=None)
print(out["states"].batch_size)  # torch.Size([4, 3])
print("frames" in out.keys())    # False
```

The [step result](terms.md#step-result) holds the final `state`, the task outputs such as `reward` and `terminated`, and the states and frames you pick with the `states` and `frames` arguments. By default you get every state and only the last frame, `None` turns either one off, and both keep a time axis even for a single action. `env.action_spec.sample` draws with `env.rng`, the generator that `reset` seeds. [Step outputs](api/env.md#step-outputs) lists every key and what happens when an episode ends inside a sequence.

## Save and restore state

`get_state` returns an independent copy of the full state, and `set_state` puts it back exactly:

```python
saved = env.get_state()
first = env.step(action)
env.set_state(saved)
again = env.step(action)
print(bool((first["state"] == again["state"]).all()))  # True
```

Stepping from a restored state repeats the same trajectory, which is what the `True` checks. `set_state` needs a full state with exactly `num_envs` rows, while most classes' `render` takes any batch size.

## Variation

Pass a [variation](terms.md#variation) to `render` to change the look, or name its leaves in the reset options to sample them for each environment:

```python
red = env.render(state, variation={"pole.color": [220, 40, 40]})
print(bool((red != frame).any()))  # True
print(env.variation_dim)           # 22

state = env.reset(seed=1, options={"variation": ["physics.gravity"]})
print(state["physics", "gravity"].shape)  # torch.Size([4, 1]), one draw per environment
```

[Randomize appearance and physics](tutorials/variation.md) covers sampling, installing and storing variation.

## Close

```python
env.close()
print(env.closed)  # True
```

Close an environment when you're done, either with a `with` block or by calling `close()`. Calling it twice is harmless. Don't count on garbage collection, because worker processes and GPU buffers stay alive until you close. After that, `reset`, `step`, `get_state`, `set_state` and `render` raise `RuntimeError` instead of quietly starting over.

On Linux, some families step large batches in worker processes, which start with `spawn`. In scripts, put environment construction and the main loop in a function that runs under `if __name__ == "__main__":`.

## Next steps

The tutorials continue from here, in this order:

1. [Store states, render later](tutorials/store-and-render.md): collect states without frames, save them, and render them while you train.
2. [Randomize appearance and physics](tutorials/variation.md): give every batch row its own look and its own physics.
3. [Write your own environment](tutorials/custom-env.md): subclass `snv.Env`, register it, and check that save and restore are exact.

[Environments](environments.md) shows every class, and [API](api/index.md) covers every function and method.
