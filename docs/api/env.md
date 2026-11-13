# Env

`snv.Env` is the base class of every environment. `snv.TwoRoom`, `snv.CartPole` and the rest share its shape and step-output conventions and the methods on this page. An environment runs `num_envs` [batch rows](../terms.md#batch-row) on one device: you reset and step it, save and restore its [state](../terms.md#state), set its [variation](../terms.md#variation), and render [frames](../terms.md#frame) from any state you kept.

=== "Step and render"

    ```python
    import torch
    import stable_envs as snv

    env = snv.TwoRoom(num_envs=4, device="cpu")
    state = env.reset(seed=0)
    out = env.step(torch.zeros(4, 8, 2), frames=slice(None))  # 8 steps, every state and frame
    print(out["reward"].shape)      # torch.Size([4, 8])
    print(out["frames"].shape)      # torch.Size([4, 8, 224, 224, 3])
    print(env.render(state).shape)  # torch.Size([4, 224, 224, 3]), a saved state, no stepping
    ```

=== "Variation"

    ```python
    env.set_variation({"agent.color": [255, 0, 0]})  # a red agent from now on
    print(env.get_variation().shape)                 # torch.Size([4, 23])
    rows = torch.as_tensor(env.sample_variation(4))  # one sampled row per batch row
    print(env.render(state, variation=rows).shape)   # torch.Size([4, 224, 224, 3])
    env.close()
    ```

## Shapes

| Call | Takes | Returns |
|---|---|---|
| `reset(seed)` | an int or None | State, a TensorDict with batch size `(B,)` |
| `step(actions)` | `(B, *A)` or `(B, T, *A)` | a [step result](../terms.md#step-result) with batch size `(B,)`, see [Step outputs](#step-outputs) |
| `get_state()`, `state` | nothing | State with batch size `(B,)` |
| `set_state(state)` | State with batch size `(B,)` | None |
| `render(x)` | None for the live state, a state with batch size `(N,)`, or `(N, state_dim)` compact rows | `(N, H, W, 3)` uint8 frames |
| `get_variation()` | nothing | `(B, variation_dim)` float32 |
| `set_variation(v)` | a dict of leaf values, or a `(variation_dim,)` or `(B, variation_dim)` row | `(B, variation_dim)` float32 |
| `sample_variation(n)` | a row count | `(n, variation_dim)` float32 NumPy array |

`snv.State` and `snv.StepResult` are both `tensordict.TensorDict`. `snv.VariationLike` is what `set_variation` and `render(variation=...)` accept: a mapping of leaf paths to values, or a tensor or NumPy row.

## Step outputs

```text
step(actions, *, states=slice(None), frames=slice(-1, None))
```

`actions` is `(B, A)` for one action per batch row or `(B, T, A)` for a sequence, with `T` at least 1. Discrete actions use `A = 1`, and a single discrete action can also be an integer tensor of shape `(B,)`. `step` keeps the dtype you pass. `states` and `frames` are keyword-only Python slices over the states after each action, independent of each other, and `None` turns that history off. The first output is the state after the first action, not the state you started from.

| You want | `states` | `frames` |
|---|---|---|
| all states, final frame (default) | `slice(None)` | `slice(-1, None)` |
| final state and frame only | `None` | `slice(-1, None)` |
| all states, no rendering | `slice(None)` | `None` |
| all states and frames | `slice(None)` | `slice(None)` |
| States at 0, 4, 8... and frames at 3, 7, 11... | `slice(None, None, 4)` | `slice(3, None, 4)` |

Slices follow Python's rules for clipping, negative indices and negative strides. A reverse slice changes the order of the outputs, not the order the simulation runs in. A selector must be a slice or `None`, and an integer raises `TypeError`. `step` checks selectors, strides and action shapes before it moves the simulation. A bad call leaves the environment where it was.

| Key | Shape | Notes |
|---|---|---|
| `state` | batch `(B,)` | always present, the state after the last action |
| `states` | batch `(B, S)` | the `S` selected states, missing when `states=None` |
| `frames` | `(B, F, H, W, 3)` uint8 | the `F` selected frames, missing when `frames=None` |
| `valid` | `(B, T)` | whether each action ran |
| `state_valid`, `frame_valid` | `(B, S)` and `(B, F)` | whether each selected state or frame is real, in output order |

A single action counts as a sequence of length one: `frames` is `(B, 1, H, W, 3)` even then, and `frames[:, 0]` is the default final frame. An empty `frames` slice gives `F = 0` and renders nothing. Task outputs such as `reward` and `terminated` keep the names and dtypes of the [upstream library](../terms.md#upstream-library), and gain a time axis for a sequence. Classes without RGB output need `frames=None`.

Each batch row stops at its first termination or truncation and skips the rest of its actions. The transition that ended it is valid, and its later task outputs are zero. Its later selected states and frames repeat the final one with `state_valid` and `frame_valid` set to false. `step` never resets a batch row. The next `step` call steps every batch row again, and a single action after the end behaves as it does in the upstream library.

Returned states and frames own their memory, and later calls leave them unchanged. A few classes return the same task output tensor at every step, such as `info.goal` of `SimplePointMaze`; treat it as read-only. Selecting fewer frames skips rendering, while physics still runs for every action. Calling `get_state()` right after `step()` copies `result["state"]` again. Send a sequence only when you already know every action in it. To branch from a state, use a second environment with `set_state`.

## Construction

`num_envs` is fixed at construction, and `len(env)` equals it. Families add their own keywords, such as `simulation_backend`, which their [environment pages](../environments.md) describe.

`device` decides where the environment runs and where every returned tensor lives. Its default depends on the class:

| Default `device` | Classes |
|---|---|
| the global `default_device`, which is `"auto"` unless you change it with `snv.set` | classic control, navigation, PushT, Craftax, Miniworld, MS-Human, Roadster |
| `"cuda"` | DM Control, OGBench mazes |
| `"cpu"` | Fetch, `OGBCube`, `OGBScene`, `PyFlytRocketLanding` |

`"auto"` picks CUDA when torch can see a GPU, and the CPU otherwise.

::: stable_envs.env.Env
    options:
      show_root_heading: false
      members: false

These class attributes are what `snv.info` reads without building anything:

| Attribute | Meaning |
|---|---|
| `family`, `domain` | The family ID and the task group inside it, which `snv.list_envs` filters on |
| `name` | The environment slug that `env.id` returns, such as `"two-room"` |
| `reference_id` | The upstream library's id this class is compared with, such as `"CartPole-v1"`, or None |
| `default_size` | Frame size in pixels when `size` is omitted |
| `native_size` | The only `size` that the class draws, or None when the class draws any size |
| `requires_cuda` | True when the class has no CPU path |

## Lifecycle

Closing releases only what that environment owns. Rendering contexts that several environments share in one process stay alive for the life of the process, and the other environments keep working. After `close`, `reset`, `step`, `get_state`, `set_state` and `render` raise `RuntimeError`.

::: stable_envs.env.Env.reset

::: stable_envs.env.Env.step

::: stable_envs.env.Env.close

::: stable_envs.env.Env.closed

## State

`get_state` and the `state` property return a copy of the live state that later steps leave unchanged. `set_state` needs a full state with exactly `num_envs` batch rows, refuses anything else, and returns `None`. The environment has to use a variation compatible with the one the state came from: a state carries only part of the configuration.

::: stable_envs.env.Env.get_state

::: stable_envs.env.Env.state

::: stable_envs.env.Env.set_state

## Specs

Each spec describes one batch row, without the batch axis, using the [spec classes](specs.md).

::: stable_envs.env.Env.state_spec

::: stable_envs.env.Env.action_spec

::: stable_envs.env.Env.frame_spec

::: stable_envs.env.Env.variation_spec

## Variation

`set_variation` installs a variation that stays until you change it. Every later `reset` restores the values you set for each leaf its options skip. Reset options sample or set leaves for one episode only. `variation_space` is the instance's [`stable_envs.spaces.Dict`](specs.md#spaces) of leaves, and `get_variation()` is the compiled form that rendering and physics read. [Randomize appearance and physics](../tutorials/variation.md) walks through both.

::: stable_envs.env.Env.variation_dim

::: stable_envs.env.Env.variation_names

::: stable_envs.env.Env.default_variation

::: stable_envs.env.Env.get_variation

::: stable_envs.env.Env.set_variation

::: stable_envs.env.Env.sample_variation

::: stable_envs.env.Env.variation_values

::: stable_envs.env.Env.draw_reset_values

`dynamics_variation_leaves` is a frozenset of the variation leaves that change dynamics, such as `physics.gravity` on CartPole, and every other leaf changes only the picture. `physics_variation_leaves` is narrower: the leaves the DM Control and Roadster simulators read as physics parameters. It's empty for the other families.

::: stable_envs.env.Env.physics_variation_leaves

## Rendering

`render(x)` is the one public way to draw frames. It does not change the live simulation. `x` is one of these forms:

- None: the frames of the live batch rows.
- A full state from `reset`, `step` or `get_state`, with any batch size `N`. `PyFlytRocketLanding` is the exception: it needs exactly `num_envs` batch rows.
- Compact rows `(N, state_dim)`, NumPy or Torch, as `sample_states` returns them.
- A RenderState from `to_render_state`, for `OGBCube`, `OGBScene`, `PushT`, `PushTDiscrete` and `PyFlytRocketLanding`.

A state draws with its own variation. Compact rows draw with the current variation, or with `variation=` for this call only. `render(state)` checks only the state fields that the frame reads: their presence, shape and finite values. `set_state` checks the full state.

Some classes do not draw compact rows. Classes with `state_dim` 0, such as `PushT`, `MiniWorldMovingBlockWorld` and the Craftax classes, refuse them. `PyFlytRocketLanding` draws rows on CUDA only. `OGBCube`, `OGBScene` and the Roadster classes draw rows on each device. A form that the class does not draw raises `TypeError`, and the message gives the forms that the class accepts.

`native_size` is the only `size` that a class draws, or None when the class draws any size. Classic control, navigation and Craftax set it, because their upstream libraries draw each primitive in fixed pixels. Another `size` raises `ValueError`.

::: stable_envs.env.Env.render

::: stable_envs.env.Env.sample_states

::: stable_envs.env.Env.describe

## Metadata

::: stable_envs.env.Env.num_envs

::: stable_envs.env.Env.has_dynamics

::: stable_envs.env.Env.class_name

::: stable_envs.env.Env.id

::: stable_envs.env.Env.resolve_device

::: stable_envs.env.Env.state_names_of

::: stable_envs.env.Env.render_state_dim

::: stable_envs.env.Env.render_state_names

## Family hooks

A family implements these hooks; see [Adding an environment](../contributing/environment.md#renderer).

::: stable_envs.env.Env.make_variation_space

::: stable_envs.env.Env.scene
