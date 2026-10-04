# State, RenderState, Action, Variation, and Frame

| Value | Meaning |
|---|---|
| **Env** | Batched simulator/task and rendering interface |
| **State** | Independent TensorDict checkpoint of all future-relevant simulator, task, RNG and lifecycle fields under the declared persistent configuration |
| **RenderState** | Private projection sufficient for rendering; may omit dynamics fields |
| **Action** | Intervention at a timestep, with source-defined dtype/bounds/scaling |
| **Variation** | Bounded persistent or reset configuration: physics, geometry, task or appearance |
| **Frame** | Rendered `(B,H,W,3)` uint8 pixels |
| **Learner observation** | Downstream selection/packaging of frames, proprioception, goals or other inputs in stable-worldmodel |

`snv.State` and `snv.StepResult` are aliases for `tensordict.TensorDict`.
State may contain privileged information. It is not necessarily the learner's
observation and must not be exposed to learners by default.

```python
import numpy as np
import stable_envs as snv

with snv.CartPole(num_envs=2, device="cpu") as env:
    state = env.reset(seed=17)
    assert state["x"].shape == (2, 1)
    assert state["lifecycle", "elapsed_steps"].shape == (2, 1)
    out = env.step(np.array([[0], [1]], dtype=np.int64))
    frame = env.render(out["state"])
    assert env.set_state(state) is None
```

`get_state()` and its `state` alias return independent snapshots. `set_state()`
restores a validated full State and returns `None`. State keys/dtypes are
family-specific and described by `state_spec`; persistent configuration must be
compatible between the saved and receiving instances.

## PushT checkpoints

`PushT` and `PushTDiscrete` use complete numeric replay States. Save all fields,
including the reset recipe, RNGs and action history. Restoring positions alone
cannot reconstruct the physics engine's contact history. `set_state()` rebuilds
that history, so its cost grows with the episode prefix. Ordinary `step()` does
not restore or replay earlier steps. The history axis grows; `state_spec`
describes the current checkpoint's shape rather than imposing an episode limit.

```python
with snv.PushT(num_envs=2, device="cpu", size=96) as env:
    checkpoint = env.reset(seed=17)
    actions = np.zeros((2, 8, 2), dtype=np.float32)
    first = env.step(actions, frames=None)
    env.set_state(checkpoint)
    repeated = env.step(actions, frames=None)
    assert (first["state"] == repeated["state"]).all()
```

Continuous actions are relative controls in `[-1, 1]`. Discrete actions have
shape `[B,2]` or `[B,T,2]`, with each polar-action index in `[0,16)`. This replaces
the earlier approximate public PushT solver's absolute pixel-target actions.
Old five-coordinate renderer datasets are not complete simulator checkpoints.

Rendering a saved State uses a separate scene and its numeric contact cache;
it neither restores nor advances live physics. The current public PushT
renderer executes on CPU, including when tensors are stored on CUDA.
`render_action=True` requires a resolution of at least 96, matching the source
renderer's supported marker thickness.

## Compact RenderState is private

Renderer implementations use `_pack_render_state`, `_unpack_render_state`,
`_get_render_state`, `_set_render_state` and `_render`. Compact rows are often
float32 and retain frozen layouts. Decoding a lossy row does not invent omitted
velocities, lifecycle or RNG fields. Such rows and partial dictionaries must be
refused by `set_state()`.

Applications render full snapshots with `env.render(state)`. The old public
`pack_render_state`/`unpack_render_state` names are not supported. Contributor
examples for testing the private boundary are in the [renderer guide](../adding/renderer.md).
Pixel equality is not proof of exact future simulation.

## Specifications and results

`state_spec` is a CompositeSpec mirroring State fields. `action_spec`, `frame_spec`
and `variation_spec` describe per-environment values without the batch dimension.
A Spec carries shape, dtype, device and applicable bounds/names. It is descriptive;
the backend must still validate source-specific semantic constraints.

`step(actions)` returns a StepResult containing `state` and the environment's
defined reward, success and termination/truncation information. Existing families
may carry additional reference diagnostic fields; those are not a learner
observation-selection API. Renderer-only cells do not fabricate simulator results.

Variation includes physics/task/reset leaves as well as appearance. See
[Variation](variation-spaces.md), [dynamics](dynamics.md) and the
[contributor layer overview](../adding/index.md).
