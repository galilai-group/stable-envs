# State, Action, Variation, and Frame

Five nouns, two usage modes.

| noun | what it is |
|---|---|
| **Env** | a batched environment class (`snv.PushT`, …); owns the relationship among the others |
| **State** | instantaneous dynamic world state: a `tensordict.TensorDict` grouped by entity |
| **Action** | intervention at a timestep |
| **Variation** | persistent / episode-level environment configuration (appearance, geometry, morphology, physics, reset distributions, task configuration) |
| **Frame** | rendered observation: `(B, H, W, 3)` uint8 from `render` |

Two ways to use an Env:

```python
import stable_envs as snv

with snv.PushT(num_envs=64) as env:          # device="auto" → CUDA if available, else CPU
    state = env.reset(seed=0)                # State
    action = env.action_spec.sample(64, env.rng)

    out = env.step(action)                   # STATEFUL: writes env.state
    state = out["state"]
    nxt = env.transition(state, action)      # FUNCTIONAL: env.state is untouched
    traj = env.unroll(state, action[None])   # FUNCTIONAL, known actions, time first
    frame = env.render(state)                # Frame
```

`reset` / `step` return State-oriented values. Pixels come from `render`. There is no Gym-shaped "observation" on `step`.

## State

`snv.State` is an alias for `tensordict.TensorDict`, not a subclass.

```python
state = env.reset(seed=0)          # batch size [num_envs]
state["agent", "position"]         # PushT: (num_envs, 2)
state["agent", "velocity"]         # dynamics-only; may be absent from the render row
state["block", "angle"]            # (num_envs, 1)
```

Trivial cells may expose a single key (`{"x": (d,)}`) rather than an entity tree.

## Compact render state is not the full State

Every renderer and frozen corpus speaks a compact float32 row used to **regenerate pixels**. That row is *render state*. It is not always a lossless / Markov dump of public State — dynamics-only fields can be omitted.

```python
row = env.pack_render_state(state)     # (n, render_state_dim) float32
state = env.unpack_render_state(row)   # entity-grouped view; omitted fields come back as zeros
env.render_state_dim
env.render_state_names                 # column labels of that row
```

PushT's agent velocity is public State (`state["agent", "velocity"]`) and is **dropped** by `pack_render_state`, because the frozen row never carried it. That is why `transition` can be a pure function of the State it is given.

`unpack_render_state` **copies**. Mutating `state["agent", "position"]` does not silently change the next `step`.

## Frame

```python
frame = env.render(state)          # (num_envs, H, W, 3) uint8 on env.device
env.frame_spec                     # (H, W, 3) uint8 — what render produces
```

`frame_spec` is not what `reset` / `step` return. Those carry State.

## Variation

```python
env.variation_spec
env.variation_dim, env.variation_names
env.default_variation()
env.sample_variation(env.num_envs)
frame = env.render(state, variation={"block.color": [220, 40, 40]})
```

Defaults are the official configuration. Domain randomisation is a tensor, not a loop. See [Variation](variation-spaces.md).

## Specs

One `Spec` dataclass (`shape`, `dtype`, `low`, `high`, `device`, `names`, `discrete_n`) plus `CompositeSpec` as a nested dict of Specs.

```python
env.state_spec                     # CompositeSpec mirroring TensorDict keys
env.action_spec                    # per-environment action; renderer-only cells raise
env.frame_spec                     # (H, W, 3) uint8
env.variation_spec                 # flat Variation row (one Spec, not a tree)
```

`snv.Spec` describes tensor contracts. `snv.info("PushT")` is environment **metadata** (family, domain, slug) — it is not a `Spec`.

`shape` is the **per-environment** shape, without the batch axis. `sample(num_envs, rng)` draws uniformly inside the bounds; an unbounded continuous spec refuses rather than inventing a distribution — reachable states come from `env.sample_states`.

## StepResult

`env.step(actions)` returns a TensorDict (`snv.StepResult`, the same type as `State`):

```python
out = env.step(action)
state = out["state"]               # always present
```

`reward` / `terminated` / `truncated` (or a combined `done`) exist **only where that environment defines those semantics**. They are not fabricated for renderer-only cells. PushT currently returns `{"state": …}` only.
