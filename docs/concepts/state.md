# State and specs

The public state describes the **environment**, not a renderer kernel's packed layout.

```python
state = env.reset(seed=0)          # tensordict.TensorDict, batch size [num_envs]
state["agent", "position"]         # PushT: (num_envs, 2)
state["agent", "velocity"]         # hidden dynamics live here so transition can be pure
state["block", "angle"]            # (num_envs, 1)
```

`snv.State` is an alias for `tensordict.TensorDict`, not a subclass. Anything that consumes a TensorDict consumes a stable-envs state.

Trivial cells may expose a single key (`{"x": (d,)}`) rather than an entity tree.

## Packed row vs public State

Every renderer, frozen corpus and certification check still speaks a float32 row of width `state_dim`:

```python
rows = env.pack(state)             # (n, state_dim) float32 — the frozen packed layout
state = env.unpack(rows)           # entity-grouped view; hidden fields the row does not carry come back as zeros
names = env.state_names            # column labels of that packed row
```

`unpack` **copies**. A State never aliases the env buffer, so mutating `state["agent", "position"]` does not silently change the next `step`.

PushT's agent velocity is public state (`state["agent", "velocity"]`) and is **dropped** by `pack`, because the frozen row never carried it. That is why `transition` can be a pure function of the State it is given: as long as one dynamical variable lived in an instance attribute, two calls with the same packed row could disagree.

## Specs

One `Spec` dataclass (`shape`, `dtype`, `low`, `high`, `device`, `names`, `discrete_n`) plus `CompositeSpec` as a nested dict of Specs. There is no torchrl dependency and no class hierarchy per spec kind.

```python
env.state_spec                     # CompositeSpec mirroring TensorDict keys
env.action_spec                    # per-environment action; renderer-only cells raise
env.observation_spec               # (H, W, 3) uint8
env.variation_spec                 # flat style layout (one Spec, not a tree)
```

`shape` is the **per-environment** shape, without the batch axis. `sample(num_envs, rng)` draws uniformly inside the bounds; an unbounded continuous spec refuses rather than inventing a distribution — reachable states come from `env.sample_states`.

`variation_space` remains the gymnasium-style tree the certification harness samples. `variation_spec` is the same layout in the package's own vocabulary.

## Step result

`env.step(actions)` returns a TensorDict (`snv.StepResult`, the same type as `State`) with:

- `"state"` — always
- `"reward"` / `"done"` — only if the cell has those semantics

Pure renderers omit the keys rather than fabricating zeros. PushT currently returns `{"state": …}` only.
