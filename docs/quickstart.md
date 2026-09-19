# Quickstart

Install the core, construct an environment by **class name**, and render. Optional extras add families; they are not required to draw PushT.

```bash
pip install -e .
```

```python
import stable_envs as snv

env = snv.PushT(num_envs=64, device="cpu")   # or device="cuda"
state = env.reset(seed=0)
pixels = env.render(state)                   # (64, H, W, 3) uint8
env.close()
```

`num_envs` is how many copies this object holds. `len(env) == env.num_envs`. A short last batch (`n <= num_envs`) is accepted by `set_state` / `render` and draws exactly those rows.

The same object as a context manager — `close()` is idempotent and correctness never depends on `__del__`:

```python
with snv.PushT(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    pixels = env.render(state)
```

## Strings are secondary

```python
env = snv.make("PushT", num_envs=8, device="cpu")
env = snv.make("swm/PushT-v1", num_envs=8, device="cpu")
info = snv.spec("PushT")          # no construction, no GPU
names = snv.list_envs()           # what this interpreter actually imported
```

An unknown name raises with the registered classes (and, for `swm/…` ids, the alias table). Two SWM ids are **symbolic** (`CraftaxClassicSymbolic-v1`, `CraftaxSymbolic-v1`): this package does not pretend to render them.

## Stateful step, functional dynamics

```python
state = env.reset(seed=0)
action = env.action_spec.sample(env.num_envs, env.rng)

out = env.step(action)                    # writes env.state; returns a StepResult TensorDict
nxt = env.transition(state, action)       # does not touch env.state
traj = env.unroll(state, actions)         # actions are (T, num_envs, …), time first
```

`unroll` is open-loop known actions, not a policy rollout. Cells without a batched simulator raise `NotImplementedError` from `step` / `transition` and keep `reset` / `render` working. See [dynamics](concepts/dynamics.md).

## Specs

```python
env.state_spec          # nested, mirrors TensorDict keys
env.action_spec         # raises on renderer-only cells
env.observation_spec    # (H, W, 3) uint8
env.variation_spec      # flat style row
```

PushT's public state is grouped by entity — `state["agent", "position"]`, `state["agent", "velocity"]`, `state["block", "position"]`, `state["block", "angle"]` — not the packed float32 row the renderer uses internally. `env.pack(state)` / `env.unpack(rows)` convert; the packed layout is the Step-8 corpus layout, unchanged.

## Appearance

```python
env.variation_space.names()
pixels = env.render(state, variation={"block.color": [220, 40, 40]})
```

Defaults are the official appearance. Domain randomisation is a tensor, not a loop. See [variation spaces](concepts/variation-spaces.md).

## Logging

```python
snv.set(verbose="INFO")          # TRACE, DEBUG, INFO, WARNING, ERROR
cfg = snv.get_config()           # verbose, default_device, warn_on_deprecated
```

## Next

- [Environment listing](environments.md)
- [Lifecycle](concepts/lifecycle.md)
- [stable-worldmodel ownership](concepts/worldmodel.md)
- [Adding an environment](adding-an-env.md)
- Examples: `python examples/render.py --help`
