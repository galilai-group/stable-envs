# Quickstart

Install the core, construct an environment by **class name**, and render. Optional extras add families; they are not required to draw PushT.

```bash
pip install stable-envs
```

Family extras, pins, and the nvdiffrast source build: [Install](install.md). From a clone, `pip install -e .` — see [Contributing](https://github.com/galilai-group/stable-envs/blob/main/CONTRIBUTING.md).

Four nouns — **Env**, **State**, **Variation**, **Frame** — and two usage modes:

```python
import stable_envs as snv

env = snv.PushT(num_envs=64)                 # device="auto" → CUDA if available, else CPU
state = env.reset(seed=0)                    # State
pixels = env.render(state)                   # Frame: (64, H, W, 3) uint8
action = env.action_spec.sample(env.num_envs, env.rng)

out = env.step(action)                       # STATEFUL: writes env.state
state = out["state"]
nxt = env.transition(state, action)          # FUNCTIONAL: env.state is untouched
traj = env.unroll(state, action[None])       # known actions, time first
env.close()
```

`num_envs` is how many copies this object holds. `len(env) == env.num_envs`. A short last batch (`n <= num_envs`) is accepted by `set_state` / `render` and draws exactly those rows.

The same object as a context manager — `close()` is idempotent and correctness never depends on `__del__`:

```python
with snv.PushT(num_envs=8) as env:
    state = env.reset(seed=0)
    pixels = env.render(state)
```

## Strings are secondary

```python
env = snv.make("PushT", num_envs=8)
info = snv.info("PushT")          # metadata, no construction, no GPU — not a snv.Spec
names = snv.list_envs()           # canonical class names this interpreter imported
snv.list_envs(family="dmcontrol")
snv.list_families()
```

An unknown name raises with the registered class names. Strings are those class names, not sibling-library ids.

## Stateful step, functional dynamics

```python
state = env.reset(seed=0)
action = env.action_spec.sample(env.num_envs, env.rng)

out = env.step(action)                    # writes env.state; returns a StepResult TensorDict
state = out["state"]                      # always
# out["reward"] / out["terminated"] exist only where that env defines those semantics
nxt = env.transition(state, action)       # does not touch env.state
traj = env.unroll(state, actions)         # actions are (T, num_envs, …), time first
```

`unroll` is open-loop known actions, not a policy rollout. Cells without a batched simulator raise `NotImplementedError` from `step` / `transition` and keep `reset` / `render` working. See [dynamics](concepts/dynamics.md).

## Specs

```python
env.state_spec          # nested, mirrors TensorDict keys
env.action_spec         # raises on renderer-only cells
env.frame_spec          # (H, W, 3) uint8 — what render() produces
env.variation_spec      # flat Variation row
```

PushT's public state is grouped by entity — `state["agent", "position"]`, `state["agent", "velocity"]`, `state["block", "position"]`, `state["block", "angle"]` — not the compact render row. `env.pack_render_state(state)` / `env.unpack_render_state(rows)` convert; that packed layout is the frozen certification layout and **may omit dynamics-only fields**.

## Variation / configuration

```python
env.variation_spec                    # flat row contract
env.variation_dim, env.variation_names
row = env.default_variation()         # official configuration
rows = env.sample_variation(env.num_envs)
pixels = env.render(state, variation={"block.color": [220, 40, 40]})
pixels = env.render(state, variation=rows)
```

A mapping overrides named leaves; a `(variation_dim,)` or `(B, variation_dim)` row is the full
configuration. Defaults are the official picture. Leaves may be render-only, dynamics-only,
reset/task, or both. See [Variation](concepts/variation-spaces.md).

## Logging

```python
snv.set(verbose="INFO", default_device="auto")
cfg = snv.get_config()           # verbose, default_device
```

## Next

- [Environment listing](environments.md)
- [State, Variation, Frame](concepts/state.md)
- [Lifecycle](concepts/lifecycle.md)
- [Sibling projects](concepts/worldmodel.md)
- [Adding an environment](adding-an-env.md)
- Examples: `python examples/render.py --help`
