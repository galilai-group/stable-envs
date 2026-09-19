# Step 9 — Wave-0 contract (coordinator-frozen, 2026-09-17)

Binding for every Step-9 workstream. Derived from STEP_09_OSS_REFACTOR_AND_PRODUCTIZATION.md §2, §5, §6,
§11 and the inventory of the Step-8 tree at 21cb306. Changes to this file are coordinator-only.

## 1. Public namespace (`stable_envs/__init__.py`)
```python
import stable_envs as snv
snv.PushT, snv.CartPole, snv.DMCReacherHard, snv.OGBCube, ...   # concrete env classes, canonical
snv.make("swm/PushT-v1", device="cuda")                          # registry, secondary
snv.spec("PushT"); snv.list_envs()
snv.set(verbose="INFO"); snv.get_config()
snv.Env, snv.State (TensorDict alias), snv.Spec                   # the three vocabulary types
```
Nothing from `render/`, `testing/`, `interop.py` at top level. `snv.__version__` from `__about__.py`.
During Step 9 the old names (`StableEnv`, `REGISTRY`, `register`, `se.make(name, batch=...)`) keep
working as thin deprecated aliases so the ~470 existing test call sites keep running until Workstream
K migrates them; they are removed in Wave 3, never before the oracle comparison is green.

## 2. Constructor and batch/device semantics
```python
env = snv.PushT(num_envs=64, device="cuda", size=None, supersample=1, backend="auto", seed=0)
```
`num_envs` replaces `batch` (alias `batch=` accepted, deprecated). `env.num_envs`, `env.device`
(`torch.device`), `env.size`. `len(env) == num_envs`. Short batches (`n <= num_envs`) keep working
exactly as `set_state` documents today.

## 3. State
The public state is a `tensordict.TensorDict` with batch size `[num_envs]`, keys grouped by entity,
e.g. PushT `{"agent": {"position": (2,), "velocity": (2,)}, "block": {"position": (2,), "angle": (1,)}}`.
It describes the environment, not the renderer's packed row. The packed float32 row used by every
renderer, corpus and certification check is the INTERNAL layout: `env.pack(state) -> (n, state_dim)`
and `env.unpack(rows) -> State`; `env.state_names` is unchanged. **Hidden dynamics variables become
state**: PushT's agent velocity (today `_agent_v`) is `state["agent","velocity"]`, so `transition` is
pure. The render subset of the state is exactly the Step-8 packed row, so every frozen digest, corpus
and `check_state_roundtrip` is unchanged. Trivial envs may expose a single key (e.g. `{"x": (d,)}`)
rather than ceremony. `tensordict` becomes a core dependency (pure-torch, light).

## 4. Specs (`specs.py`)
One small `Spec` (dataclass: `shape`, `dtype`, `low`, `high`, `device`, `names`, `discrete_n`) plus
`CompositeSpec` = a nested dict of Specs mirroring the TensorDict keys, with `sample(num_envs, rng)`,
`validate(x)`, `zeros(num_envs)`, `to(device)`. No torchrl dependency, no class hierarchy per spec kind.
`env.state_spec`, `env.action_spec`, `env.observation_spec` (pixels: `(H, W, 3)` uint8),
`env.variation_spec` (the flat style layout; `variation_space` stays as the internal gymnasium-style
space object used by the certification harness).

## 5. Lifecycle and stepping
```python
state = env.reset(seed=0)                     # -> State; also sets the env's held state
out   = env.step(actions)                     # stateful; -> StepResult (TensorDict): state, obs?, reward?, done?
next_state = env.transition(state, action)    # functional, pure in `state`; no env buffer is touched
traj = env.unroll(state, actions)             # functional, open-loop known actions; actions (T, num_envs, ...)
pixels = env.render(state | packed rows)      # unchanged semantics; accepts State or packed rows
env.close(); with snv.PushT(...) as env: ...  # explicit resources; correctness never depends on __del__
```
`step` returns a TensorDict `{"state": State, "reward": (num_envs,) float32 or absent, "done":
(num_envs,) bool or absent}`; envs that have no reward/termination semantics (pure renderers) omit
them rather than fabricating zeros. `step` advances `env.state`; `transition`/`unroll` never do.
Envs without dynamics (renderer-only cells) raise `NotImplementedError("PushT has dynamics; X does
not")`-style errors from `step`/`transition` with the exact reason; `reset`/`render` always work.
No `env.rollout(policy)`: `rollout.py` (policies, sample) moves out of the package core; keep it only
as an example under `examples/` or in `interop.py` if SWM needs it.

## 6. Registry (`registry.py`)
`make(name, **kw)`, `spec(name)`, `list_envs(family=None)`; decorators/declarations register a class
with its canonical name and its SWM id aliases (from the Step-8 registry). Classes are canonical; the
registry holds no semantics. Auto-discovery of `stable_envs/envs/*` stays (import errors recorded,
`snv.import_errors()`), so a missing optional dependency costs exactly that family.

## 7. Config/logging (`_config.py`)
`snv.set(verbose=...)`, `snv.get_config()` -> dict; Loguru logger `stable_envs` (add `loguru` as a
core dependency); the only config keys are `verbose`, `default_device`, `warn_on_deprecated`.

## 8. Dependency direction
`env.py`, `dynamics.py`, `specs.py`, `registry.py`, `_config.py` import nothing from `render/`,
`envs/`, `interop.py`. `render/*` never imports `envs/*` or SWM. `envs/*` import core + render.
`interop.py` (renamed from `compat/worldmodel.py` + `compat/variation_maps/`) imports public
stable-envs concepts and, lazily, `stable_worldmodel`/`gymnasium`. Circularity is a review BLOCK.

## 9. Tests
pytest markers `unit`, `integration`, `gpu`, `certification`, `slow` (existing `reference`, `smoke`,
`selftest` kept until Workstream K retires them). New foundation tests live in `tests/unit/`.

## 10. Oracle rule for Wave 0
PushT's packed rows, default frame digest, `sample_states`, `set_state`, `render`, style layout and
the Step-8 support-matrix row must be byte-identical before/after. `transition(state, a)` must equal
today's `step(a)` (graph or eager) to the bit on the same device for the same `state`; `unroll`
equals repeated `transition`. Measured with a local CPU test and one GPU check (PushT is a CPU-capable
cell: run both).

## Coordinator addendum (2026-09-19)

v0 keeps the deprecated aliases (`StableEnv`, `REGISTRY`, `register`, `batch=`, `core`/`compat` shims)
importable and unadvertised. `stable_envs.testing` stays the public measurement library.
`stable_envs.rollout` stays (`snv.sample` / `BasePolicy`; no `env.rollout(policy)`). There is no `[all]`
extra. Native `swm.World(env=snv.Env)` is unsupported by the installed `stable-worldmodel`.
